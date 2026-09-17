"""Run with: python test_sla.py

Guards the three things that would silently invalidate this project: a backlog
feature that peeks at the future, service windows fitted on data the model is
scored against, and censored cases labelled as if they were resolved.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import sla as S


def _frame(rows):
    df = pd.DataFrame(rows)
    for c in ("created_date", "closed_date"):
        df[c] = pd.to_datetime(df.get(c), format="mixed")
    df["resolution_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    return df.sort_values("created_date").reset_index(drop=True)


def test_backlog_counts_only_what_was_open_at_arrival():
    df = _frame([
        {"agency": "A", "complaint_type": "T", "created_date": "2024-01-01", "closed_date": "2024-01-05"},
        {"agency": "A", "complaint_type": "T", "created_date": "2024-01-03", "closed_date": "2024-01-10"},
        {"agency": "A", "complaint_type": "T", "created_date": "2024-01-07", "closed_date": "2024-01-08"},
    ])
    out = S.add_workload(df)
    # Nothing exists before the first case.
    assert out.loc[0, "agency_backlog"] == 0, out.loc[0, "agency_backlog"]
    # At 01-03 only case 1 is open.
    assert out.loc[1, "agency_backlog"] == 1, out.loc[1, "agency_backlog"]
    # At 01-07 case 1 has closed but case 2 is still open. Case 2 closes on 01-10,
    # which is in the future here, and must not retroactively empty the queue.
    # This assertion is the whole reason this file exists.
    assert out.loc[2, "agency_backlog"] == 1, out.loc[2, "agency_backlog"]
    # A case never counts itself.
    assert (out["agency_backlog"] <= np.arange(len(out))).all()


def test_backlog_is_per_agency():
    df = _frame([
        {"agency": "A", "complaint_type": "T", "created_date": "2024-01-01", "closed_date": "2024-02-01"},
        {"agency": "B", "complaint_type": "T", "created_date": "2024-01-02", "closed_date": "2024-02-01"},
    ])
    out = S.add_workload(df)
    assert out.loc[1, "agency_backlog"] == 0, "another agency's queue is not this agency's backlog"


def test_sla_fitted_on_train_only():
    train = _frame([
        {"agency": "A", "complaint_type": "FAST", "created_date": f"2024-01-{d:02d}",
         "closed_date": f"2024-01-{d:02d} 01:00"} for d in range(1, 11)
    ])
    sla = S.fit_sla(train, min_cases=5)
    assert "FAST" in sla["per_type"]
    assert 0.9 < sla["per_type"]["FAST"] < 1.1, sla["per_type"]["FAST"]
    # A type the training period never saw falls back to the global window.
    assert sla["per_type"].get("NEVER-SEEN") is None


def test_rare_types_fall_back_to_default():
    train = _frame(
        [{"agency": "A", "complaint_type": "COMMON", "created_date": f"2024-01-{d:02d}",
          "closed_date": f"2024-01-{d:02d} 02:00"} for d in range(1, 11)]
        + [{"agency": "A", "complaint_type": "RARE", "created_date": "2024-01-05",
            "closed_date": "2024-01-25"}]
    )
    sla = S.fit_sla(train, min_cases=5)
    assert "RARE" not in sla["per_type"], "a window fitted to one case is not a window"
    labelled = S.apply_sla(train, sla, snapshot=pd.Timestamp("2024-02-01"))
    assert (labelled.loc[labelled.complaint_type == "RARE", "sla_hours"] == sla["default"]).all()


def test_open_case_past_its_window_is_a_breach():
    sla = {"per_type": {"T": 24.0}, "default": 24.0, "quantile": 0.75}
    df = _frame([
        # closed inside the window -> 0
        {"agency": "A", "complaint_type": "T", "created_date": "2024-01-01", "closed_date": "2024-01-01 06:00"},
        # closed outside the window -> 1
        {"agency": "A", "complaint_type": "T", "created_date": "2024-01-02", "closed_date": "2024-01-05"},
        # still open, already 10 days past a 24h window -> known breach, keep it
        {"agency": "A", "complaint_type": "T", "created_date": "2024-01-03", "closed_date": None},
        # still open, but only 2h elapsed at snapshot -> genuinely unknown, drop it
        {"agency": "A", "complaint_type": "T", "created_date": "2024-01-20 22:00", "closed_date": None},
    ])
    out = S.apply_sla(df, sla, snapshot=pd.Timestamp("2024-01-21"))
    assert len(out) == 3, "only the still-inside-window open case should be dropped"
    assert out.attrs["censored_dropped"] == 1
    assert list(out["y"]) == [0, 1, 1], list(out["y"])


def test_split_is_chronological_and_whole_days():
    df = _frame([
        {"agency": "A", "complaint_type": "T", "created_date": f"2024-01-{d:02d} {h:02d}:00",
         "closed_date": f"2024-02-{d:02d}"}
        for d in range(1, 21) for h in (9, 15)
    ])
    tr, va, te = S.time_split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["created_date"].max() < va["created_date"].min()
    assert va["created_date"].max() < te["created_date"].min()
    days = lambda d: set(d["created_date"].dt.normalize())
    assert not days(tr) & days(va) and not days(va) & days(te), "a day must not straddle two sets"


def test_outcome_columns_are_not_features():
    for col in S.LEAKY_COLUMNS:
        assert col not in S.FEATURES, col
    assert "resolution_hours" not in S.FEATURES
    assert "closed_date" in S.LEAKY_COLUMNS


def test_recall_at_capacity():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    good = np.array([.9, .8, .1, .1, .1, .1, .1, .1, .1, .1])
    assert S.recall_at_capacity(y, good, 0.20) == 1.0
    assert S.recall_at_capacity(y, -good, 0.20) == 0.0


def test_threshold_at_capacity_flags_that_share():
    scores = np.linspace(0, 1, 1000)
    flagged = (scores >= S.threshold_at_capacity(scores, 0.20)).mean()
    assert abs(flagged - 0.20) < 0.01, flagged


def test_clean_rejects_impossible_durations():
    raw = pd.DataFrame({
        "unique_key": [1, 2, 3],
        "created_date": ["2024-01-02 10:00", "2024-01-02 10:00", "2024-01-02 10:00"],
        "closed_date": ["2024-01-03 10:00", "2024-01-01 10:00", "2024-01-02 10:00:10"],
        "agency": ["A", "A", "A"], "complaint_type": ["t", "t", "t"], "descriptor": ["d", "d", "d"],
        "borough": ["BX", "BX", "BX"], "incident_zip": [10001, None, 10001],
        "open_data_channel_type": ["ONLINE"] * 3, "location_type": ["x"] * 3,
        "address_type": ["ADDRESS"] * 3, "status": ["Closed"] * 3,
    })
    out = S.clean(raw)
    assert out.loc[0, "resolution_hours"] == 24
    assert pd.isna(out.loc[1, "resolution_hours"]), "closure before intake is an error, not -24h"
    assert pd.isna(out.loc[2, "resolution_hours"]), "a 10-second closure is an auto-close artefact"
    assert out.loc[0, "has_zip"] == 1 and out.loc[1, "has_zip"] == 0
    assert out.loc[0, "complaint_type"] == "T", "categoricals are normalised to upper case"


def test_validate_record():
    sla = {"per_type": {"ILLEGAL PARKING": 2.8}, "default": 41.7, "quantile": 0.75}
    ok = {"created_date": "2024-03-20T14:30:00", "complaint_type": "Illegal Parking",
          "agency": "NYPD", "borough": "BROOKLYN"}
    row = S.validate_record(ok, sla)
    assert row["created_hour"] == 14 and row["is_weekend"] == 0
    assert row["sla_hours"] == 2.8, "the window comes from the fitted SLA, not the caller"
    assert row["has_zip"] == 0 and row["complaint_type"] == "ILLEGAL PARKING"

    unknown = S.validate_record({**ok, "complaint_type": "Some New Type"}, sla)
    assert unknown["sla_hours"] == 41.7, "unseen types fall back to the default window"

    for bad, why in [
        ({k: v for k, v in ok.items() if k != "agency"}, "missing field"),
        ({**ok, "created_date": "not-a-date"}, "bad timestamp"),
        ({**ok, "agency_backlog": -5}, "negative backlog"),
        ({**ok, "agency_backlog": "many"}, "non-numeric backlog"),
        ({**ok, "complaint_type": "  "}, "blank complaint type"),
    ]:
        try:
            S.validate_record(bad, sla)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have rejected: {why}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
