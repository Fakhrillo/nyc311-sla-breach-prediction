"""GOV-02 Public Service SLA Breach Prediction (NYC 311).

Single module: Socrata download -> causal workload features -> target definition
-> models -> evaluation -> saved artifacts -> single-case inference.

The city publishes a `due_date` column but fills it for under 1% of records, so
the service window has to be defined here. See fit_sla().
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nyc311_q1_2024.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"

ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
WINDOW = ("2024-01-01", "2024-03-31")

# The 12 request types that make up ~60% of Q1 2024 volume. Scoping to them keeps
# the per-type service windows well estimated instead of fitting a percentile to
# a handful of cases, and still spans four agencies with very different work.
SCOPE_TYPES = [
    "Illegal Parking",
    "HEAT/HOT WATER",
    "Noise - Residential",
    "Blocked Driveway",
    "UNSANITARY CONDITION",
    "Street Condition",
    "Abandoned Vehicle",
    "PLUMBING",
    "PAINT/PLASTER",
    "Noise - Street/Sidewalk",
    "Noise - Commercial",
    "WATER LEAK",
]

PULL_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "complaint_type",
    "descriptor",
    "borough",
    "incident_zip",
    "open_data_channel_type",
    "location_type",
    "address_type",
    "status",
]

# Known at intake. Nothing here depends on how the case was eventually handled.
NUMERIC = ["created_hour", "agency_backlog", "agency_7d_volume", "type_7d_volume", "sla_hours"]
BINARY = ["is_weekend", "has_zip"]
CATEGORICAL = [
    "complaint_type",
    "agency",
    "borough",
    "open_data_channel_type",
    "location_type",
    "address_type",
    "descriptor",
    "created_dow",
]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Present in the raw file but describing the outcome, not the intake.
LEAKY_COLUMNS = {
    "closed_date": "the resolution timestamp the target is computed from",
    "status": "reflects whether the case is already resolved",
    "resolution_description": "written by staff when the case is closed",
    "resolution_action_updated_date": "the last time the outcome changed",
}

# A handful of closures are logged seconds after intake (auto-closed duplicates)
# and some predate their own creation. Both are recording errors, not fast work.
MIN_RESOLUTION_HOURS = 0.02


# --------------------------------------------------------------------------- data


def fetch(path: Path | str = DATA, chunk_days: int = 3, retries: int = 4) -> Path:
    """Download the scoped slice from the NYC Open Data API.

    Paged by date window rather than `$offset`. Socrata times out on any `$order`
    over a range this size, and unordered offset paging is not safe -- without a
    sort the server may repeat or skip rows between pages. Consecutive date
    windows are indexed, fast, and provably cover the period exactly once.

    Request types are filtered here rather than server-side: an `IN(...)` clause
    over twelve types makes Socrata scan the table and time out.
    """
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    scope = {t.upper() for t in SCOPE_TYPES}
    edges = pd.date_range(WINDOW[0], pd.Timestamp(WINDOW[1]) + pd.Timedelta(days=1),
                          freq=f"{chunk_days}D")
    frames, scanned = [], 0

    for lo, hi in zip(edges[:-1], edges[1:]):
        query = urllib.parse.urlencode(
            {
                "$select": ",".join(PULL_COLUMNS),
                "$where": f"created_date >= '{lo:%Y-%m-%d}' AND created_date < '{hi:%Y-%m-%d}'",
                "$limit": 50_000,
            }
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=180) as resp:
                    chunk = pd.read_csv(resp)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == retries - 1:
                    raise RuntimeError(f"NYC 311 API failed for {lo:%Y-%m-%d}: {exc}") from exc
                time.sleep(2 ** attempt)

        if len(chunk) >= 50_000:
            raise RuntimeError(
                f"{lo:%Y-%m-%d} hit the 50k row cap -- lower chunk_days or rows are being lost"
            )
        scanned += len(chunk)
        frames.append(chunk[chunk["complaint_type"].str.upper().isin(scope)])
        print(f"  {lo:%Y-%m-%d} | scanned {scanned:,} kept {sum(map(len, frames)):,}", end="\r")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False)
    print(f"  scanned {scanned:,} rows, kept {len(out):,} -> {path.name}" + " " * 20)
    return path


def load_raw(path: Path | str = DATA) -> pd.DataFrame:
    return pd.read_csv(fetch(path), low_memory=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps, compute elapsed time, drop impossible records."""
    df = raw.copy()
    for col in ("created_date", "closed_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    before = len(df)
    df = df[df["created_date"].notna()].drop_duplicates(subset="unique_key")

    df["resolution_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    # A closure before its own intake is a recording error, not a negative duration.
    df.loc[df["resolution_hours"] < 0, "resolution_hours"] = np.nan
    df.loc[df["resolution_hours"] < MIN_RESOLUTION_HOURS, "resolution_hours"] = np.nan

    for col in ("agency", "complaint_type", "borough", "open_data_channel_type",
                "location_type", "address_type", "descriptor"):
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    df["has_zip"] = df["incident_zip"].notna().astype(int)
    df.attrs["dropped_rows"] = before - len(df)
    return df.sort_values(["created_date", "unique_key"]).reset_index(drop=True)


def add_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Backlog and recent intake volume, as known at the moment each case arrives.

    `agency_backlog` counts the agency's cases that were created before this one
    and had not yet closed when it arrived. That uses other cases' closing times
    only to ask "were you still open at time T", which the queue genuinely knows
    at T -- it never looks at whether a case closes after T.
    """
    df = df.copy()
    created = df["created_date"].to_numpy()
    backlog = np.zeros(len(df), dtype=np.int32)
    vol_agency = np.zeros(len(df), dtype=np.int32)
    vol_type = np.zeros(len(df), dtype=np.int32)
    week = np.timedelta64(7, "D")

    for _, grp in df.groupby("agency", sort=False):
        pos = grp.index.to_numpy()
        opened = np.sort(grp["created_date"].to_numpy())
        closed = np.sort(grp["closed_date"].dropna().to_numpy())
        at = created[pos]
        n_opened = np.searchsorted(opened, at, side="left")
        n_closed = np.searchsorted(closed, at, side="left")
        backlog[pos] = n_opened - n_closed
        vol_agency[pos] = n_opened - np.searchsorted(opened, at - week, side="left")

    for _, grp in df.groupby("complaint_type", sort=False):
        pos = grp.index.to_numpy()
        opened = np.sort(grp["created_date"].to_numpy())
        at = created[pos]
        vol_type[pos] = np.searchsorted(opened, at, side="left") - np.searchsorted(
            opened, at - week, side="left"
        )

    df["agency_backlog"] = backlog
    df["agency_7d_volume"] = vol_agency
    df["type_7d_volume"] = vol_type
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_hour"] = df["created_date"].dt.hour
    df["created_dow"] = df["created_date"].dt.dayofweek.astype(str)
    df["is_weekend"] = (df["created_date"].dt.dayofweek >= 5).astype(int)
    return df


def build_dataset(path: Path | str = DATA) -> pd.DataFrame:
    return add_calendar(add_workload(clean(load_raw(path))))


def time_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """Chronological split on intake date, cutting on whole days.

    Cases arrive continuously, so the deployment question is always "given what
    the queue looked like up to today, which of today's arrivals will run long".
    A random split would train on next month's cases to score last month's, and
    would also smear the backlog features across the boundary.
    """
    day = df["created_date"].dt.normalize()
    share = day.value_counts().sort_index().cumsum() / len(df)
    val_start = share.index[int((share >= 1 - val_frac - test_frac).argmax())]
    test_start = share.index[int((share >= 1 - test_frac).argmax())]
    return (
        df[day < val_start].copy(),
        df[(day >= val_start) & (day < test_start)].copy(),
        df[day >= test_start].copy(),
    )


# ------------------------------------------------------------------------- target


def fit_sla(train: pd.DataFrame, quantile: float = 0.75, min_cases: int = 200) -> dict:
    """Define the service window per request type, from training cases only.

    The city's own `due_date` is unusable (<1% populated), so the window is the
    time by which 3 in 4 cases of that type were historically resolved. Per type
    rather than one global deadline: a noise call and a street repair are not the
    same promise, and a single threshold would reduce the task to guessing the
    complaint type.

    Fitting on train only matters -- thresholds derived from the full data would
    encode the test period's outcomes into the labels.
    """
    done = train.loc[train["resolution_hours"].notna()]
    overall = float(done["resolution_hours"].quantile(quantile))
    per_type = done.groupby("complaint_type")["resolution_hours"].agg(["quantile", "size"])
    sla = {
        t: float(done.loc[done["complaint_type"] == t, "resolution_hours"].quantile(quantile))
        for t, row in per_type.iterrows()
        if row["size"] >= min_cases
    }
    return {"per_type": sla, "default": overall, "quantile": quantile}


def apply_sla(df: pd.DataFrame, sla: dict, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Label each case as a breach, dropping only the genuinely unknown ones.

    A case still open at the snapshot is not automatically unlabelled: if it has
    already been open longer than its window it has breached, whatever happens
    next. Only a case that is still open *and* still inside its window is unknown,
    and those are dropped. Silently dropping every open case instead would bias
    the data against exactly the slow cases the client cares about.
    """
    df = df.copy()
    snapshot = snapshot or df["created_date"].max()
    df["sla_hours"] = df["complaint_type"].map(sla["per_type"]).fillna(sla["default"])

    elapsed = (snapshot - df["created_date"]).dt.total_seconds() / 3600
    closed = df["resolution_hours"].notna()

    y = pd.Series(np.nan, index=df.index)
    y[closed] = (df.loc[closed, "resolution_hours"] > df.loc[closed, "sla_hours"]).astype(float)
    already = ~closed & (elapsed > df["sla_hours"])
    y[already] = 1.0

    df["y"] = y
    df.attrs["censored_dropped"] = int(y.isna().sum())
    return df[y.notna()].assign(y=lambda d: d["y"].astype(int))


# ------------------------------------------------------------------------- models


def make_model(name: str, **kwargs):
    """Return an unfitted pipeline. `name` is one of the keys below."""
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    if name == "baseline_prior":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))])

    if name == "baseline_type":
        # Stronger, more honest baseline: the historical breach rate of the
        # request type alone. Beating the prior is trivial; beating this means
        # the model found something beyond "what kind of case is this".
        return Pipeline([("clf", _TypeRateBaseline())])

    if name == "logreg":
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("sc", StandardScaler())]), NUMERIC),
                ("bin", "passthrough", BINARY),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
            ]
        )
        return Pipeline([("pre", pre),
                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", **kwargs))])

    if name in ("hgb", "rf"):
        pre = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), NUMERIC),
                ("bin", "passthrough", BINARY),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                 CATEGORICAL),
            ]
        )
        cat_mask = [False] * (len(NUMERIC) + len(BINARY)) + [True] * len(CATEGORICAL)
        if name == "hgb":
            clf = HistGradientBoostingClassifier(categorical_features=cat_mask,
                                                 random_state=0, **kwargs)
        else:
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=25, n_jobs=-1,
                                         random_state=0, **kwargs)
        return Pipeline([("pre", pre), ("clf", clf)])

    raise ValueError(f"unknown model {name!r}")


class _TypeRateBaseline(BaseEstimator, ClassifierMixin):
    """Predicts each request type's historical breach rate. No learning beyond a mean."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.global_ = float(y.mean())
        self.rates_ = pd.Series(y).groupby(np.asarray(X["complaint_type"])).mean().to_dict()
        return self

    def predict_proba(self, X):
        p = np.array([self.rates_.get(t, self.global_) for t in X["complaint_type"]])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------- evaluation


def recall_at_capacity(y_true, scores, capacity: float = 0.20) -> float:
    """Share of real breaches caught if only `capacity` of arrivals can be expedited."""
    y_true = np.asarray(y_true)
    k = max(1, int(len(scores) * capacity))
    top = np.argsort(np.asarray(scores))[::-1][:k]
    total = y_true.sum()
    return float(y_true[top].sum() / total) if total else float("nan")


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    y_true, scores = np.asarray(y_true), np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "pr_auc": float(average_precision_score(y_true, scores)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "recall_at_20pct": recall_at_capacity(y_true, scores, 0.20),
            "recall_at_10pct": recall_at_capacity(y_true, scores, 0.10),
            "brier": float(brier_score_loss(y_true, scores)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "threshold": float(threshold),
            "positive_rate": float(y_true.mean()),
            "n": int(len(y_true)),
        }


def threshold_at_capacity(scores, capacity: float = 0.20) -> float:
    """Score cutoff that flags exactly `capacity` of arrivals -- the shipped rule.

    Expediting a case costs supervisor attention that exists in fixed supply, so
    the budget comes first and the question is what it buys.
    """
    return float(np.quantile(np.asarray(scores), 1 - capacity))


# ---------------------------------------------------------------------- inference

REQUIRED_FIELDS = {"created_date", "complaint_type", "agency", "borough"}
OPTIONAL_DEFAULTS = {
    "descriptor": "UNKNOWN",
    "open_data_channel_type": "UNKNOWN",
    "location_type": "UNKNOWN",
    "address_type": "UNKNOWN",
    "incident_zip": None,
    "agency_backlog": 0,
    "agency_7d_volume": 0,
    "type_7d_volume": 0,
}


def validate_record(record: dict, sla: dict) -> dict:
    """Check one intake record and return it as a clean feature row.

    Raises ValueError with a message a caller can show to a user.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"missing required field(s): {sorted(missing)}")

    r = {**OPTIONAL_DEFAULTS, **record}
    try:
        created = pd.to_datetime(r["created_date"])
    except Exception as exc:
        raise ValueError(f"unparseable created_date: {exc}") from exc
    if pd.isna(created):
        raise ValueError("created_date must be a valid timestamp")

    for col in ("agency_backlog", "agency_7d_volume", "type_7d_volume"):
        if not isinstance(r[col], (int, float)) or isinstance(r[col], bool) or r[col] < 0:
            raise ValueError(f"{col} must be a number >= 0")

    ctype = str(r["complaint_type"]).strip().upper()
    if not ctype or ctype == "UNKNOWN":
        raise ValueError("complaint_type is required and cannot be UNKNOWN")

    return {
        "created_hour": float(created.hour),
        "agency_backlog": float(r["agency_backlog"]),
        "agency_7d_volume": float(r["agency_7d_volume"]),
        "type_7d_volume": float(r["type_7d_volume"]),
        "sla_hours": float(sla["per_type"].get(ctype, sla["default"])),
        "is_weekend": int(created.dayofweek >= 5),
        "has_zip": int(r["incident_zip"] is not None),
        "complaint_type": ctype,
        "agency": str(r["agency"]).strip().upper(),
        "borough": str(r["borough"]).strip().upper(),
        "open_data_channel_type": str(r["open_data_channel_type"]).strip().upper(),
        "location_type": str(r["location_type"]).strip().upper(),
        "address_type": str(r["address_type"]).strip().upper(),
        "descriptor": str(r["descriptor"]).strip().upper(),
        "created_dow": str(created.dayofweek),
    }


def save_model(pipeline, threshold: float, sla: dict, metrics: dict,
               path: Path = MODEL_PATH) -> Path:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold, "sla": sla,
                 "features": FEATURES, "metrics": metrics}, path)
    (path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return path


def load_model(path: Path = MODEL_PATH) -> dict:
    import joblib

    if not Path(path).exists():
        raise FileNotFoundError(f"no model at {path}; run `python train.py` first")
    return joblib.load(path)


def predict_one(record: dict, bundle: dict | None = None) -> dict:
    """Score one incoming case. Returns breach risk plus the window it is judged against."""
    bundle = bundle or load_model()
    row = validate_record(record, bundle["sla"])
    frame = pd.DataFrame([row])[bundle["features"]]
    prob = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    return {
        "breach_probability": round(prob, 4),
        "service_window_hours": round(row["sla_hours"], 1),
        "risk_band": "high" if prob >= bundle["threshold"]
        else ("medium" if prob >= 0.5 * bundle["threshold"] else "low"),
        "flagged_for_expediting": prob >= bundle["threshold"],
        "threshold": bundle["threshold"],
    }
