"""Train, compare and save the SLA breach model.

    python train.py              # full run: experiments -> final model -> test report
    python train.py --quick      # fewer configs, for a smoke test

Writes artifacts/model.joblib, metrics.json, runs.csv and error_analysis.md.
Logs to MLflow when it is installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import sla as S  # noqa: E402

RUNS_CSV = S.ARTIFACTS / "runs.csv"

# Share of incoming cases a supervisor can realistically expedite.
CAPACITY = 0.20

EXPERIMENTS = {
    "baseline_prior": [{}],
    "baseline_type": [{}],
    "logreg": [{"C": 1.0}],
    "hgb": [
        {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31},
        {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 50},
        {"max_iter": 600, "learning_rate": 0.05, "max_leaf_nodes": 63, "l2_regularization": 1.0},
    ],
    "rf": [{}],
}
QUICK = {"baseline_prior": [{}], "baseline_type": [{}], "hgb": [EXPERIMENTS["hgb"][0]]}


def _mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri(f"file:{S.ROOT / 'mlruns'}")
        mlflow.set_experiment("gov02-sla-breach")
        return mlflow
    except Exception:
        return None


def score(pipe, X):
    return pipe.predict_proba(X)[:, 1]


def run_experiments(train, val, configs, mlf) -> pd.DataFrame:
    rows = []
    Xtr, ytr = train[S.FEATURES], train["y"]
    Xva, yva = val[S.FEATURES], val["y"]
    for name, grid in configs.items():
        for cfg in grid:
            t0 = time.time()
            pipe = S.make_model(name, **cfg).fit(Xtr, ytr)
            m = S.evaluate(yva, score(pipe, Xva))
            row = {"model": name, "params": json.dumps(cfg, sort_keys=True),
                   "fit_seconds": round(time.time() - t0, 1), **m}
            rows.append(row)
            print(f"  {name:<16} {row['params'][:44]:<44} "
                  f"PR-AUC {m['pr_auc']:.4f}  ROC {m['roc_auc']:.4f}  "
                  f"R@20% {m['recall_at_20pct']:.3f}  ({row['fit_seconds']}s)")
            if mlf:
                with mlf.start_run(run_name=f"{name}-{row['params'][:30]}"):
                    mlf.log_params({"model": name, **cfg})
                    mlf.log_metrics({k: v for k, v in m.items() if isinstance(v, float)})
    return pd.DataFrame(rows)


def split_ablation(df: pd.DataFrame, sla_def: dict, cfg: dict) -> str:
    """What a random split would have reported, versus the chronological one.

    The brief asks when the prediction is made. A random split answers "some time
    never" -- it trains on March to score January, and smears the backlog features
    across the boundary. It is the single easiest way to overstate this project.
    """
    from sklearn.model_selection import train_test_split

    labelled = S.apply_sla(df, sla_def, snapshot=df["created_date"].max())
    rtr, rva = train_test_split(labelled, test_size=0.15, random_state=0, shuffle=True)
    pipe = S.make_model("hgb", **cfg).fit(rtr[S.FEATURES], rtr["y"])
    rnd = S.evaluate(rva["y"], score(pipe, rva[S.FEATURES]))
    return f"random split PR-AUC {rnd['pr_auc']:.4f} / ROC {rnd['roc_auc']:.4f}"


def error_analysis(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> str:
    df = test.copy()
    df["score"] = scores
    df["pred"] = (scores >= threshold).astype(int)

    fn = df[(df["y"] == 1) & (df["pred"] == 0)]
    fp = df[(df["y"] == 0) & (df["pred"] == 1)]
    lines = [
        "# Error analysis (test set, unseen)",
        "",
        f"- Test cases: {len(df):,} | actual breach rate: {df['y'].mean():.3f}",
        f"- Threshold: {threshold:.3f} | flagged: {df['pred'].mean():.3f} of arrivals",
        f"- Missed breaches (FN): {len(fn):,} | false alarms (FP): {len(fp):,}",
        "",
        "## Performance by slice",
        "",
        "| slice | n | breach rate | recall | precision |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    def row(label, mask):
        sub = df[mask]
        if len(sub) < 200:
            return None
        rec = sub.loc[sub["y"] == 1, "pred"].mean() if (sub["y"] == 1).any() else float("nan")
        prec = sub.loc[sub["pred"] == 1, "y"].mean() if (sub["pred"] == 1).any() else float("nan")
        return f"| {label} | {len(sub):,} | {sub['y'].mean():.3f} | {rec:.3f} | {prec:.3f} |"

    q = df["agency_backlog"].quantile([.25, .5, .75]).to_list()
    buckets = (
        [(f"agency: {a}", df["agency"] == a) for a in df["agency"].value_counts().index[:4]]
        + [(f"borough: {b}", df["borough"] == b) for b in df["borough"].value_counts().index[:6]]
        + [(f"channel: {c}", df["open_data_channel_type"] == c)
           for c in df["open_data_channel_type"].value_counts().index[:4]]
        + [
            (f"backlog < {q[0]:.0f}", df["agency_backlog"] < q[0]),
            (f"backlog {q[0]:.0f}-{q[2]:.0f}", df["agency_backlog"].between(q[0], q[2])),
            (f"backlog > {q[2]:.0f}", df["agency_backlog"] > q[2]),
            ("weekend arrival", df["is_weekend"] == 1),
            ("weekday arrival", df["is_weekend"] == 0),
            ("overnight (00-06h)", df["created_hour"] < 6),
            ("business hours (09-17h)", df["created_hour"].between(9, 17)),
        ]
    )
    lines += [r for r in (row(l, m) for l, m in buckets) if r]

    # Fairness is the brief's explicit ask: report the spread, do not assert parity.
    groups = {f"borough: {b}": df["borough"] == b for b in df["borough"].value_counts().index[:5]}
    groups |= {f"channel: {c}": df["open_data_channel_type"] == c
               for c in df["open_data_channel_type"].value_counts().index[:3]}
    recalls = {k: df.loc[m & (df["y"] == 1), "pred"].mean()
               for k, m in groups.items() if (m & (df["y"] == 1)).sum() >= 200}
    lo, hi = min(recalls, key=recalls.get), max(recalls, key=recalls.get)

    lines += [
        "",
        "## Fairness check",
        "",
        f"Recall across boroughs and intake channels ranges from **{recalls[lo]:.3f}** ({lo}) to "
        f"**{recalls[hi]:.3f}** ({hi}), a gap of {recalls[hi] - recalls[lo]:.3f}.",
        "",
        "This matters more here than in a commercial setting. Expediting is a public "
        "service being rationed, so a model that systematically surfaces cases from one "
        f"borough over another redistributes municipal attention along geographic lines. "
        f"Residents of **{lo}** would see their slow cases escalated least often, and "
        "nothing in the data tells us their cases are less urgent -- only that the "
        "historical queue treated them differently, which the model then learns to repeat.",
        "",
        "Intake channel carries the same risk in a different shape: if phone reports are "
        "escalated less than online ones, the system quietly penalises whoever is less "
        "likely to file online.",
        "",
        "## Hardest cases",
        "",
        f"- False alarms: median backlog {fp['agency_backlog'].median():.0f}, "
        f"median window {fp['sla_hours'].median():.1f} h",
        f"- Missed breaches: median backlog {fn['agency_backlog'].median():.0f}, "
        f"median window {fn['sla_hours'].median():.1f} h",
        "",
        "The target measures whether a case took longer than its type usually takes. It "
        "does not measure whether the resolution was any good, nor whether the case "
        "mattered. A fast closure and a good outcome are not the same event, and this "
        "model only ever sees the first.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=str(S.DATA))
    args = ap.parse_args()

    print("Loading NYC 311 and building features...")
    df = S.build_dataset(args.data)
    snapshot = df["created_date"].max()
    print(f"  {len(df):,} cases | {df['created_date'].min():%Y-%m-%d} -> {snapshot:%Y-%m-%d}")

    train, val, test = S.time_split(df)
    sla_def = S.fit_sla(train)          # windows come from the training period only
    train = S.apply_sla(train, sla_def, snapshot)
    val = S.apply_sla(val, sla_def, snapshot)
    test = S.apply_sla(test, sla_def, snapshot)
    print(f"  train {len(train):,} ({train.y.mean():.3f}) | val {len(val):,} ({val.y.mean():.3f})"
          f" | test {len(test):,} ({test.y.mean():.3f})  <- breach rates")
    print(f"  service windows: {len(sla_def['per_type'])} types, "
          f"{min(sla_def['per_type'].values()):.1f}h to {max(sla_def['per_type'].values()):.1f}h")

    mlf = _mlflow()
    print(f"\nExperiments (MLflow {'on' if mlf else 'off - runs.csv only'}):")
    runs = run_experiments(train, val, QUICK if args.quick else EXPERIMENTS, mlf)
    S.ARTIFACTS.mkdir(exist_ok=True)
    runs.to_csv(RUNS_CSV, index=False)

    # Rank well and stay calibrated: supervisors see a probability, so a model
    # scoring worse than the no-skill prior on Brier is not shippable.
    no_skill = float(runs.loc[runs["model"] == "baseline_prior", "brier"].iloc[0])
    real = runs[~runs["model"].str.startswith("baseline")]
    for _, r in real[real["brier"] > no_skill].iterrows():
        print(f"  rejected {r['model']} (PR-AUC {r['pr_auc']:.4f}): Brier {r['brier']:.4f} "
              f"> no-skill {no_skill:.4f}, probabilities not trustworthy")
    eligible = real[real["brier"] <= no_skill]
    if eligible.empty:
        eligible = real
    best = eligible.loc[eligible["pr_auc"].idxmax()]
    type_base = float(runs.loc[runs["model"] == "baseline_type", "pr_auc"].iloc[0])
    print(f"\nBest on validation: {best['model']} {best['params']} (PR-AUC {best['pr_auc']:.4f}, "
          f"vs {type_base:.4f} for request type alone)")

    final = S.make_model(best["model"], **json.loads(best["params"]))
    trainval = pd.concat([train, val])
    final.fit(trainval[S.FEATURES], trainval["y"])

    tuner = S.make_model(best["model"], **json.loads(best["params"])).fit(
        train[S.FEATURES], train["y"])
    threshold = S.threshold_at_capacity(score(tuner, val[S.FEATURES]), CAPACITY)
    test_scores = score(final, test[S.FEATURES])
    test_metrics = S.evaluate(test["y"], test_scores, threshold)

    base = S.make_model("baseline_type").fit(trainval[S.FEATURES], trainval["y"])
    base_metrics = S.evaluate(test["y"], score(base, test[S.FEATURES]), threshold)

    print(f"\nThreshold {threshold:.3f} = top {CAPACITY:.0%} expediting capacity")
    print("\nTEST (unseen):")
    for k in ("pr_auc", "roc_auc", "recall_at_20pct", "recall_at_10pct", "brier",
              "precision", "recall"):
        print(f"  {k:<18} {test_metrics[k]:.4f}   (type-rate baseline {base_metrics[k]:.4f})")

    print("\nSplit ablation:", end=" ")
    ablation = split_ablation(df, sla_def, json.loads(best["params"])
                              if best["model"] == "hgb" else {"max_iter": 200})
    print(f"{ablation}  |  chronological (reported) PR-AUC {test_metrics['pr_auc']:.4f}")

    report = {
        "final_model": best["model"],
        "final_params": json.loads(best["params"]),
        "threshold": threshold,
        "threshold_rule": f"top {CAPACITY:.0%} of arrivals by score (expediting capacity)",
        "test": test_metrics,
        "baseline_type_test": base_metrics,
        "validation_runs": runs.to_dict("records"),
        "split_ablation": ablation,
        "sla": {"quantile": sla_def["quantile"], "per_type_hours": sla_def["per_type"],
                "default_hours": sla_def["default"]},
        "split": {
            "train": len(train), "val": len(val), "test": len(test),
            "strategy": "chronological on created_date, whole days, ~70/15/15",
            "test_days": [str(test["created_date"].min().date()),
                          str(test["created_date"].max().date())],
        },
    }
    S.save_model(final, threshold, sla_def, report)
    (S.ARTIFACTS / "error_analysis.md").write_text(error_analysis(test, test_scores, threshold))
    print(f"\nSaved model.joblib, metrics.json, runs.csv, error_analysis.md -> {S.ARTIFACTS}")


if __name__ == "__main__":
    main()
