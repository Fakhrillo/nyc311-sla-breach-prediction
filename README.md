# GOV-02 — Public Service SLA Breach Prediction

**Student:** Fakhrillo
**Track:** Field-Based Scenario — GOV-02 (GovTech)
**Client:** Government service centre / municipal 311 operation

---

## Problem statement

A public service centre takes in citizen cases continuously — noise complaints, heating
failures, illegal parking, water leaks. Some are closed the same hour; some run for weeks.
Managers usually learn a case has stalled only once the deadline is near or already gone,
and a supervisor can personally expedite only a small share of daily arrivals.

**The operational question this project answers:** *of the cases arriving today, which ~20%
should a supervisor expedite?*

That framing — ranking under a fixed budget — drives the metric and the threshold below.

## ML task

| | |
| --- | --- |
| Task type | Binary classification, used as a **ranking** problem |
| One record | One service request, scored at the moment it is filed |
| Target | `y = 1` if the case took longer to close than its type's service window |
| Input at inference | Request type, descriptor, agency, borough, intake channel, location and address type, arrival timestamp, and the agency's queue state on arrival |
| Output | Breach probability, the service window it is judged against, and a flag for the expediting budget |
| Success criterion | Beat the request type's own historical breach rate — not just the prior |

## Dataset

**NYC 311 Service Requests** via the NYC Open Data (Socrata) API — no key, no login.

- Scope: **1 Jan – 30 Mar 2024**, the **12 request types** that make up ~60% of volume
- **481,212 cases** scanned from 786,857 in the window
- Agencies: NYPD (279,086), HPD (182,509), DOT (18,757), DOB (860)
- 0.14% still open at the snapshot; 0.81% have unusable durations (closure before intake, or
  auto-closed within seconds)
- Licence: NYC Open Data, public domain. No personal identifiers are pulled — addresses and
  coordinates are excluded from the download.

The CSV (~73 MB) is **not committed**. `src/sla.py` downloads it on first run, so a clean
Colab runtime needs no manual upload.

### Downloading it correctly is part of the work

Two API behaviours had to be handled, and both would silently corrupt the dataset:

- **Any `$order` times out** on a range this size (90 s+, no response). But paging by `$offset`
  *without* a sort is unsafe — the server may repeat or skip rows between pages. The fix is to
  page by **consecutive date windows**, which are indexed, fast, and provably cover the period
  exactly once. The fetcher errors out if any window hits the 50k row cap rather than losing rows.
- **An `IN(...)` clause over twelve request types** makes Socrata scan the table and time out,
  so type filtering happens client-side.

## Defining the target — the decision this project rests on

NYC 311 publishes a `due_date` column, which would be the ideal target. It is populated for
**0.53%** of Q1 2024 cases (4,168 of 786,857). The city's own SLA field is abandoned in
practice, so the service window has to be defined here.

**The definition used:** a case breaches if it took longer than the **75th percentile of
resolution time for its own request type**, measured on the training period only.

Two properties make it defensible:

- **Per type, not global.** Resolution time spans three orders of magnitude — a noise complaint's
  window is 0.9 hours, a water leak's is 1,032 hours (43 days). One global deadline would just
  relabel the request type.
- **Fitted on train only.** The windows are percentiles of resolution time, so fitting them on
  all the data would bake the test period's outcomes into the labels the model is scored against.

| Request type | Window | | Request type | Window |
| --- | ---: | --- | --- | ---: |
| WATER LEAK | 1032.1 h | | STREET CONDITION | 66.3 h |
| UNSANITARY CONDITION | 849.1 h | | HEAT/HOT WATER | 51.2 h |
| PLUMBING | 692.7 h | | ABANDONED VEHICLE | 4.0 h |
| PAINT/PLASTER | 574.0 h | | BLOCKED DRIVEWAY | 3.4 h |
| NOISE - RESIDENTIAL | 1.1 h | | ILLEGAL PARKING | 2.8 h |
| NOISE - COMMERCIAL | 0.9 h | | NOISE - STREET/SIDEWALK | 0.9 h |

Unseen types fall back to a global 41.7 h window; types with fewer than 200 training cases do
too, because a percentile fitted to a handful of cases is not a window.

### The target is deliberately hard

Because the window is each type's own 75th percentile, **every type breaches about 25% of the
time by construction**. Knowing the request type tells you almost nothing about the label — so
the model cannot score by guessing the category and must find signal in queue state, geography,
channel and timing instead. That is why the headline comparison is against a **type-rate
baseline**, not the prior.

### Censoring is handled, not dropped

A case still open at the snapshot is not automatically unlabelled. If it has already been open
longer than its window it **has** breached, whatever happens next. Only a case still open *and*
still inside its window is genuinely unknown. Dropping every open case instead would bias the
data against exactly the slow cases the client cares about. Dropped as unknown: 6 train,
18 validation, 148 test.

## Pipeline

```
NYC Open Data API
  └─ fetch()         date-windowed paging, retries, 50k-cap guard
  └─ clean()         parse timestamps, drop impossible durations, normalise categoricals
  └─ add_workload()  backlog + 7-day volumes as known on arrival  ← leakage-critical
  └─ add_calendar()  arrival hour, weekday, weekend
  └─ time_split()    chronological on created_date, whole days, ~70/15/15
  └─ fit_sla()       service windows from the TRAINING period only
  └─ apply_sla()     breach label, with censoring handled explicitly
  └─ make_model()    ColumnTransformer → estimator, one sklearn Pipeline
  └─ evaluate()      PR-AUC, ROC-AUC, recall@capacity, Brier
  └─ predict_one()   validate raw intake dict → features → risk band
```

Preprocessing lives **inside** the sklearn pipeline, so the transforms fitted on training data
are the ones applied at inference.

### Features, and what was left out

Intake-time only: `created_hour`, `created_dow`, `is_weekend`, `agency_backlog`,
`agency_7d_volume`, `type_7d_volume`, `sla_hours`, `complaint_type`, `descriptor`, `agency`,
`borough`, `open_data_channel_type`, `location_type`, `address_type`, `has_zip`.

Excluded as leakage: `closed_date` (the target is computed from it), `status` (says whether the
case already resolved), `resolution_description` and `resolution_action_updated_date` (written
when the case closes).

**The workload feature is the delicate one.** `agency_backlog` counts the agency's cases created
before this one that had not yet closed when it arrived. That uses other cases' closing times
only to ask *"were you still open at time T"* — which the queue genuinely knows at T. It never
asks whether a case closes after T. A version that simply counted prior cases, or that let later
closures retroactively empty the queue, would leak. `test_sla.py` pins this behaviour.

## Split

Chronological on arrival date, cutting on whole days. Cases arrive continuously, so the
deployment question is always "given the queue up to today, which of today's arrivals will run
long". A random split would train on late March to score early January, and would smear the
backlog features across the boundary.

| Split | Cases | Arrival dates | Breach rate |
| --- | ---: | --- | ---: |
| Train | 335,616 | 2024-01-01 → 2024-03-02 | 0.255 |
| Validation | 72,033 | 2024-03-03 → 2024-03-16 | 0.256 |
| Test | 73,391 | 2024-03-17 → 2024-03-30 | 0.245 |

### What a random split would have reported

Same model, same features, same target — only the split changed:

| Split | PR-AUC | ROC-AUC |
| --- | ---: | ---: |
| Random (wrong) | 0.4775 | 0.7272 |
| **Chronological (reported)** | **0.4016** | **0.6747** |

A **19% overstatement** of PR-AUC, from one line of code. This is the single easiest way to
inflate a project like this, and it is why the honest number is the smaller one.

## Models compared

All scored on validation. `baseline_type` predicts each request type's historical breach rate.

| Model | PR-AUC | ROC-AUC | Recall@20% | Brier | Fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline — prior | 0.2560 | 0.5000 | 0.210 | 0.1905 | 0.2 s |
| Baseline — type rate | 0.3032 | 0.5697 | 0.250 | 0.1894 | 0.1 s |
| Logistic regression | 0.3721 | 0.6361 | 0.302 | 0.2320 | 2.7 s |
| **HistGradientBoosting (lr 0.1, 200 it)** | **0.4346** | **0.6937** | **0.345** | **0.1729** | 4.7 s |
| HistGradientBoosting (lr 0.05, 400 it) | 0.4303 | 0.6927 | 0.344 | 0.1734 | 12.1 s |
| HistGradientBoosting (lr 0.05, 600 it) | 0.4258 | 0.6899 | 0.342 | 0.1740 | 17.9 s |
| Random forest | 0.4219 | 0.6860 | 0.344 | 0.1749 | 7.2 s |

### Final model and why

**HistGradientBoostingClassifier**, `learning_rate=0.1, max_iter=200, max_leaf_nodes=31`.

Selection is not plain argmax on PR-AUC. Supervisors are shown a probability, so `train.py`
first rejects any candidate whose Brier score is worse than the no-skill prior, then takes the
best PR-AUC among the survivors. Logistic regression is rejected on exactly that basis
(Brier 0.2320 against a no-skill 0.1905) despite beating both baselines on ranking. The smallest
gradient-boosting configuration also won outright — the larger ones cost 2–4× the fit time and
scored slightly worse.

## Results on unseen test data

The test set was scored **once**, after the model and threshold were fixed on validation.

| Metric | Model | Type-rate baseline | |
| --- | ---: | ---: | --- |
| PR-AUC | **0.4016** | 0.2794 | 1.44× |
| ROC-AUC | **0.6747** | 0.5516 | |
| Recall @ 20% capacity | **0.3357** | 0.2467 | |
| Recall @ 10% capacity | **0.1913** | 0.1333 | |
| Brier score | **0.1714** | 0.1840 | lower is better |
| Precision @ threshold | 0.4158 | — | vs 0.245 base rate |
| Recall @ threshold | 0.3226 | — | |

**What this means for the service centre.** Spending the expediting budget on the model's top
20% catches **33.6%** of all breaches — against 24.7% for ranking by request type alone, and 20%
for expediting at random. Of the cases it flags, 41.6% do breach against a 24.5% base rate: a
1.7× concentration of supervisor attention.

ROC-AUC of 0.67 is modest, and that is the honest ceiling once the request type is neutralised
by construction. A project reporting 0.85 here is almost certainly using a random split, a
global deadline, or an outcome column.

### Threshold

Shipped threshold **0.365** = the top 20% of validation scores, i.e. the expediting budget.
Capacity comes first because supervisor attention exists in fixed supply; the question is then
what that budget buys. Applied to test it flags 19.0% — close, but it drifts, so it needs
periodic recalibration against live score distributions.

## Error analysis

Full slice tables in [`artifacts/error_analysis.md`](artifacts/error_analysis.md), regenerated
by every training run. Headlines:

- **DOT is where the model works** — recall 0.666, precision 0.562 on a 0.364 breach rate.
  Street-condition work has genuine queue dynamics the features capture.
- **HPD is where it struggles** — recall 0.217. Housing cases (heat, plumbing, leaks) run on
  inspection and landlord-compliance timelines that nothing at intake predicts.
- **Mid-range backlog is most predictable** (recall 0.412) — both an empty and an overwhelmed
  queue are harder to call than one under normal load.
- **False alarms and missed breaches look alike** on queue state (median backlog 352 vs 340),
  which says the remaining errors are not a backlog-threshold problem.

## Responsible AI

**Recall varies sharply by borough** — from **0.138 (Brooklyn)** to **0.576 (Queens)**, a gap of
0.44. This is the most important number in the project. Expediting is rationed public attention,
so a model that surfaces one borough's stalled cases four times more often than another's
redistributes municipal service along geographic lines. Nothing in the data says Brooklyn's
cases are less urgent — only that the historical queue treated them differently, and the model
learned to repeat it.

**Intake channel carries the same risk in a different shape.** If phone reports are escalated
less than online ones, the system quietly penalises whoever is least likely to file online —
typically older and lower-income residents. Channel recall spans 0.289 (phone) to 0.565
(unknown).

If equal recall across boroughs is required, the budget should be applied **per borough** rather
than globally. That is a policy decision for the agency, not a hyperparameter, and it should be
made before deployment rather than discovered after.

**The target measures speed, not quality.** A case closed fast and a case resolved well are
different events; only the first is visible. A model optimised on this target rewards closing
tickets, and if it were ever fed back into staff incentives it would push toward premature
closure — the classic way an SLA metric corrupts the service it measures. Output must stay
advisory: it prioritises attention and must never auto-reject, auto-close, or deprioritise a case.

**Privacy.** The download excludes addresses, coordinates and any free text. Borough and ZIP
presence are the only geographic attributes, and borough is retained precisely so the fairness
gap above can be measured.

**No causal claim.** This ranks which cases will run long. It does not show that expediting them
helps — that requires a trial of the intervention itself.

## Limitations

- **The target is invented.** "Slower than 3 in 4 cases of its kind" is defensible but is not a
  published or legal deadline. Every number here is relative to that choice, and a different
  percentile would move them all.
- **One quarter, one city, twelve request types.** Q1 spans the heating season, which drives HPD
  volume and behaviour; a summer quarter would look different.
- **The model learns the queue that existed**, including any bias already in it.
- **Agency coverage is uneven** — DOB contributes 860 cases, too few to say anything about.
- **Process changes break it.** If an agency reorganises its intake or triage rules, the
  historical relationship between queue state and delay no longer holds.

## Repository

```
nyc311-sla-breach-prediction/
├── README.md
├── requirements.txt
├── demo.ipynb                  ← reproducible Colab demo, run this first
├── train.py                    ← experiments → final model → test report
├── test_sla.py                 ← 11 assertions guarding the leakage-critical logic
├── src/sla.py                  ← API download, features, target, models, inference
├── data/                       ← CSV downloaded on first run (gitignored, ~73 MB)
└── artifacts/
    ├── model.joblib            ← pipeline + threshold + fitted SLA windows + metrics
    ├── metrics.json            ← full run report
    ├── runs.csv                ← every experiment (MLflow-equivalent record)
    └── error_analysis.md       ← slice tables + fairness check
```

## Setup and run

### Colab (recommended — this is the demo)

Open `demo.ipynb` in Colab (File → Open notebook → GitHub → this repo) and run all. It queries the API,
downloads the slice, trains, evaluates on unseen data, and scores example cases. No local files.

### Local

```bash
pip install -r requirements.txt

python test_sla.py      # 11 assertions, ~3 s
python train.py         # ~1 min after the data is cached; --quick for a smoke test
```

First run downloads ~73 MB from the NYC Open Data API (2–3 minutes). `train.py` writes
`artifacts/model.joblib`, `metrics.json`, `runs.csv` and `error_analysis.md`. Experiments log to
MLflow when installed (`mlflow ui --backend-store-uri file:./mlruns`); without it the same
records go to `runs.csv`.

### Example input and output

```python
import sys; sys.path.insert(0, "src")
import sla as S

S.predict_one({
    "created_date": "2024-03-20T08:00:00",
    "complaint_type": "HEAT/HOT WATER",
    "agency": "HPD",
    "borough": "BRONX",
    "open_data_channel_type": "ONLINE",
    "agency_backlog": 9000,
    "agency_7d_volume": 15000,
})
# {'breach_probability': 0.273,
#  'service_window_hours': 51.2,
#  'risk_band': 'medium',
#  'flagged_for_expediting': False,
#  'threshold': 0.3651}
```

Invalid input raises `ValueError` with a message safe to show a user — missing fields, an
unparseable timestamp, a negative backlog, or a blank request type. Request types never seen in
training fall back to the global service window rather than failing.

## Next steps

1. Evaluate a per-borough capacity rule against the global one and measure the fairness gap under both.
2. Add a second target — resolution time as regression — so the window percentile can be varied
   without retraining.
3. Extend to a full year to capture seasonality, then re-check whether Q1-trained windows hold.
4. Run a trial of expediting itself to establish whether the intervention works.
