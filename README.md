# Public Service SLA Breach Prediction (NYC 311)

**Student:** Fakhrillokhon Lutfillokhonov
**Track:** Field-Based Scenario, GOV-02 (GovTech)
**Client:** Government service centre / municipal 311 operation

---

## Problem statement

A public service centre takes in citizen cases continuously: noise complaints, heating
failures, illegal parking, water leaks. Some close within the hour, some run for weeks.
Managers normally find out a case has stalled once the deadline is near or already gone, and
a supervisor can personally chase only a small share of what arrives each day.

So the operational question is narrow and worth stating exactly:

> Of the cases arriving today, which ~20% should a supervisor expedite?

That framing is a ranking problem under a fixed budget, and it drives both the metric and the
threshold further down.

## ML task

| | |
| --- | --- |
| Task type | Binary classification, used as a ranking problem |
| One record | One service request, scored at the moment it is filed |
| Target | `y = 1` if the case took longer to close than its type's service window |
| Input at inference | Request type, descriptor, agency, borough, intake channel, location and address type, arrival timestamp, and the agency's queue state on arrival |
| Output | Breach probability, the service window it is judged against, and a flag for the expediting budget |
| Success criterion | Beat the request type's own historical breach rate, not just the prior |

## Dataset

NYC 311 Service Requests, pulled from the NYC Open Data (Socrata) API. No key, no login.

- Scope: 1 Jan to 30 Mar 2024, and the 12 request types making up roughly 60% of volume
- 481,212 cases kept, scanned out of 786,857 in the window
- Agencies: NYPD (279,086), HPD (182,509), DOT (18,757), DOB (860)
- 0.14% still open at the snapshot; 0.81% have unusable durations, meaning closure logged
  before intake or an auto-close within seconds
- Licence: NYC Open Data, public domain. The download excludes addresses and coordinates,
  so no personal identifiers enter the repository.

The CSV is about 73 MB and is not committed. `src/sla.py` downloads it on first run, so a
clean Colab runtime needs no manual upload.

### Downloading it correctly turned out to be part of the work

Two API behaviours had to be worked around, and both would have corrupted the dataset
quietly rather than loudly:

Any `$order` times out on a range this size. I measured it: unordered returns in 39 seconds,
sorting by `unique_key` or by `:id` hangs for the full 90 seconds and returns nothing. That
rules out offset paging, because `$offset` without a sort is unsafe. The server is free to
repeat or skip rows between pages and nothing in the response would tell you. Paging by
consecutive date windows avoids the problem entirely: the date column is indexed, it is fast,
and the windows provably cover the period exactly once. The fetcher also raises if any window
comes back at the 50k row cap, since that would mean cases were silently dropped.

An `IN(...)` clause over twelve request types makes Socrata scan the table and time out too,
so the type filter runs client-side on each chunk instead.

## Defining the target, which is the decision everything else rests on

NYC 311 publishes a `due_date` column. It would be the ideal target, and it is populated for
0.53% of Q1 2024 cases (4,168 out of 786,857). The city's own SLA field is abandoned in
practice, so the service window has to be defined here instead.

The definition I settled on: a case breaches if it took longer than the 75th percentile of
resolution time for its own request type, measured on the training period only.

Two properties make that defensible. First, it is per type rather than global. Resolution
time spans three orders of magnitude here, from a 0.9 hour window for a noise complaint to
1,032 hours (43 days) for a water leak, and a single global deadline would do nothing but
relabel the request type. Second, it is fitted on train only. The windows are percentiles of
resolution time, so fitting them across all the data would bake the test period's outcomes
into the labels the model is later scored against.

| Request type | Window | | Request type | Window |
| --- | ---: | --- | --- | ---: |
| WATER LEAK | 1032.1 h | | STREET CONDITION | 66.3 h |
| UNSANITARY CONDITION | 849.1 h | | HEAT/HOT WATER | 51.2 h |
| PLUMBING | 692.7 h | | ABANDONED VEHICLE | 4.0 h |
| PAINT/PLASTER | 574.0 h | | BLOCKED DRIVEWAY | 3.4 h |
| NOISE - RESIDENTIAL | 1.1 h | | ILLEGAL PARKING | 2.8 h |
| NOISE - COMMERCIAL | 0.9 h | | NOISE - STREET/SIDEWALK | 0.9 h |

Unseen types fall back to a global 41.7 h window, as do types with fewer than 200 training
cases. A percentile fitted to a handful of cases is not a window.

### The target is deliberately hard

Because each window is that type's own 75th percentile, every type breaches about 25% of the
time by construction. Knowing the request type therefore tells you almost nothing about the
label, and the model cannot score well by guessing the category. It has to find signal in
queue state, geography, channel and timing instead.

This is also why the headline comparison is against a type-rate baseline rather than the
prior. Beating the prior here would prove nothing.

### Censoring is handled rather than dropped

A case still open at the snapshot is not automatically unlabelled. If it has already been
open longer than its window then it has breached, whatever happens next. Only a case that is
still open *and* still inside its window is genuinely unknown, and only those get dropped.

The tempting shortcut is to drop every open case, and it would have been a self-serving one:
cases still open after weeks are precisely the slow cases the client cares about. Dropped as
genuinely unknown: 6 train, 18 validation, 148 test.

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

Preprocessing lives inside the sklearn pipeline rather than being applied to the dataframe
beforehand, so the transforms fitted on training data are exactly the ones applied at
inference. That is the usual source of train/serve skew and it costs nothing to avoid.

### Features, and what was deliberately left out

Intake-time only: `created_hour`, `created_dow`, `is_weekend`, `agency_backlog`,
`agency_7d_volume`, `type_7d_volume`, `sla_hours`, `complaint_type`, `descriptor`, `agency`,
`borough`, `open_data_channel_type`, `location_type`, `address_type`, `has_zip`.

Excluded as leakage: `closed_date` (the target is computed from it), `status` (says whether
the case already resolved), `resolution_description` and `resolution_action_updated_date`
(both written when the case closes).

The workload feature is the delicate one. `agency_backlog` counts the agency's cases created
before this one that had not yet closed when it arrived. It does read other cases' closing
times, which looks like cheating until you look at the question it asks them, which is only
*"were you still open at time T"*. A dispatcher looking at their queue at time T knows
exactly that. It never asks whether a case closes after T.

A version that simply counted prior cases, or one that let later closures retroactively empty
the queue, would leak. `test_sla.py` pins the behaviour down so it cannot drift.

## Split

Chronological on arrival date, cutting on whole days. Cases arrive continuously, so the
deployment question is always "given the queue up to today, which of today's arrivals will
run long". A random split would train on late March to score early January, and would smear
the backlog features straight across the boundary.

| Split | Cases | Arrival dates | Breach rate |
| --- | ---: | --- | ---: |
| Train | 335,616 | 2024-01-01 → 2024-03-02 | 0.255 |
| Validation | 72,033 | 2024-03-03 → 2024-03-16 | 0.256 |
| Test | 73,391 | 2024-03-17 → 2024-03-30 | 0.245 |

### What a random split would have reported

Same model, same features, same target. Only the split changed:

| Split | PR-AUC | ROC-AUC |
| --- | ---: | ---: |
| Random (wrong) | 0.4775 | 0.7272 |
| Chronological (reported) | **0.4016** | **0.6747** |

A 19% overstatement of PR-AUC, out of one line of code. This is the easiest way to inflate a
project like this one, which is why the number is in the README rather than left out of it,
and why the honest figure is the smaller one.

## Models compared

All scored on validation. `baseline_type` predicts each request type's historical breach rate.

| Model | PR-AUC | ROC-AUC | Recall@20% | Brier | Fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, prior | 0.2560 | 0.5000 | 0.210 | 0.1905 | 0.3 s |
| Baseline, type rate | 0.3032 | 0.5697 | 0.250 | 0.1894 | 0.1 s |
| Logistic regression | 0.3721 | 0.6361 | 0.302 | 0.2320 | 3.8 s |
| **HistGradientBoosting (lr 0.1, 200 it)** | **0.4346** | **0.6937** | **0.345** | **0.1729** | 3.2 s |
| HistGradientBoosting (lr 0.05, 400 it) | 0.4303 | 0.6927 | 0.344 | 0.1734 | 9.3 s |
| HistGradientBoosting (lr 0.05, 600 it) | 0.4258 | 0.6899 | 0.342 | 0.1740 | 13.3 s |
| Random forest | 0.4219 | 0.6860 | 0.344 | 0.1749 | 7.6 s |

### Final model, and why that one

HistGradientBoostingClassifier with `learning_rate=0.1, max_iter=200, max_leaf_nodes=31`.

Selection is not plain argmax on PR-AUC. A supervisor is shown a probability, so `train.py`
first rejects any candidate whose Brier score is worse than the no-skill prior, then takes
the best PR-AUC among whatever survives. Logistic regression gets rejected on exactly that
basis, at Brier 0.2320 against a no-skill 0.1905, even though it beats both baselines on
ranking. Its ordering is useful; its numbers are not, and 0.8 needs to mean something close
to 0.8 if anyone is going to act on it.

The smallest gradient-boosting configuration also won outright. The larger ones cost two to
four times the fit time and scored slightly worse, which says the problem is data-limited
rather than capacity-limited.

## Results on unseen test data

The test set was scored once, after the model and threshold were both fixed on validation.

| Metric | Model | Type-rate baseline | |
| --- | ---: | ---: | --- |
| PR-AUC | **0.4016** | 0.2794 | 1.44× |
| ROC-AUC | **0.6747** | 0.5516 | |
| Recall @ 20% capacity | **0.3357** | 0.2467 | |
| Recall @ 10% capacity | **0.1913** | 0.1333 | |
| Brier score | **0.1714** | 0.1840 | lower is better |
| Precision @ threshold | 0.4158 | — | vs 0.245 base rate |
| Recall @ threshold | 0.3226 | — | |

In terms the service centre would care about: spending the expediting budget on the model's
top 20% catches 33.6% of all breaches, against 24.7% for ranking by request type alone and
20% for expediting at random. Of the cases it flags, 41.6% do breach, against a 24.5% base
rate. That is a 1.7× concentration of supervisor attention.

A ROC-AUC of 0.67 is modest and I want to be direct about that rather than dress it up. It is
close to the honest ceiling once the request type has been neutralised by construction. A
project reporting 0.85 on this task is almost certainly using a random split, a global
deadline, or an outcome column.

### Threshold

The shipped threshold is 0.365, which is the top 20% of validation scores, i.e. the expediting
budget. Capacity comes first here because supervisor attention exists in fixed supply and does
not care what a cost ratio says; the question is what that budget buys. Applied to the test
period it flags 19.0%, which is close but does drift, so it needs periodic recalibration
against live score distributions.

## Error analysis

Full slice tables are in [`artifacts/error_analysis.md`](artifacts/error_analysis.md),
regenerated by every training run. The headlines:

DOT is where the model works, at recall 0.666 and precision 0.562 on a 0.364 breach rate.
Street-condition work has genuine queue dynamics and the features capture them.

HPD is where it struggles, at recall 0.217. Housing cases (heat, plumbing, leaks) run on
inspection schedules and landlord-compliance timelines, and nothing visible at intake
predicts those.

Mid-range backlog is the most predictable band at recall 0.412. Both an empty queue and an
overwhelmed one are harder to call than a queue under normal load, which makes intuitive
sense: in the first case nothing is contended, in the second everything is.

False alarms and missed breaches look almost identical on queue state, at median backlog 352
against 340. Whatever is left in the errors, it is not a backlog-threshold problem.

## Responsible AI

Recall varies sharply by borough, from 0.138 in Brooklyn to 0.576 in Queens, a gap of 0.44.
This is the most important number in the project and it is not a good one. Expediting is
rationed public attention, so a model that surfaces one borough's stalled cases four times
more often than another's is redistributing municipal service along geographic lines. Nothing
in the data says Brooklyn's cases are less urgent. It says only that the historical queue
treated them differently, and the model learned to repeat it.

Intake channel carries the same risk in a different shape. Channel recall spans 0.289 for
phone to 0.565 for unknown, and if phone reports are escalated less than online ones the
system quietly penalises whoever is least likely to file online, which typically means older
and lower-income residents.

If equal recall across boroughs is required, the budget should be applied per borough rather
than globally. That is a policy decision for the agency rather than a hyperparameter, and it
should be made before deployment rather than discovered afterwards.

The target also measures speed, not quality. A case closed fast and a case resolved well are
different events and only the first is visible here. A model optimised on this target rewards
closing tickets, and if it were ever fed back into staff incentives it would push toward
premature closure, which is the classic way an SLA metric corrupts the service it was meant
to measure. The output has to stay advisory: it prioritises attention, and must never
auto-reject, auto-close, or deprioritise a case.

On privacy, the download excludes addresses, coordinates and free text. Borough and ZIP
presence are the only geographic attributes kept, and borough is kept specifically so the
fairness gap above can be measured at all.

One thing this project does not claim: causality. It ranks which cases will run long. It does
not show that expediting them helps, and establishing that would need a trial of the
intervention itself.

## Limitations

The target is invented. "Slower than 3 in 4 cases of its kind" is defensible, but it is not a
published or legal deadline, and every number here is relative to that choice. A different
percentile would move all of them.

One quarter, one city, twelve request types. Q1 covers the heating season, which drives HPD
volume and behaviour, and a summer quarter would look different.

The model learns the queue that existed, including whatever bias was already in it.

Agency coverage is uneven. DOB contributes 860 cases, far too few to say anything about.

Process changes break it. If an agency reorganises its intake or triage rules, the historical
relationship between queue state and delay stops holding.

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

### Colab, which is the demo

Open `demo.ipynb` in Colab (File → Open notebook → GitHub → this repo) and run all. It
queries the API, downloads the slice, trains, evaluates on unseen data and scores example
cases. Nothing needs to be uploaded.

### Local

```bash
pip install -r requirements.txt

python test_sla.py      # 11 assertions, ~3 s
python train.py         # ~1 min once the data is cached; --quick for a smoke test
```

The first run downloads about 73 MB from the NYC Open Data API, which takes two to three
minutes. `train.py` writes `artifacts/model.joblib`, `metrics.json`, `runs.csv` and
`error_analysis.md`. Experiments log to MLflow when it is installed, and to `runs.csv` when it is not, so the
pipeline never depends on MLflow being present. To browse the tracked runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

SQLite rather than the usual `./mlruns` directory, because MLflow 3.x refuses the
filesystem backend and raises instead of tracking.

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

Invalid input raises `ValueError` with a message that is safe to show a user: missing fields,
an unparseable timestamp, a negative backlog, or a blank request type. Request types never
seen in training fall back to the global service window rather than failing outright.

## Sources and acknowledgements

Data comes from NYC Open Data, 311 Service Requests (dataset `erm2-nwe9`), which is public
domain. Built with scikit-learn, pandas, numpy, matplotlib and joblib, with MLflow as an
optional extra.

AI assistance: I used Claude (Anthropic) as a coding assistant while building this, mainly
for diagnosing the Socrata paging constraints, drafting the module layout, and pressure-testing
the leakage argument behind `add_workload()`. The problem framing, the target definition and
the decision to report the chronological rather than the random-split numbers are mine, and
every result in this README is reproducible by running `train.py` on a clean checkout.

## Next steps

1. Evaluate a per-borough capacity rule against the global one and measure the fairness gap
   under both.
2. Add a second target, resolution time as a regression, so the window percentile can be
   varied without retraining.
3. Extend to a full year to capture seasonality, then re-check whether Q1-trained windows
   still hold.
4. Run a trial of expediting itself, to establish whether the intervention actually works.
