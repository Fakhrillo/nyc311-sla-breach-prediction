# Public Service SLA Breach Prediction (NYC 311)

**Student:** Fakhrillokhon Lutfillokhonov
**Track:** Field-Based Scenario, GOV-02 (GovTech)
**Client:** Government service centre / municipal 311 operation

**Quick demo**, loads the trained model and scores cases in under a minute:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fakhrillo/nyc311-sla-breach-prediction/blob/main/quick_demo.ipynb)

**Full reproducible pipeline**, download to evaluation, 10 to 15 minutes mostly downloading:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fakhrillo/nyc311-sla-breach-prediction/blob/main/demo.ipynb)

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
| Target | `y = 1` if the case took longer to close than its type's service window, a proxy derived in this project rather than NYC's official SLA |
| Input at inference | Request type, descriptor, agency, borough, intake channel, location and address type, arrival timestamp, and the agency's queue state on arrival |
| Output | Breach probability, the service window it is judged against, and a flag for the expediting budget |
| Success criterion | On the chronological test set: beat the type-rate baseline on recall at 20% capacity (primary) and on PR-AUC, with a Brier score no worse than the no-skill prior |

## Dataset

NYC 311 Service Requests, pulled from the NYC Open Data (Socrata) API. No key, no login.

- Scope: 1 Jan to 30 Mar 2024, and the 12 request types making up roughly 60% of volume
- 481,212 cases kept out of 786,853 scanned, of which 477,987 are labelled
- Agencies: NYPD (279,086), HPD (182,509), DOT (18,757), DOB (860)
- 0.14% still open when the data was extracted (658 cases), all of them labelled, and a
  further 0.67% with impossible durations (3,225): closure logged before intake, or an
  auto-close within seconds. Those are dropped before labelling.
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
and the windows cover the period exactly once. The fetcher also raises if any window comes
back at the 50k row cap, since that would mean cases were silently dropped.

Getting "exactly once" right took a second attempt. The first version built its 3-day windows
with a grid that stopped short of 1 April, so 31 March was never downloaded and nothing
complained. That is why the scope above ends on 30 March. The window builder now always
reaches the end date, and `test_sla.py` pins it.

An `IN(...)` clause over twelve request types makes Socrata scan the table and time out too,
so the type filter runs client-side on each chunk instead.

## Defining the target, which is the decision everything else rests on

NYC 311 publishes a `due_date` column. It would be the ideal target, and it is populated for
0.53% of Q1 2024 cases (4,168 out of 786,857). The city's own SLA field is abandoned in
practice, so the service window has to be defined here instead.

So a word on the name. "SLA" in this project's title means that derived window: a proxy for a
service commitment, built from how long cases of each type historically took. It is not
NYC's official SLA, and nothing here should be read as the city's own deadline.

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

### Open cases, impossible durations, and the snapshot

Three kinds of record have to be told apart, and `apply_sla()` does it by `closed_date`, never
by a missing duration:

- **Closed, with a usable duration.** A breach if it took longer than its window.
- **Closed, with an impossible duration**: closed before it opened, or auto-closed within
  seconds. There is no real outcome to learn from, so these are dropped before labelling:
  1,765 train, 839 validation, 621 test. An earlier version blanked the duration and let
  these fall through as "still open", which labelled every one of them a breach. It was
  caught in review, and every figure in this README comes from the corrected run. The damage
  was not spread evenly: 8% of street-condition records are auto-closes, which is most of why
  that type used to breach 31% of the time instead of 25%.
- **Still open when the data was pulled.** Dropping these is the tempting shortcut, and a
  self-serving one: cases still open after weeks are precisely the slow cases the client cares
  about. A case open longer than its window has breached, whatever happens next.

"Open longer than its window" needs a clock, and the right one is when the extract was taken.
The CSV carries no pull date, but it cannot contain a closure that had not happened yet, so
its latest `closed_date`, 8 July 2026, is a lower bound (`extract_time()`). By then all 658
open cases had been open for more than two years, far past the longest window of 43 days, so
every one is a known breach and none is left unknown. The earlier version took the last
*intake* time, 30 March 2024, as if the data had been pulled the moment intake stopped, and
dropped 78 open cases as "unknown" that were in fact two years overdue.

## Pipeline

```
NYC Open Data API
  └─ fetch()         date-windowed paging, retries, 50k-cap guard
  └─ clean()         parse ISO timestamps, blank impossible durations, normalise categoricals
  └─ add_workload()  backlog + 7-day volumes as known on arrival  ← leakage-critical
  └─ add_calendar()  arrival hour, weekday, weekend
  └─ time_split()    chronological on created_date, whole days, ~70/15/15
  └─ fit_sla()       service windows from the TRAINING period only
  └─ apply_sla()     breach label: impossible durations dropped, open cases labelled at the snapshot
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
| Train | 333,857 | 2024-01-01 → 2024-03-02 | 0.251 |
| Validation | 71,212 | 2024-03-03 → 2024-03-16 | 0.247 |
| Test | 72,918 | 2024-03-17 → 2024-03-30 | 0.240 |

### What a random split would have reported

Same model, same features, same target. Only the split changed:

| Split | PR-AUC | ROC-AUC |
| --- | ---: | ---: |
| Random (wrong) | 0.4739 | 0.7298 |
| Chronological (reported) | **0.3990** | **0.6802** |

A 19% overstatement of PR-AUC, out of one line of code. This is the easiest way to inflate a
project like this one, which is why the number is in the README rather than left out of it,
and why the honest figure is the smaller one.

## Models compared

All scored on validation. `baseline_type` predicts each request type's historical breach rate.

| Model | PR-AUC | ROC-AUC | Recall@20% | Brier | Fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, prior | 0.2474 | 0.5000 | 0.200 | 0.1862 | 0.3 s |
| Baseline, type rate | 0.2474 | 0.4831 | 0.179 | 0.1861 | 0.1 s |
| Logistic regression | 0.3639 | 0.6347 | 0.304 | 0.2308 | 2.8 s |
| **HistGradientBoosting (lr 0.1, 200 it)** | **0.4198** | **0.6921** | **0.349** | **0.1700** | 3.3 s |
| HistGradientBoosting (lr 0.05, 400 it) | 0.4197 | 0.6924 | 0.350 | 0.1700 | 7.9 s |
| HistGradientBoosting (lr 0.05, 600 it) | 0.4141 | 0.6880 | 0.344 | 0.1709 | 11.4 s |
| Random forest | 0.4143 | 0.6874 | 0.352 | 0.1710 | 6.9 s |

Fit times are from the committed `runs.csv` and vary by machine. The random forest's scores
move slightly too: the same code and seed give PR-AUC 0.4143 in this table, run on a Mac,
and 0.4135 on Colab. The selected gradient-boosting model and every test figure come out
identical on both.

On validation the type-rate baseline is no better than the prior, with a ROC-AUC of 0.48.
That is what the target's design predicts, since every type breaches about 25% of the time in
training. Before the impossible-duration fix it looked stronger (PR-AUC 0.303), because the
mislabelled auto-closes are concentrated in a few request types and handed it a signal that
was not real. On the test period it keeps a little genuine signal, as the breach rate by type
drifts from the training period.

### Final model, and why that one

HistGradientBoostingClassifier with `learning_rate=0.1, max_iter=200, max_leaf_nodes=31`.

Selection is not plain argmax on PR-AUC. A supervisor is shown a probability, so `train.py`
first rejects any candidate whose Brier score is worse than the no-skill prior, then takes
the best PR-AUC among whatever survives. Logistic regression gets rejected on exactly that
basis, at Brier 0.2308 against a no-skill 0.1862, even though it beats both baselines on
ranking. Its ordering is useful; its numbers are not, and 0.8 needs to mean something close
to 0.8 if anyone is going to act on it.

The smallest gradient-boosting configuration won. The 400-iteration one tied it (PR-AUC
0.4197 against 0.4198) for more than twice the fit time, and the 600-iteration one scored
lower, which says the problem is data-limited rather than capacity-limited.

## Results on unseen test data

The test set was scored once, after the model and threshold were both fixed on validation.

| Metric | Model | Type-rate baseline | |
| --- | ---: | ---: | --- |
| PR-AUC | **0.3990** | 0.2658 | 1.50× |
| ROC-AUC | **0.6802** | 0.5429 | |
| Recall @ 20% capacity | **0.3401** | 0.2365 | |
| Recall @ 10% capacity | **0.1912** | 0.1254 | |
| Brier score | **0.1684** | 0.1820 | lower is better |
| Precision @ threshold | 0.4062 | — | vs 0.240 base rate |
| Recall @ threshold | 0.3441 | — | |

In terms the service centre would care about: spending the expediting budget on the model's
top 20% catches 34.0% of all breaches, against 23.6% for ranking by request type alone and
20% for expediting at random. Of the cases it flags, 40.6% do breach, against a 24.0% base
rate. That is a 1.7× concentration of supervisor attention.

Recall at capacity shares out cases tied at the cut-off in proportion. That matters for the
baseline: it gives only 12 distinct scores, so about 10,800 cases tie at the 20% line, and
before ties were handled its figure depended on how the sort happened to order them. On the
earlier labels it came out 0.2467 on a Mac and 0.2520 on Colab for the same data. The model's scores are
effectively unique, so its figures never moved.

A ROC-AUC of 0.68 is modest and I want to be direct about that rather than dress it up. It is
close to the honest ceiling once the request type has been neutralised by construction. A
project reporting 0.85 on this task is almost certainly using a random split, a global
deadline, or an outcome column.

### Threshold

The shipped threshold is 0.354, which is the top 20% of validation scores, i.e. the expediting
budget. Capacity comes first here because supervisor attention exists in fixed supply and does
not care what a cost ratio says; the question is what that budget buys. Applied to the test
period it flags 20.3%, which is close but can drift, so it needs periodic recalibration
against live score distributions.

## Error analysis

Full slice tables are in [`artifacts/error_analysis.md`](artifacts/error_analysis.md),
regenerated by every training run. The headlines:

DOT is where the model works, at recall 0.655 and precision 0.593 on a 0.325 breach rate.
Street-condition work has genuine queue dynamics and the features capture them.

HPD is where it struggles, at recall 0.227. Housing cases (heat, plumbing, leaks) run on
inspection schedules and landlord-compliance timelines, and nothing visible at intake
predicts those.

Mid-range backlog is the most predictable band at recall 0.435. Both an empty queue and an
overwhelmed one are harder to call than a queue under normal load, which makes intuitive
sense: in the first case nothing is contended, in the second everything is.

False alarms and missed breaches look almost identical on queue state, at median backlog 349
against 343. Whatever is left in the errors, it is not a backlog-threshold problem.

## Responsible AI

Recall varies sharply by borough, from 0.151 in Brooklyn to 0.620 in Queens, a gap of 0.47.
This is the most important number in the project and it is not a good one. Expediting is
rationed public attention, so a model that surfaces one borough's stalled cases four times
more often than another's is redistributing municipal service along geographic lines. Nothing
in the data says Brooklyn's cases are less urgent. It says only that the historical queue
treated them differently, and the model learned to repeat it.

Intake channel carries the same risk in a different shape. Channel recall spans 0.321 for
online to 0.511 for unknown. Phone and online are close, at 0.326 against 0.321, so the
failure I was watching for, phone reports escalated less than online ones, does not show in
this quarter. It still has to be monitored: if it appeared, it would quietly penalise whoever
is least likely to file online, which typically means older and lower-income residents.

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

Labels use every closure known when the data was extracted, two years later. A model retrained
live would see the most recent weeks' long-running cases still open, so its newest training
labels would be incomplete. That is one more reason for the monitoring lag under question 4.

Process changes break it. If an agency reorganises its intake or triage rules, the historical
relationship between queue state and delay stops holding. What to do about that is answered
under question 4 below.

## Responses to the GOV-02 brief

The brief asks for a Data & Problem Discovery table (§5), a Technical Proposal (§6), and
answers to six questions it says must be resolved (§11). They are collected here in the
brief's own structure; the sections above carry the detail.

### §5 Data & Problem Discovery

| Decision / question | Response |
| --- | --- |
| Selected dataset and source | NYC 311 Service Requests (`erm2-nwe9`) from the NYC Open Data API, public domain. 1 Jan to 30 Mar 2024, the 12 request types making up ~60% of volume: 481,212 cases, 477,987 of them labelled. |
| What one record represents | One citizen service request, from intake to closure. |
| Proposed target | `y = 1` if the case took longer to close than the 75th percentile of resolution time for its own request type, fitted on the training period only. The city's own `due_date` is populated for 0.53% of cases, so it cannot be the target. |
| Information available at prediction time | Request type and descriptor, agency, borough, intake channel, location and address type, arrival time, and the agency's queue at that moment: open backlog and 7-day intake volume. |
| Main data quality issues | No usable deadlines (0.53%). 0.14% of cases still open when extracted; 0.67% with impossible durations, dropped before labelling. Location type missing for 2.69%, intake channel recorded as UNKNOWN for 2.77%, ZIP missing for 0.12%. No duplicate case IDs and no unparseable timestamps. |
| How missing data is handled | Categorical gaps become an explicit UNKNOWN category, so missingness is itself something the model can use. Missing ZIP becomes the `has_zip` flag. Impossible durations are dropped before labelling, rather than being read as zero or as still open, and open cases are labelled by the censoring rule in `apply_sla()`. Numeric features are derived from timestamps and the queue, so they are never missing. |
| Potential leakage risks | Outcome columns (`closed_date`, `status`, `resolution_description`, `resolution_action_updated_date`); service windows fitted on the period being scored; a backlog count that lets future closures empty the queue; a random split. All four are designed out, and the outcome columns, the backlog and the split are each pinned by a test. |
| Privacy, fairness and licensing | Public-domain data. Addresses, coordinates and free text are excluded at download. Borough is kept deliberately so the fairness gap can be measured. |

### §6 Technical Proposal

| Decision / question | Response |
| --- | --- |
| ML problem formulation | Binary classification, used as a ranking problem under a fixed expediting capacity. |
| Proposed baseline | Two: the class prior, and each request type's historical breach rate. The second is the one that matters, because the target makes request type nearly uninformative by design. |
| Modelling approaches investigated | Logistic regression, histogram gradient boosting in three configurations, and a random forest. |
| Splitting and validation | Chronological on arrival date, cut on whole days, ~70/15/15. The test set is scored once. A random split is run only as an ablation, and overstates PR-AUC by 19%. |
| Primary metric and why | Recall at 20% capacity: a supervisor can expedite a fixed share of arrivals, and the question is what that share catches. PR-AUC summarises the ranking overall; Brier checks the probabilities, because a person is shown them. Accuracy is not used: at a 25% base rate it rewards calling everything on time. |
| Expected inference input | One intake record. `created_date`, `complaint_type`, `agency` and `borough` are required; channel, descriptor, location and address type, ZIP and queue counts are optional. |
| Expected inference output | Breach probability, the service window the case is judged against, a low / medium / high risk band, and whether it clears the expediting threshold. |
| Main technical risks and assumptions | The target is defined rather than given. One quarter of data. Queue behaviour may not survive process changes. Recall is unequal across boroughs. |

### §11 Questions the brief says must be resolved

**1. Is classification or processing-time prediction more useful here?** Classification. The
decision a supervisor makes is binary, expedite or not, and a per-type window turns "how long"
into "longer than usual for its kind", which is the question they act on. Processing time is
also a poor regression target in this data: the fitted windows run from 0.9 hours to 1,032,
so a single error scale across a noise complaint and a water leak is hard to read, and the
long housing cases would dominate it. Regression stays in Next steps as a complement, not a
replacement.

**2. When exactly should the system make a prediction?** At intake, the moment a case is
filed, using only what the queue knows at that instant. That is the earliest point a
supervisor could act, and it is what keeps every feature causal.

**3. Which variables may leak the final delay outcome?** The four outcome columns listed
above, all written as or after a case resolves, all excluded. Two subtler routes matter just
as much: service windows fitted on the period being scored, and a backlog count that lets
future closures empty the queue. Windows are fitted on training data only, and the backlog is
computed as known at arrival.

**4. What should happen if historical process rules change?** The model assumes queue state
relates to delay the way it did in Q1 2024, and a change to intake or triage rules breaks
that. The response has three parts. Watch for it: the fixed threshold should keep flagging
about 20% of arrivals (it flags 20.3% on the test period), so a sustained drift in that share
is the first signal, and recall at 20% on newly closed cases is the second. That second
signal lags, because a case cannot be labelled until its window has passed, which is up to 43
days for a water leak. Once a change is confirmed, refit the service windows and retrain on
post-change data only. Until there is enough of it, fall back to the type-rate baseline, which
needs far less history.

Changing *workload* is a different matter from changing *rules*. Backlog and recent intake
volume are inputs, so a surge moves the scores directly without retraining, as long as it
stays within the range the model saw in training.

**5. Which errors create the greatest operational cost?** A missed breach: a case that runs
late and nobody chased, which is the failure the system exists to prevent, and a resident
bears it. A false alarm costs supervisor attention instead, but with fixed capacity every false
alarm also takes a slot from a case that would have breached. So under a fixed budget both
errors come down to the same currency, breaches not caught, which is why the threshold is set
by capacity rather than a cost ratio and why recall at 20% is the headline metric. The cost is
not spread evenly either: the model misses the largest share of breaches in HPD housing cases
(recall 0.227) and in Brooklyn (0.151).

**6. How is fairness evaluated across regions and departments?** Recall of the expediting
flag, computed on the test set per borough, per agency and per intake channel, and regenerated
into `artifacts/error_analysis.md` on every training run. Recall is the right measure because
the harm is a breach nobody chased. Results: borough 0.151 to 0.620, agency 0.227 (HPD) to
0.655 (DOT), channel 0.321 (online) to 0.511 (unknown). What to do about the borough gap is
set out under Responsible AI.

## Repository

```
nyc311-sla-breach-prediction/
├── README.md
├── requirements.txt
├── quick_demo.ipynb            ← loads the trained model, scores cases, under a minute
├── demo.ipynb                  ← full reproducible pipeline on Colab, download to evaluation
├── train.py                    ← experiments → final model → test report
├── test_sla.py                 ← 15 tests guarding the leakage-critical logic
├── src/sla.py                  ← API download, features, target, models, inference
├── AGENTS.md                   ← working conventions for this repo
├── data/                       ← CSV downloaded on first run (gitignored, ~73 MB)
└── artifacts/
    ├── model.joblib            ← pipeline + threshold + fitted SLA windows + metrics
    ├── metrics.json            ← full run report
    ├── runs.csv                ← every experiment (MLflow-equivalent record)
    └── error_analysis.md       ← slice tables + fairness check
```

## Setup and run

### Quick demo, under a minute

[`quick_demo.ipynb`](https://colab.research.google.com/github/Fakhrillo/nyc311-sla-breach-prediction/blob/main/quick_demo.ipynb) clones the repo, loads the trained `artifacts/model.joblib`, scores
four example cases and shows bad input being rejected. It downloads no data and trains
nothing, so it is the one to run live. Change any input and re-run the cell to score a
different case.

### Full reproducible pipeline

Click the **full pipeline** badge at the top, or open `demo.ipynb` in Colab via File → Open
notebook → GitHub → this repo, then run all. It queries the API, downloads the slice, trains,
evaluates on unseen data and scores example cases. Nothing needs to be uploaded.

Almost all of the runtime is the data download, and that depends on how busy the NYC API is.
In a clean run on 26 September the download took about 13 minutes and every other cell
together took under a minute. Within the same Colab session the CSV is cached, so running the
notebook a second time skips the download.

### Local

```bash
pip install -r requirements.txt

python test_sla.py      # 15 tests, ~3 s
python train.py         # ~1 min once the data is cached; --quick for a smoke test
```

The first run downloads about 73 MB from the NYC Open Data API, which takes anywhere from a
few minutes to over ten depending on API load. `train.py` writes `artifacts/model.joblib`, `metrics.json`, `runs.csv` and
`error_analysis.md`. Every experiment is logged to MLflow when it is installed, and always to
`runs.csv`, so the pipeline never depends on MLflow being present.

To browse the tracked runs after training:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The store itself is not committed. MLflow writes absolute paths from the machine that ran it
into the database, so `runs.csv` is the committed copy of the same parameters and metrics.

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
# {'breach_probability': 0.3091,
#  'service_window_hours': 51.2,
#  'risk_band': 'medium',
#  'flagged_for_expediting': False,
#  'threshold': 0.3545}
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
