# Error analysis (test set, unseen)

- Test cases: 72,918 | actual breach rate: 0.240
- Threshold: 0.354 | flagged: 0.203 of arrivals
- Missed breaches (FN): 11,472 | false alarms (FP): 8,797

## Performance by slice

| slice | n | breach rate | recall | precision |
| --- | ---: | ---: | ---: | ---: |
| agency: NYPD | 44,497 | 0.252 | 0.367 | 0.397 |
| agency: HPD | 25,139 | 0.206 | 0.227 | 0.363 |
| agency: DOT | 3,123 | 0.325 | 0.655 | 0.593 |
| borough: BROOKLYN | 22,747 | 0.223 | 0.151 | 0.408 |
| borough: QUEENS | 17,792 | 0.292 | 0.620 | 0.398 |
| borough: BRONX | 17,275 | 0.252 | 0.242 | 0.433 |
| borough: MANHATTAN | 13,115 | 0.190 | 0.356 | 0.392 |
| borough: STATEN ISLAND | 1,980 | 0.196 | 0.239 | 0.564 |
| channel: ONLINE | 29,300 | 0.230 | 0.321 | 0.386 |
| channel: PHONE | 23,342 | 0.239 | 0.326 | 0.397 |
| channel: MOBILE | 18,172 | 0.250 | 0.378 | 0.413 |
| channel: UNKNOWN | 2,104 | 0.295 | 0.511 | 0.691 |
| backlog < 261 | 18,107 | 0.226 | 0.242 | 0.383 |
| backlog 261-20042 | 36,595 | 0.254 | 0.435 | 0.427 |
| backlog > 20042 | 18,216 | 0.226 | 0.239 | 0.357 |
| weekend arrival | 20,149 | 0.257 | 0.361 | 0.407 |
| weekday arrival | 52,769 | 0.233 | 0.337 | 0.406 |
| overnight (00-06h) | 9,717 | 0.226 | 0.334 | 0.349 |
| business hours (09-17h) | 34,047 | 0.235 | 0.322 | 0.420 |

## Fairness check

Recall across boroughs and intake channels ranges from **0.151** (borough: BROOKLYN) to **0.620** (borough: QUEENS), a gap of 0.469.

This matters more here than it would in a commercial setting. Expediting is a public service being rationed, so a model that systematically surfaces cases from one borough over another is redistributing municipal attention along geographic lines. Residents of **borough: BROOKLYN** would see their slow cases escalated least often, and nothing in the data says their cases are less urgent. It says only that the historical queue treated them differently, which the model then learned to repeat.

Intake channel carries the same risk in a different shape. If phone reports are escalated less often than online ones, the system quietly penalises whoever is least likely to file online.

## Hardest cases

- False alarms: median backlog 349, median window 2.8 h
- Missed breaches: median backlog 343, median window 3.4 h

One limitation worth stating next to these numbers: the target measures whether a case took longer than its type usually takes. It does not measure whether the resolution was any good, or whether the case mattered. A fast closure and a good outcome are different events, and this model only ever sees the first.
