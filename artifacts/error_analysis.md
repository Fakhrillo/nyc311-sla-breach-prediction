# Error analysis (test set, unseen)

- Test cases: 73,391 | actual breach rate: 0.245
- Threshold: 0.365 | flagged: 0.190 of arrivals
- Missed breaches (FN): 12,169 | false alarms (FP): 8,142

## Performance by slice

| slice | n | breach rate | recall | precision |
| --- | ---: | ---: | ---: | ---: |
| agency: NYPD | 44,787 | 0.257 | 0.331 | 0.402 |
| agency: HPD | 25,133 | 0.206 | 0.217 | 0.380 |
| agency: DOT | 3,312 | 0.364 | 0.666 | 0.562 |
| borough: BROOKLYN | 22,861 | 0.226 | 0.138 | 0.434 |
| borough: QUEENS | 17,952 | 0.298 | 0.576 | 0.404 |
| borough: BRONX | 17,344 | 0.255 | 0.219 | 0.425 |
| borough: MANHATTAN | 13,188 | 0.194 | 0.333 | 0.399 |
| borough: STATEN ISLAND | 2,035 | 0.218 | 0.399 | 0.702 |
| channel: ONLINE | 29,491 | 0.235 | 0.305 | 0.389 |
| channel: PHONE | 23,368 | 0.240 | 0.289 | 0.409 |
| channel: MOBILE | 18,237 | 0.253 | 0.348 | 0.424 |
| channel: UNKNOWN | 2,295 | 0.353 | 0.565 | 0.596 |
| backlog < 261 | 18,197 | 0.229 | 0.205 | 0.381 |
| backlog 261-20026 | 36,846 | 0.262 | 0.412 | 0.438 |
| backlog > 20026 | 18,348 | 0.225 | 0.231 | 0.370 |
| weekend arrival | 20,328 | 0.263 | 0.328 | 0.413 |
| weekday arrival | 53,063 | 0.238 | 0.320 | 0.417 |
| overnight (00-06h) | 9,826 | 0.234 | 0.281 | 0.355 |
| business hours (09-17h) | 34,243 | 0.239 | 0.301 | 0.427 |

## Fairness check

Recall across boroughs and intake channels ranges from **0.138** (borough: BROOKLYN) to **0.576** (borough: QUEENS), a gap of 0.438.

This matters more here than in a commercial setting. Expediting is a public service being rationed, so a model that systematically surfaces cases from one borough over another redistributes municipal attention along geographic lines. Residents of **borough: BROOKLYN** would see their slow cases escalated least often, and nothing in the data tells us their cases are less urgent -- only that the historical queue treated them differently, which the model then learns to repeat.

Intake channel carries the same risk in a different shape: if phone reports are escalated less than online ones, the system quietly penalises whoever is less likely to file online.

## Hardest cases

- False alarms: median backlog 352, median window 2.8 h
- Missed breaches: median backlog 340, median window 3.4 h

The target measures whether a case took longer than its type usually takes. It does not measure whether the resolution was any good, nor whether the case mattered. A fast closure and a good outcome are not the same event, and this model only ever sees the first.
