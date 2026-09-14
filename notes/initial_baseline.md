# Initial clean baseline - rejected design

## What we tested
60 clean HighwayEnv trajectories:
- 20 cautious
- 20 normal
- 20 aggressive

No poisoning was present.

## Measured results

| Style | Crash rate | Return mean | Speed mean (m/s) | Lane changes mean | Median min distance (m) | Action entropy |
|---|---:|---:|---:|---:|---:|---:|
| cautious | 0.20 | 20.9580 | 22.0415 | 0.50 | 4.0426 | 1.0683 |
| normal | 0.50 | 17.9317 | 24.1973 | 1.80 | 4.0117 | 1.2187 |
| aggressive | 0.85 | 15.9095 | 28.2617 | 3.05 | 4.0019 | 1.6375 |

## What worked
Driving style is clearly represented:
- speed increases across cautious → normal → aggressive;
- lane-change frequency increases;
- action entropy increases.

## What failed
Crash outcome is too strongly confounded with style:
- normal crashes in 50% of episodes;
- aggressive crashes in 85% of episodes.

A later anomaly detector could appear successful simply by learning that crashed
trajectories are unusual. That would weaken the scientific argument.

The median Euclidean minimum-distance feature is also almost identical (~4 m)
for all styles and is therefore not useful enough by itself.

## Decision
Do NOT proceed to poisoning or anomaly detection yet.

Preserve this run as evidence that experiment design itself can fail.

## Revision
The revised clean baseline will:
1. keep three clean driving styles;
2. use safer, velocity-aware lane-change checks;
3. slightly reduce traffic density;
4. record front-gap and time-to-collision-style safety features;
5. apply acceptance criteria before anomaly inspection begins.

