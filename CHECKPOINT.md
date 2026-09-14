# Scientific Thinking Highway Project — Checkpoint

## Current stage
**Stage 1 — Clean heterogeneous trajectory baseline COMPLETE**

## Scientific question
How much variation exists among trajectories that are all legitimate and unpoisoned?

## Fixed environment
- Environment: `highway-v0`
- Base seed: `20260915`
- Driving styles: cautious, normal, aggressive
- Episodes per style: `20`
- Total clean trajectories: 60

## Key result
This stage intentionally contains no attack. Any extreme trajectory is still clean.

### Overall
- Crash rate: 0.5167
- Mean return: 18.2664
- Return std: 7.2436
- Mean speed: 24.8335 m/s
- Mean lane changes: 1.7833

## Next stage
**Stage 2 — Naive anomaly inspection on CLEAN data only.**

We will compute simple trajectory-level anomaly scores from:
- return
- mean speed
- maximum speed
- lane changes
- minimum vehicle distance
- action entropy

The detector will rank suspicious trajectories even though every trajectory is clean.
That false-positive behaviour is the first deliberate failure in the scientific-thinking narrative.
