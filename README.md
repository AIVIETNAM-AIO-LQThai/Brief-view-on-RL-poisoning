# Scientific Thinking — Highway Trajectory Poisoning

This project demonstrates how our reasoning about poisoned RL trajectories evolves:

1. **Stage 0 — Environment fixed**
   - Define the simulator, observation/action spaces, data schema, and seeds.
2. **Stage 1 — Clean heterogeneous trajectories**
   - Generate cautious, normal, and aggressive *clean* driving trajectories.
   - Establish the normal range before attempting anomaly detection.
3. **Stage 2 — Naive anomaly inspection**
   - Rank only clean trajectories using simple statistical anomaly scores.
   - Show that unusual clean driving can create false positives.
4. **Stage 3 — Obvious poisoning**
   - Inject easy-to-detect reward/state corruption.
5. **Stage 4 — Subtle poisoning**
   - Use small, physically plausible changes that evade simple anomaly checks.
6. **Stage 5 — Intricate / decision-targeted poisoning**
   - Modify strategically influential transitions while preserving plausibility.
7. **Stage 6 — Final demo + LaTeX paper**
   - Compare detectors and explain how failed assumptions changed the solution.

## Environment

We use `highway-v0` from HighwayEnv.

The simulator is configured explicitly in `src/config.py`; we do not depend on undocumented defaults.

## Install

```bash
python -m venv .venv
# Windows PowerShell:
.\\.venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Stage 1

Run:

```bash
python -m src.stage01_collect_clean
```

This writes:

- `outputs/clean_trajectories.jsonl`
- `outputs/clean_summary.csv`
- `outputs/stage01_metrics.json`
- `outputs/stage01_representatives.json`
- `CHECKPOINT.md`
- `project_state.json`

Every episode stores its simulator seed and complete action sequence, so we can replay the same episode later for the visual demo.

## Important scientific rule

Do **not** introduce poisoning before Stage 2.

We first need to understand how strange legitimate behavior can look. Otherwise, an anomaly detector can appear successful merely because we defined "normal" too narrowly.
