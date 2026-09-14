# Highway Trajectory Poisoning

A small reinforcement-learning experiment developed for a Scientific Thinking project.

The goal is not only to demonstrate trajectory poisoning, but to document how the reasoning about the problem changes as increasingly difficult examples are tested.

## Research flow

### Establish clean driving behavior

We use `highway-v0` from HighwayEnv to generate legitimate driving trajectories with three behavior profiles:

- cautious
- normal
- aggressive

The profiles intentionally differ in speed, lane-changing behavior, and risk tolerance.

Before introducing poisoning, we first determine how much variation can naturally occur among legitimate trajectories.

### Inspect unusual clean trajectories

Simple anomaly measures are applied to clean data only.

This tests an important initial assumption:

> Does unusual behavior imply corrupted behavior?

False positives at this point demonstrate that statistical abnormality alone is not enough to identify poisoning.

### Introduce obvious poisoning

Rewards or transitions are modified in clearly abnormal ways.

These examples establish which attacks can be detected by simple statistical and physical-consistency checks.

### Test stealthy poisoning

The modifications are made smaller and physically plausible.

This tests whether a poisoned trajectory can remain inside the statistical range of clean behavior while still affecting the information available to a learner.

### Target influential decisions

Later experiments focus on transitions near important driving decisions rather than modifying samples uniformly.

The central question becomes whether small, strategically chosen modifications can produce disproportionately large changes in the learned policy.

### Evaluate policy influence

The final analysis compares data-level abnormality with downstream policy impact.

This allows us to distinguish between:

- unusual but legitimate data,
- obviously corrupted data,
- plausible but poisoned data,
- and influential poisoned data.

## Environment

The project uses:

- `highway-env==1.12.1`
- `gymnasium==1.3.0`
- NumPy
- pandas

The simulator configuration is defined in:

```text
src/experiment_config.py
```

The driving behavior is defined in:

```text
src/driving_policy.py
```

## Setup

Create and activate a Python environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Generate clean trajectories

Run:

```powershell
python -m src.collect_clean_drives
```

The experiment generates:

```text
results/clean_drives/
├── summary.csv
├── metrics.json
├── run_summary.md
└── trajectories.jsonl
```

`trajectories.jsonl` contains the full generated trajectory data. It is reproducible from the fixed simulator and policy seeds and is therefore excluded from Git.

## Previous baseline

The first clean-driving design produced excessive crash rates, especially for the aggressive profile.

Instead of discarding that experiment, its results are preserved under:

```text
results/initial_clean_baseline/
```

and the reasoning behind rejecting the design is documented in:

```text
notes/initial_baseline.md
```

This failed baseline is part of the Scientific Thinking narrative: an experiment can reveal that the experimental design itself is flawed before it reveals anything about the original hypothesis.

## Scientific constraint

Poisoning should not be introduced until the clean-driving baseline and clean-data anomaly analysis are understood.

Otherwise, a detector may appear successful simply because legitimate behavior was defined too narrowly.
