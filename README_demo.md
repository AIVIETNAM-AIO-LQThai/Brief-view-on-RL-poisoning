# Highway replay demo

This patch adds a first visual demo for the Highway trajectory project.

## Files

- `src/choose_demo_cases.py`
- `src/replay_drive.py`
- `demo/app.py`

## Extra dependency

Add to `requirements.txt`:

```text
streamlit>=1.36,<2
```

## How to use

1. Make sure the clean-driving experiment has already been run:

```powershell
python -m src.collect_clean_drives
```

2. Choose representative demo trajectories:

```powershell
python -m src.choose_demo_cases
```

3. Launch the Streamlit demo:

```powershell
streamlit run demo/app.py
```

## What the audience sees

- A replay of the actual highway simulation
- Live statistics that correspond to each frame:
  - action
  - speed
  - lane
  - front gap
  - time-to-collision
  - safety band
  - crash status
- A fixed summary for the full trajectory
- A time-series chart of speed, front gap, and TTC
- A step-by-step table for deeper inspection
