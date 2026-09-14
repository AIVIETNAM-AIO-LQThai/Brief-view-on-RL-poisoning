# Stage 1B patch

Copy these files into the existing Highway project, preserving the existing
Stage 1A outputs/checkpoint.

Run:

```powershell
python -m src.stage01b_collect_tuned_clean
```

New results are written under:

`outputs/stage01b_tuned/`

This run does not overwrite the first failed baseline.

The script has explicit acceptance criteria. If it prints `NEEDS TUNING`,
do not move to anomaly detection yet.
