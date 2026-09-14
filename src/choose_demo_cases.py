from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "clean_drives"
SUMMARY_PATH = RESULTS / "summary.csv"
ANOMALY_PATH = RESULTS / "clean_anomaly_ranking.csv"
DEMO_DIR = ROOT / "demo"
DEMO_CASES_PATH = DEMO_DIR / "demo_cases.json"


STYLE_BLURBS = {
    "cautious": "Lower target speed, very few lane changes, larger safety margins.",
    "normal": "Balanced speed and overtaking behavior with moderate safety margins.",
    "aggressive": "Higher target speed, more overtaking, and smaller safety margins.",
}


def choose_representative(part: pd.DataFrame) -> pd.Series:
    if part.empty:
        raise ValueError("No trajectories available for this style.")

    non_crashed = part[part["crashed"] == False]  # noqa: E712
    source = non_crashed if not non_crashed.empty else part

    target_return = float(source["return"].median())
    scored = source.copy()
    scored["return_distance"] = (scored["return"] - target_return).abs()
    scored = scored.sort_values(
        by=["return_distance", "lane_changes", "action_entropy_bits", "trajectory_id"]
    )
    return scored.iloc[0]


def choose_risky_clean(part: pd.DataFrame) -> pd.Series | None:
    crashed = part[part["crashed"] == True]  # noqa: E712
    if crashed.empty:
        return None

    target_return = float(crashed["return"].median())
    scored = crashed.copy()
    scored["return_distance"] = (scored["return"] - target_return).abs()
    scored = scored.sort_values(
        by=["return_distance", "lane_changes", "action_entropy_bits", "trajectory_id"]
    )
    return scored.iloc[0]


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SUMMARY_PATH}. Run python -m src.collect_clean_drives first."
        )

    df = pd.read_csv(SUMMARY_PATH)
    if not ANOMALY_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {ANOMALY_PATH}. "
            "Run python -m src.inspect_clean_anomalies first."
        )

    anomaly_df = pd.read_csv(ANOMALY_PATH)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    cases = []

    for style in ["cautious", "normal", "aggressive"]:
        row = choose_representative(df[df["style"] == style])
        cases.append(
            {
                "trajectory_id": row["trajectory_id"],
                "title": f"{style.capitalize()} clean drive",
                "style": style,
                "data_label": "clean",
                "subtitle": STYLE_BLURBS[style],
                "why_selected": "Representative drive near the middle of this style's clean behavior.",
            }
        )

    risky = choose_risky_clean(df[df["style"] == "aggressive"])
    if risky is not None:
        cases.append(
            {
                "trajectory_id": risky["trajectory_id"],
                "title": "Risky but still clean",
                "style": "aggressive",
                "data_label": "clean",
                "subtitle": "A clean trajectory can still crash or look suspicious.",
                "why_selected": "Useful for showing that risky behavior does not automatically mean poisoning.",
            }
        )

    already_selected = {
        case["trajectory_id"]
        for case in cases
    }

    false_positives = anomaly_df[
        anomaly_df["global_flag"] == True  # noqa: E712
    ].copy()

    false_positives = false_positives[
        ~false_positives["trajectory_id"].isin(
            already_selected
        )
    ]

    if not false_positives.empty:
        false_positive = (
            false_positives
            .sort_values(
                "global_anomaly_score",
                ascending=False,
            )
            .iloc[0]
        )

        cases.append(
            {
                "trajectory_id": false_positive[
                    "trajectory_id"
                ],
                "title": "Clean but flagged suspicious",
                "style": false_positive["style"],
                "data_label": "clean",
                "subtitle": (
                    "This trajectory is known to be clean, "
                    "but the naive statistical detector flags it."
                ),
                "why_selected": (
                    "Demonstrates that unusual legitimate "
                    "behavior can be mistaken for poisoning."
                ),
            }
        )

    with DEMO_CASES_PATH.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"Saved {len(cases)} demo cases to {DEMO_CASES_PATH}")
    for case in cases:
        print(f"- {case['title']}: {case['trajectory_id']}")


if __name__ == "__main__":
    main()
