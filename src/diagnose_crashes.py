from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "clean_drives"
TRAJECTORIES = RESULTS / "trajectories.jsonl"


LANE_CHANGE_ACTIONS = {
    "LANE_LEFT",
    "LANE_RIGHT",
}


def nearest_vehicle(observation):
    """
    HighwayEnv Kinematics observation:

        [presence, x, y, vx, vy]

    With absolute=False, non-ego vehicles are represented relative to ego.
    """

    obs = np.asarray(observation, dtype=float)

    candidates = []

    # Row 0 represents the ego vehicle.
    for row in obs[1:]:
        presence, x, y, vx, vy = row

        if presence < 0.5:
            continue

        distance = math.hypot(x, y)

        candidates.append(
            {
                "distance": distance,
                "relative_x": x,
                "relative_y": y,
                "relative_vx": vx,
                "relative_vy": vy,
            }
        )

    if not candidates:
        return None

    return min(candidates, key=lambda x: x["distance"])


def classify_collision(
    relative_vehicle,
    recent_lane_change,
):
    if relative_vehicle is None:
        return "unknown"

    x = relative_vehicle["relative_x"]
    y = relative_vehicle["relative_y"]

    # Near the ego vehicle laterally.
    approximately_same_lane = abs(y) < 1.8

    if approximately_same_lane:
        if x >= 0:
            return "front_collision"
        return "rear_collision"

    if recent_lane_change:
        return "lane_change_or_side"

    return "side_or_unclear"


def find_crash_transition(transitions):
    for index, transition in enumerate(transitions):
        before = transition["ego_before"]["crashed"]
        after = transition["ego_after"]["crashed"]

        if not before and after:
            return index

    return None


def analyse_trajectory(record):
    transitions = record["transitions"]
    summary = record["summary"]

    crash_index = find_crash_transition(transitions)

    base = {
        "trajectory_id": record["trajectory_id"],
        "style": record["style"],
        "crashed": summary["crashed"],
        "return": summary["return"],
        "mean_speed": summary["mean_speed"],
        "lane_changes": summary["lane_changes"],
        "min_ttc": summary["min_ttc"],
        "min_front_gap": summary["min_front_gap"],
    }

    if crash_index is None:
        return {
            **base,
            "crash_step": None,
            "last_action": None,
            "recent_lane_change": False,
            "front_gap_before_crash": None,
            "ttc_before_crash": None,
            "collision_type": None,
            "relative_x": None,
            "relative_y": None,
            "relative_vx": None,
        }

    crash_transition = transitions[crash_index]

    # Look at the last three decisions, including the crash-producing one.
    recent = transitions[max(0, crash_index - 2): crash_index + 1]

    recent_lane_change = any(
        t["action_label"] in LANE_CHANGE_ACTIONS
        for t in recent
    )

    # Post-step observation is useful because the vehicles should be nearest
    # around the actual collision.
    near = nearest_vehicle(
        crash_transition["next_observation"]
    )

    collision_type = classify_collision(
        near,
        recent_lane_change,
    )

    before = crash_transition["ego_before"]

    return {
        **base,
        "crash_step": crash_index,
        "last_action": crash_transition["action_label"],
        "recent_lane_change": recent_lane_change,
        "front_gap_before_crash": before["front_gap"],
        "ttc_before_crash": before["ttc"],
        "collision_type": collision_type,
        "relative_x": (
            None if near is None
            else near["relative_x"]
        ),
        "relative_y": (
            None if near is None
            else near["relative_y"]
        ),
        "relative_vx": (
            None if near is None
            else near["relative_vx"]
        ),
    }


def finite_median(series):
    values = [
        float(v)
        for v in series
        if pd.notna(v) and np.isfinite(float(v))
    ]

    if not values:
        return None

    return float(np.median(values))


def main():
    if not TRAJECTORIES.exists():
        raise FileNotFoundError(
            f"Could not find {TRAJECTORIES}"
        )

    records = []

    with TRAJECTORIES.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            records.append(
                analyse_trajectory(json.loads(line))
            )

    df = pd.DataFrame(records)

    df.to_csv(
        RESULTS / "crash_diagnosis.csv",
        index=False,
    )

    crashed = df[df["crashed"]].copy()
    safe = df[~df["crashed"]].copy()

    print("\n=== CRASH COUNTS ===")

    counts = (
        df.groupby("style")["crashed"]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "sum": "crashes",
                "mean": "crash_rate",
            }
        )
    )

    print(counts.to_string())

    print("\n=== COLLISION TYPES ===")

    if not crashed.empty:
        types = pd.crosstab(
            crashed["style"],
            crashed["collision_type"],
        )

        print(types.to_string())

    print("\n=== ACTION IMMEDIATELY BEFORE CRASH ===")

    if not crashed.empty:
        actions = pd.crosstab(
            crashed["style"],
            crashed["last_action"],
        )

        print(actions.to_string())

    print("\n=== RECENT LANE CHANGE ===")

    if not crashed.empty:
        lane_change = (
            crashed.groupby("style")["recent_lane_change"]
            .mean()
        )

        print(lane_change.to_string())

    print("\n=== TTC BEFORE CRASH ===")

    for style in sorted(df["style"].unique()):
        subset = crashed[
            crashed["style"] == style
        ]

        median = finite_median(
            subset["ttc_before_crash"]
        )

        print(
            f"{style:>10}: "
            f"{median if median is not None else 'n/a'}"
        )

    print("\n=== MIN TTC: CRASHED VS NON-CRASHED ===")

    comparison = []

    for style in sorted(df["style"].unique()):
        for outcome, subset in [
            (
                "crashed",
                df[
                    (df["style"] == style)
                    & (df["crashed"])
                ],
            ),
            (
                "survived",
                df[
                    (df["style"] == style)
                    & (~df["crashed"])
                ],
            ),
        ]:
            comparison.append(
                {
                    "style": style,
                    "outcome": outcome,
                    "n": len(subset),
                    "median_min_ttc": finite_median(
                        subset["min_ttc"]
                    ),
                    "median_min_front_gap": finite_median(
                        subset["min_front_gap"]
                    ),
                }
            )

    comparison_df = pd.DataFrame(comparison)

    print(comparison_df.to_string(index=False))

    summary = {
        "collision_types": (
            crashed["collision_type"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "actions_before_crash": (
            crashed["last_action"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "fraction_with_recent_lane_change": (
            float(crashed["recent_lane_change"].mean())
            if len(crashed)
            else None
        ),
    }

    with (
        RESULTS / "crash_diagnosis_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()