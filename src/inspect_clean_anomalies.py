from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "clean_drives"
SUMMARY_PATH = RESULTS / "summary.csv"

FEATURES = [
    "mean_speed",
    "lane_changes",
    "action_entropy_bits",
    "min_front_gap",
    "min_ttc",
    "mean_reward",
]

Z_THRESHOLD = 2.0


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()

    # Replace infinite safety values so they do not break the detector.
    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column in x.columns:
        x[column] = x[column].fillna(
            x[column].median()
        )

    # Gap and TTC are highly right-skewed.
    # log1p keeps very safe trajectories from dominating numerically.
    x["min_front_gap"] = np.log1p(
        x["min_front_gap"].clip(lower=0)
    )

    x["min_ttc"] = np.log1p(
        x["min_ttc"].clip(lower=0)
    )

    return x


def standardize(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean()
    std = x.std(ddof=0)

    # A constant feature carries no anomaly information.
    std = std.replace(0, 1.0)

    return (x - mean) / std


def anomaly_scores(
    z: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    score = np.sqrt(
        (z ** 2).sum(axis=1)
    )

    flagged = (
        z.abs() > Z_THRESHOLD
    ).any(axis=1)

    dominant_feature = (
        z.abs().idxmax(axis=1)
    )

    return (
        score,
        flagged,
        dominant_feature,
    )


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {SUMMARY_PATH}"
        )

    df = pd.read_csv(SUMMARY_PATH)

    x = prepare_features(df)

    # ---------------------------------------------------------
    # Naive detector:
    # Treat all clean driving styles as one population.
    # ---------------------------------------------------------

    global_z = standardize(x)

    (
        global_score,
        global_flag,
        global_feature,
    ) = anomaly_scores(global_z)

    df["global_anomaly_score"] = global_score
    df["global_flag"] = global_flag
    df["global_dominant_feature"] = global_feature

    # ---------------------------------------------------------
    # Style-aware comparison:
    # Ask how unusual a trajectory is relative to other
    # trajectories of the same legitimate driving style.
    # ---------------------------------------------------------

    style_score = pd.Series(
        index=df.index,
        dtype=float,
    )

    style_flag = pd.Series(
        index=df.index,
        dtype=bool,
    )

    style_feature = pd.Series(
        index=df.index,
        dtype=object,
    )

    for style in df["style"].unique():
        indices = df.index[
            df["style"] == style
        ]

        style_x = x.loc[indices]
        style_z = standardize(style_x)

        (
            scores,
            flags,
            features,
        ) = anomaly_scores(style_z)

        style_score.loc[indices] = scores
        style_flag.loc[indices] = flags
        style_feature.loc[indices] = features

    df["style_anomaly_score"] = style_score
    df["style_flag"] = style_flag
    df["style_dominant_feature"] = style_feature

    # Highest score = most statistically unusual.
    df["global_rank"] = (
        df["global_anomaly_score"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    df["style_rank"] = (
        df["style_anomaly_score"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    output_columns = [
        "trajectory_id",
        "style",
        "crashed",
        "mean_speed",
        "lane_changes",
        "min_front_gap",
        "min_ttc",
        "action_entropy_bits",
        "mean_reward",
        "global_anomaly_score",
        "global_flag",
        "global_dominant_feature",
        "global_rank",
        "style_anomaly_score",
        "style_flag",
        "style_dominant_feature",
        "style_rank",
    ]

    ranking = (
        df[output_columns]
        .sort_values(
            "global_anomaly_score",
            ascending=False,
        )
    )

    ranking.to_csv(
        RESULTS / "clean_anomaly_ranking.csv",
        index=False,
    )

    global_counts = (
        df.groupby("style")["global_flag"]
        .agg(["count", "sum"])
    )

    global_counts["flag_rate"] = (
        global_counts["sum"]
        / global_counts["count"]
    )

    style_counts = (
        df.groupby("style")["style_flag"]
        .agg(["count", "sum"])
    )

    style_counts["flag_rate"] = (
        style_counts["sum"]
        / style_counts["count"]
    )

    print(
        "\n=== TOP CLEAN TRAJECTORIES "
        "BY NAIVE GLOBAL ANOMALY SCORE ==="
    )

    print(
        ranking[
            [
                "trajectory_id",
                "style",
                "crashed",
                "global_anomaly_score",
                "global_flag",
                "global_dominant_feature",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== GLOBAL DETECTOR FLAGS "
        "ON 100% CLEAN DATA ==="
    )

    print(
        global_counts.round(4).to_string()
    )

    print(
        "\n=== STYLE-AWARE DETECTOR FLAGS "
        "ON 100% CLEAN DATA ==="
    )

    print(
        style_counts.round(4).to_string()
    )

    summary = {
        "experiment": "clean_anomaly_inspection",
        "data_label": "clean_only",
        "n_trajectories": int(len(df)),
        "z_threshold": Z_THRESHOLD,
        "global_flagged": int(
            df["global_flag"].sum()
        ),
        "style_aware_flagged": int(
            df["style_flag"].sum()
        ),
        "global_flagged_by_style": {
            style: int(value)
            for style, value
            in df.groupby("style")[
                "global_flag"
            ].sum().items()
        },
        "style_aware_flagged_by_style": {
            style: int(value)
            for style, value
            in df.groupby("style")[
                "style_flag"
            ].sum().items()
        },
    }

    with (
        RESULTS
        / "clean_anomaly_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()