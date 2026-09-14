from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    ROOT
    / "results"
    / "clean_drives"
    / "summary.csv"
)

TEST_DIR = (
    ROOT
    / "results"
    / "obvious_reward_poison"
)

TEST_PATH = TEST_DIR / "summary.csv"

Z_THRESHOLD = 2.0

FEATURES = [
    "mean_speed",
    "lane_changes",
    "action_entropy_bits",
    "min_front_gap",
    "min_ttc",
    "mean_reward",
]


def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    x = df[FEATURES].copy()

    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    x["min_front_gap"] = np.log1p(
        x["min_front_gap"].clip(lower=0)
    )

    x["min_ttc"] = np.log1p(
        x["min_ttc"].clip(lower=0)
    )

    return x


def fit_global_reference(
    clean_x: pd.DataFrame,
):
    medians = clean_x.median()

    clean_x = clean_x.fillna(
        medians
    )

    mean = clean_x.mean()

    std = clean_x.std(
        ddof=0
    ).replace(0, 1.0)

    return medians, mean, std


def apply_reference(
    x: pd.DataFrame,
    medians: pd.Series,
    mean: pd.Series,
    std: pd.Series,
) -> pd.DataFrame:
    x = x.fillna(
        medians
    )

    return (
        x - mean
    ) / std


def score_from_z(
    z: pd.DataFrame,
):
    anomaly_score = np.sqrt(
        (z ** 2).sum(axis=1)
    )

    flagged = (
        z.abs() > Z_THRESHOLD
    ).any(axis=1)

    dominant_feature = (
        z.abs().idxmax(axis=1)
    )

    return (
        anomaly_score,
        flagged,
        dominant_feature,
    )


def global_detector(
    clean_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    clean_x = prepare_features(
        clean_df
    )

    test_x = prepare_features(
        test_df
    )

    (
        medians,
        mean,
        std,
    ) = fit_global_reference(
        clean_x
    )

    z = apply_reference(
        test_x,
        medians,
        mean,
        std,
    )

    return score_from_z(z)


def style_aware_detector(
    clean_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    scores = pd.Series(
        index=test_df.index,
        dtype=float,
    )

    flags = pd.Series(
        index=test_df.index,
        dtype=bool,
    )

    dominant = pd.Series(
        index=test_df.index,
        dtype=object,
    )

    for style in sorted(
        test_df["style"].unique()
    ):
        clean_part = clean_df[
            clean_df["style"] == style
        ]

        test_indices = test_df.index[
            test_df["style"] == style
        ]

        test_part = test_df.loc[
            test_indices
        ]

        clean_x = prepare_features(
            clean_part
        )

        test_x = prepare_features(
            test_part
        )

        (
            medians,
            mean,
            std,
        ) = fit_global_reference(
            clean_x
        )

        z = apply_reference(
            test_x,
            medians,
            mean,
            std,
        )

        (
            style_scores,
            style_flags,
            style_dominant,
        ) = score_from_z(z)

        scores.loc[
            test_indices
        ] = style_scores.values

        flags.loc[
            test_indices
        ] = style_flags.values

        dominant.loc[
            test_indices
        ] = style_dominant.values

    return (
        scores,
        flags,
        dominant,
    )


def classification_metrics(
    truth: pd.Series,
    prediction: pd.Series,
):
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)

    tp = int(
        (truth & prediction).sum()
    )

    fp = int(
        (~truth & prediction).sum()
    )

    tn = int(
        (~truth & ~prediction).sum()
    )

    fn = int(
        (truth & ~prediction).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / len(truth)
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate":
            false_positive_rate,
        "accuracy": accuracy,
    }


def print_metrics(
    name: str,
    metrics: dict,
):
    print(
        f"\n=== {name} ==="
    )

    print(
        f"TP={metrics['tp']}  "
        f"FP={metrics['fp']}  "
        f"TN={metrics['tn']}  "
        f"FN={metrics['fn']}"
    )

    print(
        f"precision="
        f"{metrics['precision']:.3f}"
    )

    print(
        f"recall="
        f"{metrics['recall']:.3f}"
    )

    print(
        f"false_positive_rate="
        f"{metrics['false_positive_rate']:.3f}"
    )

    print(
        f"accuracy="
        f"{metrics['accuracy']:.3f}"
    )


def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            CLEAN_PATH
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            TEST_PATH
        )

    clean_df = pd.read_csv(
        CLEAN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    truth = (
        test_df["data_label"]
        == "poisoned"
    )

    (
        global_score,
        global_flag,
        global_feature,
    ) = global_detector(
        clean_df,
        test_df,
    )

    (
        style_score,
        style_flag,
        style_feature,
    ) = style_aware_detector(
        clean_df,
        test_df,
    )

    test_df[
        "global_anomaly_score"
    ] = global_score

    test_df[
        "global_flag"
    ] = global_flag

    test_df[
        "global_dominant_feature"
    ] = global_feature

    test_df[
        "style_anomaly_score"
    ] = style_score

    test_df[
        "style_flag"
    ] = style_flag

    test_df[
        "style_dominant_feature"
    ] = style_feature

    global_metrics = (
        classification_metrics(
            truth,
            global_flag,
        )
    )

    style_metrics = (
        classification_metrics(
            truth,
            style_flag,
        )
    )

    ranking = test_df.sort_values(
        "global_anomaly_score",
        ascending=False,
    )

    ranking.to_csv(
        TEST_DIR
        / "detector_results.csv",
        index=False,
    )

    print(
        "\n=== HIGHEST GLOBAL "
        "ANOMALY SCORES ==="
    )

    print(
        ranking[
            [
                "trajectory_id",
                "style",
                "data_label",
                "mean_reward",
                "global_anomaly_score",
                "global_flag",
                "global_dominant_feature",
            ]
        ]
        .head(15)
        .round(4)
        .to_string(index=False)
    )

    print_metrics(
        "GLOBAL DETECTOR",
        global_metrics,
    )

    print_metrics(
        "STYLE-AWARE DETECTOR",
        style_metrics,
    )

    poisoned = test_df[
        test_df["data_label"]
        == "poisoned"
    ]

    print(
        "\n=== POISONED "
        "TRAJECTORIES ==="
    )

    print(
        poisoned[
            [
                "trajectory_id",
                "style",
                "mean_reward",
                "global_anomaly_score",
                "global_flag",
                "style_anomaly_score",
                "style_flag",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    result = {
        "experiment":
            "obvious_reward_poison_detection",
        "z_threshold":
            Z_THRESHOLD,
        "n_total":
            int(len(test_df)),
        "n_poisoned":
            int(truth.sum()),
        "global_detector":
            global_metrics,
        "style_aware_detector":
            style_metrics,
    }

    with (
        TEST_DIR
        / "detector_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()