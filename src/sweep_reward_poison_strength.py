from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate_reward_poison import (
    classification_metrics,
    global_detector,
    style_aware_detector,
)


ROOT = Path(__file__).resolve().parents[1]

CLEAN_SUMMARY = (
    ROOT
    / "results"
    / "clean_drives"
    / "summary.csv"
)

MANIFEST_PATH = (
    ROOT
    / "results"
    / "obvious_reward_poison"
    / "poison_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "reward_poison_strength"
)


REWARD_DELTAS = [
    1.00,
    0.75,
    0.50,
    0.25,
    0.10,
    0.05,
]


def build_test_dataset(
    clean_df: pd.DataFrame,
    manifest: pd.DataFrame,
    reward_delta: float,
) -> pd.DataFrame:
    """
    Start from the accepted clean trajectory summary.

    Poison exactly the same trajectories and transition locations
    used by the obvious reward attack.

    The only thing varied across experiments is reward magnitude.
    """

    test_df = clean_df.copy()

    test_df["data_label"] = "clean"
    test_df["attack_type"] = None
    test_df["reward_delta"] = 0.0

    poisoned_ids = set(
        manifest["trajectory_id"].unique()
    )

    counts = (
        manifest
        .groupby("trajectory_id")
        .size()
        .to_dict()
    )

    for trajectory_id in poisoned_ids:
        mask = (
            test_df["trajectory_id"]
            == trajectory_id
        )

        if not mask.any():
            raise ValueError(
                f"Could not find {trajectory_id} "
                "in clean summary."
            )

        n_modified = int(
            counts[trajectory_id]
        )

        steps = int(
            test_df.loc[
                mask,
                "steps",
            ].iloc[0]
        )

        clean_mean_reward = float(
            test_df.loc[
                mask,
                "mean_reward",
            ].iloc[0]
        )

        clean_return = float(
            test_df.loc[
                mask,
                "return",
            ].iloc[0]
        )

        mean_shift = (
            reward_delta
            * n_modified
            / steps
        )

        return_shift = (
            reward_delta
            * n_modified
        )

        test_df.loc[
            mask,
            "mean_reward",
        ] = (
            clean_mean_reward
            + mean_shift
        )

        test_df.loc[
            mask,
            "return",
        ] = (
            clean_return
            + return_shift
        )

        test_df.loc[
            mask,
            "data_label",
        ] = "poisoned"

        test_df.loc[
            mask,
            "attack_type",
        ] = "additive_reward_poison"

        test_df.loc[
            mask,
            "reward_delta",
        ] = reward_delta

    return test_df


def evaluate_one_strength(
    clean_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reward_delta: float,
) -> list[dict]:

    truth = (
        test_df["data_label"]
        == "poisoned"
    )

    (
        global_score,
        global_flag,
        _,
    ) = global_detector(
        clean_df,
        test_df,
    )

    (
        style_score,
        style_flag,
        _,
    ) = style_aware_detector(
        clean_df,
        test_df,
    )

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

    rows = []

    for (
        detector_name,
        metrics,
        scores,
    ) in [
        (
            "global",
            global_metrics,
            global_score,
        ),
        (
            "style_aware",
            style_metrics,
            style_score,
        ),
    ]:

        poisoned_scores = scores[
            truth
        ]

        clean_scores = scores[
            ~truth
        ]

        rows.append(
            {
                "reward_delta":
                    reward_delta,

                "detector":
                    detector_name,

                **metrics,

                "mean_poison_score":
                    float(
                        poisoned_scores.mean()
                    ),

                "mean_clean_score":
                    float(
                        clean_scores.mean()
                    ),

                "min_poison_score":
                    float(
                        poisoned_scores.min()
                    ),

                "max_clean_score":
                    float(
                        clean_scores.max()
                    ),
            }
        )

    return rows


def main() -> None:

    if not CLEAN_SUMMARY.exists():
        raise FileNotFoundError(
            CLEAN_SUMMARY
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    clean_df = pd.read_csv(
        CLEAN_SUMMARY
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for reward_delta in REWARD_DELTAS:

        test_df = build_test_dataset(
            clean_df,
            manifest,
            reward_delta,
        )

        rows.extend(
            evaluate_one_strength(
                clean_df,
                test_df,
                reward_delta,
            )
        )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_DIR
        / "strength_sweep.csv",
        index=False,
    )

    print(
        "\n=== REWARD POISON "
        "STRENGTH SWEEP ==="
    )

    display = result[
        [
            "reward_delta",
            "detector",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "false_positive_rate",
            "mean_poison_score",
            "mean_clean_score",
        ]
    ]

    print(
        display
        .round(3)
        .to_string(index=False)
    )

    print(
        "\n=== GLOBAL DETECTOR "
        "RECALL BY ATTACK STRENGTH ==="
    )

    global_only = result[
        result["detector"]
        == "global"
    ]

    for _, row in (
        global_only
        .sort_values(
            "reward_delta",
            ascending=False,
        )
        .iterrows()
    ):
        print(
            f"delta={row['reward_delta']:.2f}"
            f" | recall={row['recall']:.3f}"
            f" | precision={row['precision']:.3f}"
            f" | TP={int(row['tp'])}"
            f" | FN={int(row['fn'])}"
        )


if __name__ == "__main__":
    main()