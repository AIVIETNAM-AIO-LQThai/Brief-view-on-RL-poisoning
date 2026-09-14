from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = (
    ROOT
    / "results"
    / "clean_drives"
)

CLEAN_TRAJECTORIES = (
    CLEAN_DIR
    / "trajectories.jsonl"
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
    / "stealthy_reward_poison"
)


REWARD_DELTA = 0.25


def load_clean_records() -> list[dict]:
    if not CLEAN_TRAJECTORIES.exists():
        raise FileNotFoundError(
            CLEAN_TRAJECTORIES
        )

    records = []

    with CLEAN_TRAJECTORIES.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    return pd.read_csv(
        MANIFEST_PATH
    )


def poison_record(
    record: dict,
    poisoned_indices: set[int],
) -> tuple[dict, list[dict]]:

    poisoned = copy.deepcopy(
        record
    )

    transitions = poisoned[
        "transitions"
    ]

    manifest_rows = []

    original_return = float(
        poisoned["summary"]["return"]
    )

    original_mean_reward = float(
        poisoned["summary"][
            "mean_reward"
        ]
    )

    for index, transition in enumerate(
        transitions
    ):
        if index not in poisoned_indices:
            transition[
                "reward_poisoned"
            ] = False
            continue

        clean_reward = float(
            transition["reward"]
        )

        poisoned_reward = (
            clean_reward
            + REWARD_DELTA
        )

        transition["clean_reward"] = (
            clean_reward
        )

        transition["reward"] = float(
            poisoned_reward
        )

        transition[
            "reward_poisoned"
        ] = True

        manifest_rows.append(
            {
                "trajectory_id":
                    poisoned[
                        "trajectory_id"
                    ],

                "style":
                    poisoned["style"],

                "transition_index":
                    index,

                "action":
                    transition[
                        "action_label"
                    ],

                "clean_reward":
                    clean_reward,

                "poisoned_reward":
                    poisoned_reward,

                "reward_delta":
                    REWARD_DELTA,

                "front_gap_before":
                    transition[
                        "ego_before"
                    ]["front_gap"],

                "ttc_before":
                    transition[
                        "ego_before"
                    ]["ttc"],
            }
        )

    new_rewards = [
        float(t["reward"])
        for t in transitions
    ]

    summary = poisoned["summary"]

    summary["clean_return"] = (
        original_return
    )

    summary["clean_mean_reward"] = (
        original_mean_reward
    )

    summary["return"] = float(
        np.sum(new_rewards)
    )

    summary["mean_reward"] = float(
        np.mean(new_rewards)
    )

    summary["poisoned"] = True

    summary[
        "n_poisoned_transitions"
    ] = len(poisoned_indices)

    summary[
        "poisoned_transition_fraction"
    ] = (
        len(poisoned_indices)
        / len(transitions)
    )

    summary["reward_delta"] = (
        REWARD_DELTA
    )

    poisoned["data_label"] = (
        "poisoned"
    )

    poisoned["attack_type"] = (
        "stealthy_reward_poison"
    )

    return (
        poisoned,
        manifest_rows,
    )


def mark_clean(
    record: dict,
) -> dict:

    clean = copy.deepcopy(
        record
    )

    clean["data_label"] = "clean"
    clean["attack_type"] = None

    clean["summary"][
        "poisoned"
    ] = False

    clean["summary"][
        "n_poisoned_transitions"
    ] = 0

    clean["summary"][
        "poisoned_transition_fraction"
    ] = 0.0

    clean["summary"][
        "reward_delta"
    ] = 0.0

    clean["summary"][
        "clean_return"
    ] = float(
        clean["summary"]["return"]
    )

    clean["summary"][
        "clean_mean_reward"
    ] = float(
        clean["summary"][
            "mean_reward"
        ]
    )

    return clean


def make_summary(
    records: list[dict],
) -> pd.DataFrame:

    rows = []

    for record in records:
        s = record["summary"]

        rows.append(
            {
                "trajectory_id":
                    record[
                        "trajectory_id"
                    ],

                "style":
                    record["style"],

                "data_label":
                    record[
                        "data_label"
                    ],

                "attack_type":
                    record[
                        "attack_type"
                    ],

                "crashed":
                    s["crashed"],

                "steps":
                    s["steps"],

                "mean_speed":
                    s["mean_speed"],

                "lane_changes":
                    s["lane_changes"],

                "action_entropy_bits":
                    s[
                        "action_entropy_bits"
                    ],

                "min_front_gap":
                    s["min_front_gap"],

                "min_ttc":
                    s["min_ttc"],

                "clean_return":
                    s[
                        "clean_return"
                    ],

                "return":
                    s["return"],

                "clean_mean_reward":
                    s[
                        "clean_mean_reward"
                    ],

                "mean_reward":
                    s["mean_reward"],

                "reward_delta":
                    s["reward_delta"],

                "n_poisoned_transitions":
                    s[
                        "n_poisoned_transitions"
                    ],

                "poisoned_transition_fraction":
                    s[
                        "poisoned_transition_fraction"
                    ],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:

    records = load_clean_records()

    manifest = load_manifest()

    targets = set(
        manifest[
            "trajectory_id"
        ].unique()
    )

    indices_by_trajectory = {
        trajectory_id: set(
            int(x)
            for x in group[
                "transition_index"
            ].tolist()
        )
        for trajectory_id, group
        in manifest.groupby(
            "trajectory_id"
        )
    }

    output_records = []
    attack_manifest = []

    for record in records:

        trajectory_id = record[
            "trajectory_id"
        ]

        if trajectory_id in targets:

            poisoned, rows = (
                poison_record(
                    record,
                    indices_by_trajectory[
                        trajectory_id
                    ],
                )
            )

            output_records.append(
                poisoned
            )

            attack_manifest.extend(
                rows
            )

        else:

            output_records.append(
                mark_clean(record)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        OUTPUT_DIR
        / "trajectories.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in output_records:

            f.write(
                json.dumps(
                    record,
                    allow_nan=True,
                )
                + "\n"
            )

    summary_df = make_summary(
        output_records
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    pd.DataFrame(
        attack_manifest
    ).to_csv(
        OUTPUT_DIR
        / "poison_manifest.csv",
        index=False,
    )

    metadata = {
        "experiment":
            "stealthy_reward_poison",

        "source":
            "accepted_clean_driving_baseline",

        "selection_source":
            "obvious_reward_poison_manifest",

        "reward_delta":
            REWARD_DELTA,

        "n_total_trajectories":
            len(output_records),

        "n_poisoned_trajectories":
            len(targets),

        "target_trajectory_ids":
            sorted(targets),
    }

    with (
        OUTPUT_DIR
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    poisoned_df = summary_df[
        summary_df[
            "data_label"
        ] == "poisoned"
    ]

    print(
        "\n=== STEALTHY REWARD POISON ==="
    )

    print(
        f"Reward delta: "
        f"{REWARD_DELTA}"
    )

    print(
        f"Poisoned trajectories: "
        f"{len(poisoned_df)}"
    )

    print()

    for _, row in (
        poisoned_df.iterrows()
    ):

        print(
            f"{row['trajectory_id']:>18}"
            f" | {row['style']:>10}"
            f" | clean="
            f"{row['clean_mean_reward']:.3f}"
            f" | poisoned="
            f"{row['mean_reward']:.3f}"
            f" | shift="
            f"{row['mean_reward'] - row['clean_mean_reward']:.3f}"
        )


if __name__ == "__main__":
    main()