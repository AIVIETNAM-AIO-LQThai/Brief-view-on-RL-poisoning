from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT / "results" / "clean_drives"
SOURCE_TRAJECTORIES = SOURCE_DIR / "trajectories.jsonl"

OUTPUT_DIR = ROOT / "results" / "obvious_reward_poison"

ATTACK_SEED = 20260915

# Poison exactly two trajectories from each legitimate driving style.
POISONED_TRAJECTORIES_PER_STYLE = 2

# Modify 20% of transitions inside each selected trajectory.
TRANSITION_POISON_FRACTION = 0.20

# Clean HighwayEnv rewards are roughly around 0–1.
# 5.0 is intentionally conspicuous for this first attack.
POISONED_REWARD = 5.0

STYLE_ORDER = [
    "cautious",
    "normal",
    "aggressive",
]


def load_records() -> list[dict]:
    if not SOURCE_TRAJECTORIES.exists():
        raise FileNotFoundError(
            f"Could not find {SOURCE_TRAJECTORIES}"
        )

    records = []

    with SOURCE_TRAJECTORIES.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def choose_target_trajectories(
    records: list[dict],
) -> set[str]:
    rng = np.random.default_rng(
        ATTACK_SEED
    )

    selected = set()

    for style in STYLE_ORDER:
        candidates = sorted(
            record["trajectory_id"]
            for record in records
            if record["style"] == style
        )

        if (
            len(candidates)
            < POISONED_TRAJECTORIES_PER_STYLE
        ):
            raise ValueError(
                f"Not enough {style} trajectories."
            )

        chosen = rng.choice(
            candidates,
            size=POISONED_TRAJECTORIES_PER_STYLE,
            replace=False,
        )

        selected.update(
            str(x)
            for x in chosen
        )

    return selected


def poison_record(
    record: dict,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    poisoned = copy.deepcopy(record)

    transitions = poisoned["transitions"]

    n_poisoned = max(
        1,
        round(
            len(transitions)
            * TRANSITION_POISON_FRACTION
        ),
    )

    chosen_indices = sorted(
        int(x)
        for x in rng.choice(
            len(transitions),
            size=n_poisoned,
            replace=False,
        )
    )

    manifest_rows = []

    original_return = float(
        poisoned["summary"]["return"]
    )

    original_mean_reward = float(
        poisoned["summary"]["mean_reward"]
    )

    for transition_index in chosen_indices:
        transition = transitions[
            transition_index
        ]

        clean_reward = float(
            transition["reward"]
        )

        transition["clean_reward"] = (
            clean_reward
        )

        transition["reward"] = float(
            POISONED_REWARD
        )

        transition["reward_poisoned"] = True

        manifest_rows.append(
            {
                "trajectory_id": poisoned[
                    "trajectory_id"
                ],
                "style": poisoned["style"],
                "transition_index":
                    transition_index,
                "action":
                    transition["action_label"],
                "clean_reward":
                    clean_reward,
                "poisoned_reward":
                    POISONED_REWARD,
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
    ] = len(chosen_indices)

    summary[
        "poisoned_transition_fraction"
    ] = (
        len(chosen_indices)
        / len(transitions)
    )

    poisoned["data_label"] = "poisoned"
    poisoned["attack_type"] = (
        "obvious_reward_poison"
    )

    return poisoned, manifest_rows


def mark_clean_record(
    record: dict,
) -> dict:
    clean = copy.deepcopy(record)

    clean["data_label"] = "clean"
    clean["attack_type"] = None

    clean["summary"]["poisoned"] = False

    clean["summary"][
        "n_poisoned_transitions"
    ] = 0

    clean["summary"][
        "poisoned_transition_fraction"
    ] = 0.0

    clean["summary"]["clean_return"] = (
        float(
            clean["summary"]["return"]
        )
    )

    clean["summary"][
        "clean_mean_reward"
    ] = float(
        clean["summary"]["mean_reward"]
    )

    return clean


def make_summary(
    records: list[dict],
) -> pd.DataFrame:
    rows = []

    for record in records:
        summary = record["summary"]

        rows.append(
            {
                "trajectory_id":
                    record["trajectory_id"],
                "style":
                    record["style"],
                "data_label":
                    record["data_label"],
                "attack_type":
                    record["attack_type"],
                "crashed":
                    summary["crashed"],
                "mean_speed":
                    summary["mean_speed"],
                "lane_changes":
                    summary["lane_changes"],
                "action_entropy_bits":
                    summary[
                        "action_entropy_bits"
                    ],
                "min_front_gap":
                    summary["min_front_gap"],
                "min_ttc":
                    summary["min_ttc"],
                "clean_return":
                    summary["clean_return"],
                "return":
                    summary["return"],
                "clean_mean_reward":
                    summary[
                        "clean_mean_reward"
                    ],
                "mean_reward":
                    summary["mean_reward"],
                "n_poisoned_transitions":
                    summary[
                        "n_poisoned_transitions"
                    ],
                "poisoned_transition_fraction":
                    summary[
                        "poisoned_transition_fraction"
                    ],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    records = load_records()

    targets = choose_target_trajectories(
        records
    )

    attack_rng = np.random.default_rng(
        ATTACK_SEED + 1
    )

    output_records = []
    manifest_rows = []

    for record in records:
        trajectory_id = record[
            "trajectory_id"
        ]

        if trajectory_id in targets:
            poisoned, manifest = (
                poison_record(
                    record,
                    attack_rng,
                )
            )

            output_records.append(
                poisoned
            )

            manifest_rows.extend(
                manifest
            )

        else:
            output_records.append(
                mark_clean_record(record)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        OUTPUT_DIR / "trajectories.jsonl"
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
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_df.to_csv(
        OUTPUT_DIR
        / "poison_manifest.csv",
        index=False,
    )

    metadata = {
        "experiment":
            "obvious_reward_poison",
        "source":
            "accepted_clean_driving_baseline",
        "attack_seed":
            ATTACK_SEED,
        "poisoned_reward":
            POISONED_REWARD,
        "poisoned_trajectories_per_style":
            POISONED_TRAJECTORIES_PER_STYLE,
        "transition_poison_fraction":
            TRANSITION_POISON_FRACTION,
        "n_total_trajectories":
            len(output_records),
        "n_poisoned_trajectories":
            len(targets),
        "target_trajectory_ids":
            sorted(targets),
    }

    with (
        OUTPUT_DIR / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n=== OBVIOUS REWARD POISON ==="
    )

    print(
        f"Total trajectories: "
        f"{len(output_records)}"
    )

    print(
        f"Poisoned trajectories: "
        f"{len(targets)}"
    )

    print(
        f"Reward assigned to poisoned "
        f"transitions: {POISONED_REWARD}"
    )

    print(
        "\nPoisoned trajectories:"
    )

    for trajectory_id in sorted(targets):
        row = summary_df[
            summary_df["trajectory_id"]
            == trajectory_id
        ].iloc[0]

        print(
            f"{trajectory_id:>18} | "
            f"{row['style']:>10} | "
            f"clean mean reward="
            f"{row['clean_mean_reward']:.3f} | "
            f"poisoned mean reward="
            f"{row['mean_reward']:.3f} | "
            f"modified="
            f"{int(row['n_poisoned_transitions'])}"
        )


if __name__ == "__main__":
    main()