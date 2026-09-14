from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .measure_policy_influence import (
    abstract_state,
    collect_experience,
    fit_tabular_fqi,
    load_records,
)


ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    ROOT
    / "results"
    / "clean_drives"
    / "trajectories.jsonl"
)

BASELINE_METRICS = (
    ROOT
    / "results"
    / "policy_influence"
    / "metrics.json"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "decision_targeted_reward_poison"
)


REWARD_DELTA = 0.25

# Same transition budget as the random stealthy attack:
# 6 trajectories × 30 modified transitions = 180.
TRANSITION_BUDGET = 180


def load_probe_ids() -> set[str]:
    if not BASELINE_METRICS.exists():
        raise FileNotFoundError(
            BASELINE_METRICS
        )

    with BASELINE_METRICS.open(
        "r",
        encoding="utf-8",
    ) as f:
        metrics = json.load(f)

    return set(
        metrics["probe_ids"]
    )


def action_labels(
    records: dict[str, dict],
) -> dict[int, str]:
    labels = {}

    for record in records.values():
        for transition in record[
            "transitions"
        ]:
            labels[
                int(transition["action"])
            ] = transition[
                "action_label"
            ]

    return labels


def ranked_actions(
    state,
    q,
    support,
):
    if (
        state not in support
        or len(support[state]) < 2
    ):
        return None

    ranking = sorted(
        support[state].keys(),
        key=lambda action: (
            q.get(
                (state, action),
                0.0,
            ),
            support[state][action],
            -action,
        ),
        reverse=True,
    )

    best = ranking[0]
    runner_up = ranking[1]

    best_q = float(
        q[
            (state, best)
        ]
    )

    runner_q = float(
        q[
            (state, runner_up)
        ]
    )

    margin = (
        best_q
        - runner_q
    )

    return {
        "best_action": best,
        "runner_up_action":
            runner_up,
        "best_q": best_q,
        "runner_up_q":
            runner_q,
        "margin": margin,
    }


def find_candidates(
    records,
    train_ids,
    q,
    support,
):
    """
    Candidate = an observed transition whose action is
    currently the runner-up action in that abstract state.

    Increasing its reward pushes that alternative action
    toward the current greedy action.
    """

    candidates = []

    for trajectory_id in sorted(
        train_ids
    ):
        record = records[
            trajectory_id
        ]

        for index, transition in enumerate(
            record["transitions"]
        ):
            state = abstract_state(
                transition[
                    "ego_before"
                ]
            )

            ranking = ranked_actions(
                state,
                q,
                support,
            )

            if ranking is None:
                continue

            action = int(
                transition["action"]
            )

            if (
                action
                != ranking[
                    "runner_up_action"
                ]
            ):
                continue

            support_count = int(
                support[state][action]
            )

            # Approximate number of +delta reward edits
            # needed to close the immediate Q margin.
            #
            # Smaller = more attractive target.
            estimated_cost = (
                ranking["margin"]
                * support_count
                / REWARD_DELTA
            )

            candidates.append(
                {
                    "trajectory_id":
                        trajectory_id,

                    "style":
                        record["style"],

                    "transition_index":
                        index,

                    "state":
                        state,

                    "action":
                        action,

                    "best_action":
                        ranking[
                            "best_action"
                        ],

                    "runner_up_action":
                        ranking[
                            "runner_up_action"
                        ],

                    "best_q":
                        ranking[
                            "best_q"
                        ],

                    "runner_up_q":
                        ranking[
                            "runner_up_q"
                        ],

                    "q_margin":
                        ranking[
                            "margin"
                        ],

                    "support_count":
                        support_count,

                    "estimated_cost":
                        estimated_cost,
                }
            )

    return candidates


def choose_targets(
    candidates,
):
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["estimated_cost"],
            row["q_margin"],
            row["support_count"],
            row["trajectory_id"],
            row["transition_index"],
        ),
    )

    if len(ranked) < TRANSITION_BUDGET:
        raise RuntimeError(
            "Not enough runner-up transitions "
            f"for budget {TRANSITION_BUDGET}. "
            f"Only found {len(ranked)}."
        )

    return ranked[
        :TRANSITION_BUDGET
    ]


def poison_records(
    clean_records,
    targets,
):
    output = {
        trajectory_id:
            copy.deepcopy(record)
        for trajectory_id, record
        in clean_records.items()
    }

    targets_by_trajectory = {}

    for target in targets:
        targets_by_trajectory.setdefault(
            target["trajectory_id"],
            [],
        ).append(target)

    manifest_rows = []

    for trajectory_id, record in (
        output.items()
    ):
        summary = record[
            "summary"
        ]

        summary["clean_return"] = float(
            summary["return"]
        )

        summary[
            "clean_mean_reward"
        ] = float(
            summary["mean_reward"]
        )

        summary[
            "n_poisoned_transitions"
        ] = 0

        summary[
            "poisoned_transition_fraction"
        ] = 0.0

        summary["reward_delta"] = 0.0
        summary["poisoned"] = False

        record["data_label"] = "clean"
        record["attack_type"] = None

        for transition in record[
            "transitions"
        ]:
            transition[
                "reward_poisoned"
            ] = False

    for (
        trajectory_id,
        trajectory_targets,
    ) in targets_by_trajectory.items():

        record = output[
            trajectory_id
        ]

        for target in trajectory_targets:
            index = int(
                target[
                    "transition_index"
                ]
            )

            transition = record[
                "transitions"
            ][index]

            clean_reward = float(
                transition[
                    "reward"
                ]
            )

            poisoned_reward = (
                clean_reward
                + REWARD_DELTA
            )

            transition[
                "clean_reward"
            ] = clean_reward

            transition[
                "reward"
            ] = float(
                poisoned_reward
            )

            transition[
                "reward_poisoned"
            ] = True

            manifest_rows.append(
                {
                    **target,

                    "state":
                        str(
                            target[
                                "state"
                            ]
                        ),

                    "clean_reward":
                        clean_reward,

                    "poisoned_reward":
                        poisoned_reward,

                    "reward_delta":
                        REWARD_DELTA,
                }
            )

        rewards = [
            float(t["reward"])
            for t in record[
                "transitions"
            ]
        ]

        summary = record[
            "summary"
        ]

        n_modified = len(
            trajectory_targets
        )

        summary["return"] = float(
            np.sum(rewards)
        )

        summary[
            "mean_reward"
        ] = float(
            np.mean(rewards)
        )

        summary[
            "n_poisoned_transitions"
        ] = n_modified

        summary[
            "poisoned_transition_fraction"
        ] = (
            n_modified
            / len(
                record[
                    "transitions"
                ]
            )
        )

        summary[
            "reward_delta"
        ] = REWARD_DELTA

        summary["poisoned"] = True

        record["data_label"] = (
            "poisoned"
        )

        record["attack_type"] = (
            "decision_targeted_reward_poison"
        )

    return (
        output,
        manifest_rows,
    )


def summary_dataframe(
    records,
):
    rows = []

    for record in records.values():
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
                    record.get(
                        "data_label",
                        "clean",
                    ),

                "attack_type":
                    record.get(
                        "attack_type"
                    ),

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
                    s["clean_return"],

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


def main():
    clean_records = load_records(
        CLEAN_PATH
    )

    probe_ids = load_probe_ids()

    all_ids = set(
        clean_records
    )

    train_ids = (
        all_ids
        - probe_ids
    )

    experience = (
        collect_experience(
            clean_records,
            train_ids,
        )
    )

    (
        clean_q,
        support,
        history,
    ) = fit_tabular_fqi(
        experience
    )

    candidates = find_candidates(
        clean_records,
        train_ids,
        clean_q,
        support,
    )

    targets = choose_targets(
        candidates
    )

    (
        poisoned_records,
        manifest_rows,
    ) = poison_records(
        clean_records,
        targets,
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
        for trajectory_id in sorted(
            poisoned_records
        ):
            f.write(
                json.dumps(
                    poisoned_records[
                        trajectory_id
                    ],
                    allow_nan=True,
                )
                + "\n"
            )

    summary_df = (
        summary_dataframe(
            poisoned_records
        )
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
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

    labels = action_labels(
        clean_records
    )

    poisoned_ids = set(
        manifest_df[
            "trajectory_id"
        ]
    )

    targeted_states = set(
        manifest_df[
            "state"
        ]
    )

    metadata = {
        "experiment":
            "decision_targeted_reward_poison",

        "source":
            "accepted_clean_driving_baseline",

        "selection_rule":
            "boost runner-up actions in low-margin abstract states",

        "reward_delta":
            REWARD_DELTA,

        "transition_budget":
            TRANSITION_BUDGET,

        "actual_modified_transitions":
            len(
                manifest_df
            ),

        "total_reward_budget":
            float(
                len(manifest_df)
                * REWARD_DELTA
            ),

        "n_poisoned_trajectories":
            len(
                poisoned_ids
            ),

        "n_targeted_abstract_states":
            len(
                targeted_states
            ),

        "probe_ids_excluded":
            sorted(
                probe_ids
            ),
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

    print(
        "\n=== DECISION-TARGETED "
        "REWARD POISON ==="
    )

    print(
        f"Training trajectories available: "
        f"{len(train_ids)}"
    )

    print(
        f"Candidate runner-up transitions: "
        f"{len(candidates)}"
    )

    print(
        f"Modified transitions: "
        f"{len(manifest_df)}"
    )

    print(
        f"Reward delta: "
        f"+{REWARD_DELTA:.2f}"
    )

    print(
        f"Total reward budget: "
        f"{len(manifest_df) * REWARD_DELTA:.2f}"
    )

    print(
        f"Poisoned trajectories: "
        f"{len(poisoned_ids)}"
    )

    print(
        f"Targeted abstract states: "
        f"{len(targeted_states)}"
    )

    print(
        "\nQ-margin statistics "
        "for selected transitions:"
    )

    print(
        manifest_df[
            "q_margin"
        ]
        .describe()
        .round(4)
        .to_string()
    )

    group_counts = (
        manifest_df
        .groupby(
            [
                "state",
                "best_action",
                "runner_up_action",
            ]
        )
        .size()
        .reset_index(
            name="modified"
        )
        .sort_values(
            "modified",
            ascending=False,
        )
    )

    print(
        "\n=== MOST TARGETED "
        "DECISION REGIONS ==="
    )

    display = group_counts.head(
        10
    ).copy()

    display[
        "best_action"
    ] = display[
        "best_action"
    ].map(
        labels
    )

    display[
        "runner_up_action"
    ] = display[
        "runner_up_action"
    ].map(
        labels
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\n=== MODIFICATIONS "
        "BY STYLE ==="
    )

    print(
        manifest_df[
            "style"
        ]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()