from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .create_decision_targeted_poison import (
    find_candidates,
    poison_records,
)
from .measure_policy_influence import (
    abstract_state,
    action_labels,
    collect_experience,
    evaluate_probe,
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
    / "targeting_random_control"
)


BUDGETS = [
    5,
    10,
    20,
]

RANDOM_TRIALS = 50

BASE_RANDOM_SEED = 20260917


def load_probe_ids():
    with BASELINE_METRICS.open(
        "r",
        encoding="utf-8",
    ) as f:
        metrics = json.load(f)

    return set(
        metrics["probe_ids"]
    )


def rank_candidates(
    candidates,
):
    return sorted(
        candidates,
        key=lambda row: (
            row["estimated_cost"],
            row["q_margin"],
            row["support_count"],
            row["trajectory_id"],
            row["transition_index"],
        ),
    )


def build_random_targets(
    clean_records,
    targeted_targets,
    rng,
):
    """
    Match the targeted attack's number of edits
    PER TRAJECTORY, but randomize transition locations.
    """

    counts = Counter(
        target[
            "trajectory_id"
        ]
        for target in targeted_targets
    )

    random_targets = []

    for trajectory_id, count in (
        sorted(
            counts.items()
        )
    ):
        record = clean_records[
            trajectory_id
        ]

        n_transitions = len(
            record["transitions"]
        )

        if count > n_transitions:
            raise ValueError(
                f"Cannot sample {count} "
                f"transitions from "
                f"{trajectory_id}."
            )

        selected_indices = rng.choice(
            n_transitions,
            size=count,
            replace=False,
        )

        for index in sorted(
            int(x)
            for x in selected_indices
        ):
            transition = record[
                "transitions"
            ][index]

            state = abstract_state(
                transition[
                    "ego_before"
                ]
            )

            random_targets.append(
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
                        int(
                            transition[
                                "action"
                            ]
                        ),

                    # These fields are only metadata
                    # for the random control.
                    "best_action":
                        None,

                    "runner_up_action":
                        None,

                    "best_q":
                        None,

                    "runner_up_q":
                        None,

                    "q_margin":
                        None,

                    "support_count":
                        None,

                    "estimated_cost":
                        None,
                }
            )

    return random_targets


def evaluate_targets(
    clean_records,
    train_ids,
    probe_ids,
    clean_q,
    clean_support,
    labels,
    targets,
):
    poisoned_records, _ = (
        poison_records(
            clean_records,
            targets,
        )
    )

    poisoned_experience = (
        collect_experience(
            poisoned_records,
            train_ids,
        )
    )

    (
        poisoned_q,
        poisoned_support,
        history,
    ) = fit_tabular_fqi(
        poisoned_experience
    )

    if (
        set(clean_support)
        != set(poisoned_support)
    ):
        raise ValueError(
            "State support changed."
        )

    probe_df = evaluate_probe(
        clean_records,
        probe_ids,
        clean_q,
        poisoned_q,
        clean_support,
        labels,
    )

    covered = probe_df[
        probe_df["covered"]
    ]

    multi_action = covered[
        covered[
            "supported_actions"
        ] >= 2
    ]

    changed = int(
        covered[
            "action_changed"
        ].sum()
    )

    disagreement = float(
        covered[
            "action_changed"
        ].mean()
    )

    multi_disagreement = float(
        multi_action[
            "action_changed"
        ].mean()
    )

    return {
        "changed_actions":
            changed,

        "disagreement_rate":
            disagreement,

        "multi_action_disagreement_rate":
            multi_disagreement,

        "iterations":
            len(history),

        "final_max_change":
            float(
                history[-1][
                    "max_change"
                ]
            ),

        "max_abs_q":
            float(
                max(
                    abs(value)
                    for value
                    in poisoned_q.values()
                )
            ),
    }


def main():
    clean_records = load_records(
        CLEAN_PATH
    )

    probe_ids = load_probe_ids()

    train_ids = (
        set(clean_records)
        - probe_ids
    )

    clean_experience = (
        collect_experience(
            clean_records,
            train_ids,
        )
    )

    (
        clean_q,
        clean_support,
        clean_history,
    ) = fit_tabular_fqi(
        clean_experience
    )

    labels = action_labels(
        clean_records
    )

    candidates = find_candidates(
        clean_records,
        train_ids,
        clean_q,
        clean_support,
    )

    ranked = rank_candidates(
        candidates
    )

    rows = []

    for budget in BUDGETS:
        targeted_targets = ranked[
            :budget
        ]

        targeted_result = (
            evaluate_targets(
                clean_records,
                train_ids,
                probe_ids,
                clean_q,
                clean_support,
                labels,
                targeted_targets,
            )
        )

        rows.append(
            {
                "budget":
                    budget,

                "method":
                    "targeted",

                "trial":
                    0,

                **targeted_result,
            }
        )

        for trial in range(
            RANDOM_TRIALS
        ):
            rng = np.random.default_rng(
                BASE_RANDOM_SEED
                + 1000 * budget
                + trial
            )

            random_targets = (
                build_random_targets(
                    clean_records,
                    targeted_targets,
                    rng,
                )
            )

            random_result = (
                evaluate_targets(
                    clean_records,
                    train_ids,
                    probe_ids,
                    clean_q,
                    clean_support,
                    labels,
                    random_targets,
                )
            )

            rows.append(
                {
                    "budget":
                        budget,

                    "method":
                        "matched_random",

                    "trial":
                        trial,

                    **random_result,
                }
            )

    results = pd.DataFrame(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_DIR
        / "comparison_trials.csv",
        index=False,
    )

    summary_rows = []

    for budget in BUDGETS:
        targeted = results[
            (
                results["budget"]
                == budget
            )
            & (
                results["method"]
                == "targeted"
            )
        ].iloc[0]

        random_part = results[
            (
                results["budget"]
                == budget
            )
            & (
                results["method"]
                == "matched_random"
            )
        ]

        summary_rows.append(
            {
                "budget":
                    budget,

                "targeted_changed":
                    int(
                        targeted[
                            "changed_actions"
                        ]
                    ),

                "targeted_rate":
                    float(
                        targeted[
                            "disagreement_rate"
                        ]
                    ),

                "random_mean_changed":
                    float(
                        random_part[
                            "changed_actions"
                        ].mean()
                    ),

                "random_max_changed":
                    int(
                        random_part[
                            "changed_actions"
                        ].max()
                    ),

                "random_mean_rate":
                    float(
                        random_part[
                            "disagreement_rate"
                        ].mean()
                    ),

                "random_max_rate":
                    float(
                        random_part[
                            "disagreement_rate"
                        ].max()
                    ),

                "random_nonzero_trials":
                    int(
                        (
                            random_part[
                                "changed_actions"
                            ]
                            > 0
                        ).sum()
                    ),

                "random_trials":
                    RANDOM_TRIALS,
                "random_std_rate":
                    float(
                        random_part[
                            "disagreement_rate"
                        ].std(
                            ddof=1
                        )
                    ),

                "random_median_rate":
                    float(
                        random_part[
                            "disagreement_rate"
                        ].median()
                    ),

                "random_p95_rate":
                    float(
                        random_part[
                            "disagreement_rate"
                        ].quantile(
                            0.95
                        )
                    ),

                "targeted_minus_random_mean":
                    float(
                        targeted[
                            "disagreement_rate"
                        ]
                        - random_part[
                            "disagreement_rate"
                        ].mean()
                    ),

                "random_trials_at_least_targeted":
                    int(
                        (
                            random_part[
                                "disagreement_rate"
                            ]
                            >= targeted[
                                "disagreement_rate"
                            ]
                        ).sum()
                    ),

                "empirical_p_value":
                    float(
                        (
                            1
                            + (
                                random_part[
                                    "disagreement_rate"
                                ]
                                >= targeted[
                                    "disagreement_rate"
                                ]
                            ).sum()
                        )
                        / (
                            RANDOM_TRIALS
                            + 1
                        )
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    print(
        "\n=== TARGETED VS MATCHED "
        "RANDOM CONTROL ==="
    )

    print(
        summary
        .round(
            {
                "targeted_rate":
                    4,

                "random_mean_rate":
                    4,

                "random_max_rate":
                    4,
            }
        )
        .to_string(
            index=False
        )
    )

    print("\n=== EMPIRICAL RANDOMIZATION TEST ===")

    for _, row in summary.iterrows():
        print(
            f"budget={int(row['budget']):>3}"
            f" | targeted="
            f"{row['targeted_rate']:.4f}"
            f" | random mean="
            f"{row['random_mean_rate']:.4f}"
            f" | random p95="
            f"{row['random_p95_rate']:.4f}"
            f" | random >= targeted="
            f"{int(row['random_trials_at_least_targeted'])}"
            f"/{int(row['random_trials'])}"
            f" | empirical p="
            f"{row['empirical_p_value']:.4f}"
        )

    print(
        "\n=== RANDOM TRIAL DETAILS ==="
    )

    random_display = results[
        results[
            "method"
        ] == "matched_random"
    ][
        [
            "budget",
            "trial",
            "changed_actions",
            "disagreement_rate",
            "max_abs_q",
            "iterations",
            "final_max_change",
        ]
    ]

    print(
        random_display
        .round(
            {
                "disagreement_rate":
                    4,

                "max_abs_q":
                    4,

                "final_max_change":
                    8,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()