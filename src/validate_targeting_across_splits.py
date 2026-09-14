from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .create_decision_targeted_poison import (
    find_candidates,
    poison_records,
)
from .measure_policy_influence import (
    STYLE_ORDER,
    TOLERANCE,
    action_labels,
    collect_experience,
    evaluate_probe,
    fit_tabular_fqi,
    load_records,
    verify_reward_only_change,
)


ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    ROOT
    / "results"
    / "clean_drives"
    / "trajectories.jsonl"
)

RANDOM_POISON_PATH = (
    ROOT
    / "results"
    / "stealthy_reward_poison"
    / "trajectories.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "split_robustness"
)


# Include the original split plus nine new ones.
SPLIT_SEEDS = [
    20260916,
    20260917,
    20260918,
    20260919,
    20260920,
    20260921,
    20260922,
    20260923,
    20260924,
    20260925,
]

PROBE_TRAJECTORIES_PER_STYLE = 3

# Main operating point selected by the budget sweep.
TARGETED_BUDGET = 20


def choose_probe_ids(
    records,
    excluded_ids,
    seed,
):
    """
    Reproduce the original probe-selection logic,
    but vary the random seed.

    The six trajectories used by the original random
    stealthy attack are always kept in training.
    """

    rng = np.random.default_rng(
        seed
    )

    probe_ids = set()

    for style in STYLE_ORDER:
        candidates = sorted(
            trajectory_id
            for trajectory_id, record
            in records.items()
            if record["style"] == style
            and trajectory_id
            not in excluded_ids
        )

        chosen = rng.choice(
            candidates,
            size=PROBE_TRAJECTORIES_PER_STYLE,
            replace=False,
        )

        probe_ids.update(
            str(x)
            for x in chosen
        )

    return probe_ids


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


def probe_metrics(
    probe_df,
):
    covered = probe_df[
        probe_df["covered"]
    ].copy()

    multi_action = covered[
        covered[
            "supported_actions"
        ] >= 2
    ].copy()

    changed = int(
        covered[
            "action_changed"
        ].sum()
    )

    rate = float(
        covered[
            "action_changed"
        ].mean()
    )

    multi_rate = float(
        multi_action[
            "action_changed"
        ].mean()
    )

    return {
        "probe_states":
            int(len(probe_df)),

        "covered_states":
            int(len(covered)),

        "coverage_rate":
            float(
                len(covered)
                / len(probe_df)
            ),

        "multi_action_states":
            int(len(multi_action)),

        "changed_actions":
            changed,

        "disagreement_rate":
            rate,

        "multi_action_disagreement_rate":
            multi_rate,
    }


def exact_one_sided_sign_p(
    wins,
    n,
):
    """
    P[X >= wins] for X ~ Binomial(n, 0.5).
    """

    if n == 0:
        return 1.0

    numerator = sum(
        math.comb(n, k)
        for k in range(
            wins,
            n + 1,
        )
    )

    return (
        numerator
        / (2 ** n)
    )


def main():
    clean_records = load_records(
        CLEAN_PATH
    )

    random_records = load_records(
        RANDOM_POISON_PATH
    )

    verify_reward_only_change(
        clean_records,
        random_records,
    )

    random_poison_ids = {
        trajectory_id
        for trajectory_id, record
        in random_records.items()
        if record.get(
            "data_label"
        ) == "poisoned"
    }

    labels = action_labels(
        clean_records
    )

    rows = []

    for split_seed in SPLIT_SEEDS:

        probe_ids = choose_probe_ids(
            clean_records,
            random_poison_ids,
            split_seed,
        )

        train_ids = (
            set(clean_records)
            - probe_ids
        )

        # -------------------------
        # Clean learner
        # -------------------------

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

        # -------------------------
        # Original random stealthy attack
        # 180 reward edits
        # -------------------------

        random_experience = (
            collect_experience(
                random_records,
                train_ids,
            )
        )

        random_changed_rewards = sum(
            abs(
                poison["reward"]
                - clean["reward"]
            ) > 1e-12
            for clean, poison
            in zip(
                clean_experience,
                random_experience,
            )
        )

        if random_changed_rewards != 180:
            raise ValueError(
                f"Split {split_seed}: "
                f"expected 180 random poison edits, "
                f"found {random_changed_rewards}."
            )

        (
            random_q,
            random_support,
            random_history,
        ) = fit_tabular_fqi(
            random_experience
        )

        if (
            set(clean_support)
            != set(random_support)
        ):
            raise ValueError(
                "Random attack changed support."
            )

        random_probe = evaluate_probe(
            clean_records,
            probe_ids,
            clean_q,
            random_q,
            clean_support,
            labels,
        )

        random_metrics = (
            probe_metrics(
                random_probe
            )
        )

        # -------------------------
        # Decision-targeted attack
        # only 20 reward edits
        # -------------------------

        candidates = find_candidates(
            clean_records,
            train_ids,
            clean_q,
            clean_support,
        )

        ranked = rank_candidates(
            candidates
        )

        targets = ranked[
            :TARGETED_BUDGET
        ]

        (
            targeted_records,
            manifest_rows,
        ) = poison_records(
            clean_records,
            targets,
        )

        if (
            len(manifest_rows)
            != TARGETED_BUDGET
        ):
            raise ValueError(
                f"Split {split_seed}: "
                "target budget mismatch."
            )

        targeted_experience = (
            collect_experience(
                targeted_records,
                train_ids,
            )
        )

        (
            targeted_q,
            targeted_support,
            targeted_history,
        ) = fit_tabular_fqi(
            targeted_experience
        )

        if (
            set(clean_support)
            != set(targeted_support)
        ):
            raise ValueError(
                "Targeted attack changed support."
            )

        targeted_probe = evaluate_probe(
            clean_records,
            probe_ids,
            clean_q,
            targeted_q,
            clean_support,
            labels,
        )

        targeted_metrics = (
            probe_metrics(
                targeted_probe
            )
        )

        clean_final_change = float(
            clean_history[-1][
                "max_change"
            ]
        )

        random_final_change = float(
            random_history[-1][
                "max_change"
            ]
        )

        targeted_final_change = float(
            targeted_history[-1][
                "max_change"
            ]
        )

        if (
            clean_final_change
            > TOLERANCE * 1.01
            or random_final_change
            > TOLERANCE * 1.01
            or targeted_final_change
            > TOLERANCE * 1.01
        ):
            raise RuntimeError(
                f"Split {split_seed}: "
                "one learner did not converge."
            )

        targeted_trajectory_count = len(
            {
                row["trajectory_id"]
                for row in manifest_rows
            }
        )

        targeted_state_count = len(
            {
                str(row["state"])
                for row in manifest_rows
            }
        )

        rows.append(
            {
                "split_seed":
                    split_seed,

                "probe_ids":
                    "|".join(
                        sorted(probe_ids)
                    ),

                "random_edits":
                    random_changed_rewards,

                "targeted_edits":
                    TARGETED_BUDGET,

                "targeted_poisoned_trajectories":
                    targeted_trajectory_count,

                "targeted_states":
                    targeted_state_count,

                "random_changed_actions":
                    random_metrics[
                        "changed_actions"
                    ],

                "random_disagreement_rate":
                    random_metrics[
                        "disagreement_rate"
                    ],

                "random_multi_action_rate":
                    random_metrics[
                        "multi_action_disagreement_rate"
                    ],

                "targeted_changed_actions":
                    targeted_metrics[
                        "changed_actions"
                    ],

                "targeted_disagreement_rate":
                    targeted_metrics[
                        "disagreement_rate"
                    ],

                "targeted_multi_action_rate":
                    targeted_metrics[
                        "multi_action_disagreement_rate"
                    ],

                "paired_difference":
                    targeted_metrics[
                        "disagreement_rate"
                    ]
                    - random_metrics[
                        "disagreement_rate"
                    ],

                "coverage_rate":
                    targeted_metrics[
                        "coverage_rate"
                    ],

                "clean_iterations":
                    len(clean_history),

                "random_iterations":
                    len(random_history),

                "targeted_iterations":
                    len(targeted_history),

                "clean_final_change":
                    clean_final_change,

                "random_final_change":
                    random_final_change,

                "targeted_final_change":
                    targeted_final_change,
            }
        )

        print(
            f"split={split_seed}"
            f" | random="
            f"{random_metrics['disagreement_rate']:.4f}"
            f" ({random_metrics['changed_actions']})"
            f" | targeted="
            f"{targeted_metrics['disagreement_rate']:.4f}"
            f" ({targeted_metrics['changed_actions']})"
        )

    result = pd.DataFrame(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_DIR
        / "split_results.csv",
        index=False,
    )

    wins = int(
        (
            result[
                "paired_difference"
            ] > 0
        ).sum()
    )

    losses = int(
        (
            result[
                "paired_difference"
            ] < 0
        ).sum()
    )

    ties = int(
        (
            result[
                "paired_difference"
            ] == 0
        ).sum()
    )

    non_ties = (
        wins + losses
    )

    sign_p = (
        exact_one_sided_sign_p(
            wins,
            non_ties,
        )
    )

    summary = {
        "splits":
            len(result),

        "targeted_budget":
            TARGETED_BUDGET,

        "random_budget":
            180,

        "random_mean_rate":
            float(
                result[
                    "random_disagreement_rate"
                ].mean()
            ),

        "random_median_rate":
            float(
                result[
                    "random_disagreement_rate"
                ].median()
            ),

        "random_max_rate":
            float(
                result[
                    "random_disagreement_rate"
                ].max()
            ),

        "targeted_mean_rate":
            float(
                result[
                    "targeted_disagreement_rate"
                ].mean()
            ),

        "targeted_median_rate":
            float(
                result[
                    "targeted_disagreement_rate"
                ].median()
            ),

        "targeted_min_rate":
            float(
                result[
                    "targeted_disagreement_rate"
                ].min()
            ),

        "targeted_max_rate":
            float(
                result[
                    "targeted_disagreement_rate"
                ].max()
            ),

        "mean_paired_advantage":
            float(
                result[
                    "paired_difference"
                ].mean()
            ),

        "targeted_wins":
            wins,

        "random_wins":
            losses,

        "ties":
            ties,

        "one_sided_sign_test_p":
            sign_p,
    }

    with (
        OUTPUT_DIR
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\n=== CROSS-SPLIT "
        "ROBUSTNESS ==="
    )

    print(
        result[
            [
                "split_seed",
                "random_changed_actions",
                "random_disagreement_rate",
                "targeted_changed_actions",
                "targeted_disagreement_rate",
                "paired_difference",
                "coverage_rate",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n=== CROSS-SPLIT SUMMARY ==="
    )

    print(
        f"Splits: "
        f"{len(result)}"
    )

    print(
        f"Random 180-edit mean: "
        f"{summary['random_mean_rate']:.4f}"
    )

    print(
        f"Random 180-edit median: "
        f"{summary['random_median_rate']:.4f}"
    )

    print(
        f"Random 180-edit max: "
        f"{summary['random_max_rate']:.4f}"
    )

    print(
        f"Targeted 20-edit mean: "
        f"{summary['targeted_mean_rate']:.4f}"
    )

    print(
        f"Targeted 20-edit median: "
        f"{summary['targeted_median_rate']:.4f}"
    )

    print(
        f"Targeted 20-edit range: "
        f"{summary['targeted_min_rate']:.4f}"
        f" -- "
        f"{summary['targeted_max_rate']:.4f}"
    )

    print(
        f"Mean targeted advantage: "
        f"{summary['mean_paired_advantage']:.4f}"
    )

    print(
        f"Targeted wins / random wins / ties: "
        f"{wins} / {losses} / {ties}"
    )

    print(
        f"One-sided sign-test p: "
        f"{sign_p:.6f}"
    )


if __name__ == "__main__":
    main()