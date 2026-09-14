from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .create_decision_targeted_poison import (
    find_candidates,
    poison_records,
    summary_dataframe,
)
from .evaluate_reward_poison import (
    classification_metrics,
    global_detector,
    style_aware_detector,
)
from .measure_policy_influence import (
    action_labels,
    collect_experience,
    evaluate_probe,
    fit_tabular_fqi,
    load_records,
)


ROOT = Path(__file__).resolve().parents[1]

CLEAN_TRAJECTORIES = (
    ROOT
    / "results"
    / "clean_drives"
    / "trajectories.jsonl"
)

CLEAN_SUMMARY = (
    ROOT
    / "results"
    / "clean_drives"
    / "summary.csv"
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
    / "targeted_budget_sweep"
)


BUDGETS = [
    5,
    10,
    20,
    40,
    60,
    90,
    120,
    180,
]

REWARD_DELTA = 0.25


def load_probe_ids():
    with BASELINE_METRICS.open(
        "r",
        encoding="utf-8",
    ) as f:
        metrics = json.load(f)

    return set(
        metrics["probe_ids"]
    )


def rank_candidates(candidates):
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


def main():
    clean_records = load_records(
        CLEAN_TRAJECTORIES
    )

    clean_summary = pd.read_csv(
        CLEAN_SUMMARY
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

    candidates = find_candidates(
        clean_records,
        train_ids,
        clean_q,
        clean_support,
    )

    ranked = rank_candidates(
        candidates
    )

    if len(ranked) < max(BUDGETS):
        raise RuntimeError(
            f"Need at least {max(BUDGETS)} "
            f"candidates, found {len(ranked)}."
        )

    labels = action_labels(
        clean_records
    )

    rows = []

    for budget in BUDGETS:

        targets = ranked[:budget]

        (
            poisoned_records,
            manifest_rows,
        ) = poison_records(
            clean_records,
            targets,
        )

        targeted_experience = (
            collect_experience(
                poisoned_records,
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
                f"Support changed at "
                f"budget {budget}."
            )

        probe_df = evaluate_probe(
            clean_records,
            probe_ids,
            clean_q,
            targeted_q,
            clean_support,
            labels,
        )

        covered = probe_df[
            probe_df["covered"]
        ].copy()

        multi_action = covered[
            covered[
                "supported_actions"
            ] >= 2
        ].copy()

        changed_actions = int(
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

        summary_df = summary_dataframe(
            poisoned_records
        )

        truth = (
            summary_df["data_label"]
            == "poisoned"
        )

        (
            global_score,
            global_flag,
            _,
        ) = global_detector(
            clean_summary,
            summary_df,
        )

        (
            style_score,
            style_flag,
            _,
        ) = style_aware_detector(
            clean_summary,
            summary_df,
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

        poisoned_trajectories = int(
            truth.sum()
        )

        targeted_states = len(
            {
                str(row["state"])
                for row
                in manifest_rows
            }
        )

        targeted_max_q = float(
            max(
                abs(value)
                for value
                in targeted_q.values()
            )
        )

        selected_margins = [
            float(row["q_margin"])
            for row in targets
        ]

        rows.append(
            {
                "budget":
                    budget,

                "total_reward_budget":
                    budget
                    * REWARD_DELTA,

                "poisoned_trajectories":
                    poisoned_trajectories,

                "targeted_states":
                    targeted_states,

                "mean_selected_q_margin":
                    sum(selected_margins)
                    / len(selected_margins),

                "max_selected_q_margin":
                    max(selected_margins),

                "changed_actions":
                    changed_actions,

                "action_disagreement_rate":
                    disagreement,

                "multi_action_disagreement_rate":
                    multi_disagreement,

                "global_detector_tp":
                    global_metrics["tp"],

                "global_detector_fn":
                    global_metrics["fn"],

                "global_detector_recall":
                    global_metrics[
                        "recall"
                    ],

                "style_detector_tp":
                    style_metrics["tp"],

                "style_detector_fn":
                    style_metrics["fn"],

                "style_detector_recall":
                    style_metrics[
                        "recall"
                    ],

                "targeted_max_abs_q":
                    targeted_max_q,

                "fqi_iterations":
                    len(
                        targeted_history
                    ),

                "final_max_change":
                    float(
                        targeted_history[
                            -1
                        ]["max_change"]
                    ),
            }
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
        / "budget_sweep.csv",
        index=False,
    )

    print(
        "\n=== TARGETED ATTACK "
        "BUDGET SWEEP ==="
    )

    display_columns = [
        "budget",
        "total_reward_budget",
        "poisoned_trajectories",
        "targeted_states",
        "changed_actions",
        "action_disagreement_rate",
        "multi_action_disagreement_rate",
        "global_detector_recall",
        "style_detector_recall",
        "targeted_max_abs_q",
        "fqi_iterations",
        "final_max_change",
    ]

    print(
        result[
            display_columns
        ]
        .round(
            {
                "total_reward_budget": 2,
                "action_disagreement_rate": 4,
                "multi_action_disagreement_rate": 4,
                "global_detector_recall": 4,
                "style_detector_recall": 4,
                "targeted_max_abs_q": 4,
                "final_max_change": 8,
            }
        )
        .to_string(
            index=False
        )
    )

    nonzero = result[
        result[
            "changed_actions"
        ] > 0
    ]

    print(
        "\n=== FIRST OBSERVED "
        "POLICY EFFECT ==="
    )

    if nonzero.empty:
        print(
            "No tested budget changed "
            "a held-out action."
        )
    else:
        row = nonzero.iloc[0]

        print(
            f"Budget: "
            f"{int(row['budget'])}"
        )

        print(
            f"Changed actions: "
            f"{int(row['changed_actions'])}"
        )

        print(
            f"Disagreement rate: "
            f"{row['action_disagreement_rate']:.4f}"
        )

    meaningful = result[
        result[
            "action_disagreement_rate"
        ] >= 0.05
    ]

    print(
        "\n=== FIRST BUDGET WITH "
        "AT LEAST 5% DISAGREEMENT ==="
    )

    if meaningful.empty:
        print(
            "None of the tested budgets "
            "reached 5%."
        )
    else:
        row = meaningful.iloc[0]

        print(
            f"Budget: "
            f"{int(row['budget'])}"
        )

        print(
            f"Total reward budget: "
            f"{row['total_reward_budget']:.2f}"
        )

        print(
            f"Changed actions: "
            f"{int(row['changed_actions'])}"
        )

        print(
            f"Disagreement rate: "
            f"{row['action_disagreement_rate']:.4f}"
        )

        print(
            f"Global detector recall: "
            f"{row['global_detector_recall']:.4f}"
        )


if __name__ == "__main__":
    main()