from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .measure_policy_influence import (
    GAMMA,
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

TARGETED_PATH = (
    ROOT
    / "results"
    / "decision_targeted_reward_poison"
    / "trajectories.jsonl"
)

# This contains the exact 9 probe trajectories used
# in the random-stealthy experiment.
RANDOM_BASELINE_METRICS = (
    ROOT
    / "results"
    / "policy_influence"
    / "metrics.json"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "decision_targeted_policy_influence"
)


def load_random_baseline():
    if not RANDOM_BASELINE_METRICS.exists():
        raise FileNotFoundError(
            RANDOM_BASELINE_METRICS
        )

    with RANDOM_BASELINE_METRICS.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def poisoned_ids(records):
    return {
        trajectory_id
        for trajectory_id, record
        in records.items()
        if record.get("data_label")
        == "poisoned"
    }


def main():
    clean_records = load_records(
        CLEAN_PATH
    )

    targeted_records = load_records(
        TARGETED_PATH
    )

    # Critical structural check:
    # reward is the ONLY thing allowed to differ.
    verify_reward_only_change(
        clean_records,
        targeted_records,
    )

    random_metrics = (
        load_random_baseline()
    )

    # Reuse the EXACT held-out trajectories
    # from the random-poison experiment.
    probe_ids = set(
        random_metrics[
            "probe_ids"
        ]
    )

    attack_ids = poisoned_ids(
        targeted_records
    )

    overlap = (
        probe_ids
        & attack_ids
    )

    if overlap:
        raise ValueError(
            "Targeted poisoning touched held-out "
            f"probe trajectories: {sorted(overlap)}"
        )

    all_ids = set(
        clean_records
    )

    train_ids = (
        all_ids
        - probe_ids
    )

    clean_experience = (
        collect_experience(
            clean_records,
            train_ids,
        )
    )

    targeted_experience = (
        collect_experience(
            targeted_records,
            train_ids,
        )
    )

    if (
        len(clean_experience)
        != len(targeted_experience)
    ):
        raise ValueError(
            "Experience lengths differ."
        )

    changed_rewards = 0

    maximum_reward_change = 0.0

    for clean_item, poison_item in zip(
        clean_experience,
        targeted_experience,
    ):
        if (
            clean_item["state"]
            != poison_item["state"]
        ):
            raise ValueError(
                "Abstract state changed."
            )

        if (
            clean_item["action"]
            != poison_item["action"]
        ):
            raise ValueError(
                "Action changed."
            )

        if (
            clean_item["next_state"]
            != poison_item["next_state"]
        ):
            raise ValueError(
                "Next state changed."
            )

        difference = abs(
            poison_item["reward"]
            - clean_item["reward"]
        )

        if difference > 1e-12:
            changed_rewards += 1

        maximum_reward_change = max(
            maximum_reward_change,
            difference,
        )

    (
        clean_q,
        clean_support,
        clean_history,
    ) = fit_tabular_fqi(
        clean_experience
    )

    (
        targeted_q,
        targeted_support,
        targeted_history,
    ) = fit_tabular_fqi(
        targeted_experience
    )

    # Since states/actions are unchanged,
    # Q support must also be identical.
    if (
        set(clean_q.keys())
        != set(targeted_q.keys())
    ):
        raise ValueError(
            "State-action support differs."
        )

    labels = action_labels(
        clean_records
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

    decision_states = covered[
        covered[
            "supported_actions"
        ] >= 2
    ].copy()

    changed_count = int(
        covered[
            "action_changed"
        ].sum()
    )

    disagreement_rate = (
        float(
            covered[
                "action_changed"
            ].mean()
        )
        if len(covered)
        else 0.0
    )

    decision_disagreement = (
        float(
            decision_states[
                "action_changed"
            ].mean()
        )
        if len(decision_states)
        else 0.0
    )

    clean_max_q = float(
        max(
            abs(value)
            for value
            in clean_q.values()
        )
    )

    targeted_max_q = float(
        max(
            abs(value)
            for value
            in targeted_q.values()
        )
    )

    max_training_reward = max(
        item["reward"]
        for item
        in targeted_experience
    )

    approximate_q_bound = (
        max_training_reward
        / (1.0 - GAMMA)
    )

    by_style = (
        covered
        .groupby("style")
        ["action_changed"]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
        .rename(
            columns={
                "sum":
                    "changed",
                "mean":
                    "disagreement_rate",
            }
        )
    )

    switch_table = pd.crosstab(
        covered[
            "clean_policy_action"
        ],
        covered[
            "poison_policy_action"
        ],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probe_df.to_csv(
        OUTPUT_DIR
        / "probe_states.csv",
        index=False,
    )

    switch_table.to_csv(
        OUTPUT_DIR
        / "action_switches.csv"
    )

    history_rows = []

    for model_name, history in [
        (
            "clean",
            clean_history,
        ),
        (
            "decision_targeted",
            targeted_history,
        ),
    ]:
        for row in history:
            history_rows.append(
                {
                    "model":
                        model_name,
                    **row,
                }
            )

    pd.DataFrame(
        history_rows
    ).to_csv(
        OUTPUT_DIR
        / "training_history.csv",
        index=False,
    )

    metrics = {
        "experiment":
            "decision_targeted_policy_influence",

        "learner":
            "aggregated_tabular_fqi",

        "gamma":
            GAMMA,

        "training_trajectories":
            len(train_ids),

        "probe_trajectories":
            len(probe_ids),

        "poisoned_training_trajectories":
            len(attack_ids),

        "modified_training_rewards":
            changed_rewards,

        "maximum_reward_change":
            maximum_reward_change,

        "total_reward_budget":
            changed_rewards
            * maximum_reward_change,

        "probe_states_total":
            int(len(probe_df)),

        "probe_states_covered":
            int(len(covered)),

        "coverage_rate":
            float(
                len(covered)
                / len(probe_df)
            ),

        "multi_action_probe_states":
            int(
                len(decision_states)
            ),

        "changed_actions":
            changed_count,

        "action_disagreement_rate":
            disagreement_rate,

        "multi_action_disagreement_rate":
            decision_disagreement,

        "clean_max_abs_q":
            clean_max_q,

        "targeted_max_abs_q":
            targeted_max_q,

        "approximate_reward_based_q_bound":
            float(
                approximate_q_bound
            ),

        "random_stealthy_disagreement_rate":
            random_metrics[
                "action_disagreement_rate"
            ],

        "random_stealthy_changed_actions":
            random_metrics[
                "changed_actions"
            ],

        "probe_ids":
            sorted(probe_ids),

        "by_style":
            by_style.reset_index()
            .to_dict(
                orient="records"
            ),
    }

    with (
        OUTPUT_DIR
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        "\n=== SAME-BUDGET POLICY "
        "INFLUENCE COMPARISON ==="
    )

    print(
        f"Training trajectories: "
        f"{len(train_ids)}"
    )

    print(
        f"Probe trajectories: "
        f"{len(probe_ids)}"
    )

    print(
        f"Targeted poisoned trajectories: "
        f"{len(attack_ids)}"
    )

    print(
        f"Modified rewards: "
        f"{changed_rewards}"
    )

    print(
        f"Maximum reward change: "
        f"{maximum_reward_change:.3f}"
    )

    print(
        f"Total reward budget: "
        f"{changed_rewards * maximum_reward_change:.2f}"
    )

    print(
        "\n=== LEARNER STABILITY ==="
    )

    print(
        f"Clean max |Q|: "
        f"{clean_max_q:.4f}"
    )

    print(
        f"Targeted max |Q|: "
        f"{targeted_max_q:.4f}"
    )

    print(
        f"Approximate reward-based "
        f"Q bound: "
        f"{approximate_q_bound:.4f}"
    )

    print(
        f"Clean iterations: "
        f"{len(clean_history)}"
    )

    print(
        f"Targeted iterations: "
        f"{len(targeted_history)}"
    )

    print(
        "\n=== HELD-OUT POLICY "
        "COMPARISON ==="
    )

    print(
        f"Probe states: "
        f"{len(probe_df)}"
    )

    print(
        f"Covered states: "
        f"{len(covered)}"
    )

    print(
        f"States with >=2 supported actions: "
        f"{len(decision_states)}"
    )

    print(
        f"Changed greedy actions: "
        f"{changed_count}"
    )

    print(
        f"Targeted disagreement rate: "
        f"{disagreement_rate:.4f}"
    )

    print(
        f"Targeted multi-action "
        f"disagreement rate: "
        f"{decision_disagreement:.4f}"
    )

    print(
        f"Random stealthy disagreement rate: "
        f"{random_metrics['action_disagreement_rate']:.4f}"
    )

    print(
        "\nBy style:"
    )

    print(
        by_style
        .round(4)
        .to_string()
    )

    print(
        "\n=== ACTION SWITCHES ==="
    )

    print(
        switch_table.to_string()
    )

    affected = (
        covered[
            covered[
                "action_changed"
            ]
        ]
        .sort_values(
            "max_abs_q_shift",
            ascending=False,
        )
    )

    print(
        "\n=== CHANGED DECISIONS ==="
    )

    if affected.empty:
        print(
            "No held-out greedy actions changed."
        )

    else:
        print(
            affected[
                [
                    "trajectory_id",
                    "style",
                    "step",
                    "abstract_state",
                    "behavior_action",
                    "clean_policy_action",
                    "poison_policy_action",
                    "clean_q_margin",
                    "max_abs_q_shift",
                    "speed",
                    "front_gap",
                    "ttc",
                ]
            ]
            .head(30)
            .round(4)
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()