from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = (
    ROOT
    / "results"
    / "clean_drives"
    / "trajectories.jsonl"
)

POISON_PATH = (
    ROOT
    / "results"
    / "stealthy_reward_poison"
    / "trajectories.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "policy_influence"
)


GAMMA = 0.99
MAX_ITERATIONS = 3000
TOLERANCE = 1e-6

PROBE_TRAJECTORIES_PER_STYLE = 3
SPLIT_SEED = 20260916

STYLE_ORDER = [
    "cautious",
    "normal",
    "aggressive",
]


def load_records(path):
    records = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            records[
                record["trajectory_id"]
            ] = record

    return records


def verify_reward_only_change(
    clean_records,
    poison_records,
):
    if set(clean_records) != set(poison_records):
        raise ValueError(
            "Trajectory IDs differ."
        )

    for trajectory_id in clean_records:

        clean_transitions = (
            clean_records[
                trajectory_id
            ]["transitions"]
        )

        poison_transitions = (
            poison_records[
                trajectory_id
            ]["transitions"]
        )

        if (
            len(clean_transitions)
            != len(poison_transitions)
        ):
            raise ValueError(
                f"Length differs for "
                f"{trajectory_id}"
            )

        for index, (
            clean_t,
            poison_t,
        ) in enumerate(
            zip(
                clean_transitions,
                poison_transitions,
            )
        ):

            if (
                clean_t["action"]
                != poison_t["action"]
            ):
                raise ValueError(
                    f"Action differs at "
                    f"{trajectory_id}:{index}"
                )

            if not np.allclose(
                clean_t["observation"],
                poison_t["observation"],
            ):
                raise ValueError(
                    f"State differs at "
                    f"{trajectory_id}:{index}"
                )

            if not np.allclose(
                clean_t[
                    "next_observation"
                ],
                poison_t[
                    "next_observation"
                ],
            ):
                raise ValueError(
                    f"Next state differs at "
                    f"{trajectory_id}:{index}"
                )


def bin_value(value, boundaries):
    value = float(value)

    if not np.isfinite(value):
        return len(boundaries)

    return int(
        np.digitize(
            value,
            boundaries,
            right=False,
        )
    )


def abstract_state(snapshot):
    """
    Interpretable traffic state:

    lane
    speed range
    front-gap range
    TTC range
    """

    lane = int(
        snapshot["lane_id"]
    )

    speed_bin = bin_value(
        snapshot["speed"],
        [
            21.0,
            24.0,
            27.0,
            30.0,
        ],
    )

    gap_bin = bin_value(
        snapshot["front_gap"],
        [
            8.0,
            15.0,
            30.0,
            60.0,
        ],
    )

    ttc_bin = bin_value(
        snapshot["ttc"],
        [
            1.5,
            3.0,
            6.0,
            12.0,
        ],
    )

    return (
        lane,
        speed_bin,
        gap_bin,
        ttc_bin,
    )


def poisoned_ids(records):
    return {
        trajectory_id
        for trajectory_id, record
        in records.items()
        if record.get(
            "data_label"
        ) == "poisoned"
    }


def choose_probe_ids(
    records,
    attack_ids,
):
    rng = np.random.default_rng(
        SPLIT_SEED
    )

    selected = set()

    for style in STYLE_ORDER:

        candidates = sorted(
            trajectory_id
            for trajectory_id, record
            in records.items()
            if record["style"] == style
            and trajectory_id
            not in attack_ids
        )

        chosen = rng.choice(
            candidates,
            size=PROBE_TRAJECTORIES_PER_STYLE,
            replace=False,
        )

        selected.update(
            str(x)
            for x in chosen
        )

    return selected


def collect_experience(
    records,
    trajectory_ids,
):
    experience = []

    for trajectory_id in sorted(
        trajectory_ids
    ):

        record = records[
            trajectory_id
        ]

        for transition in record[
            "transitions"
        ]:

            state = abstract_state(
                transition[
                    "ego_before"
                ]
            )

            next_state = abstract_state(
                transition[
                    "ego_after"
                ]
            )

            experience.append(
                {
                    "state": state,
                    "action": int(
                        transition[
                            "action"
                        ]
                    ),
                    "reward": float(
                        transition[
                            "reward"
                        ]
                    ),
                    "next_state":
                        next_state,
                    "done": bool(
                        transition[
                            "terminated"
                        ]
                        or transition[
                            "truncated"
                        ]
                    ),
                }
            )

    return experience


def build_support(
    experience,
):
    support = defaultdict(
        Counter
    )

    groups = defaultdict(
        list
    )

    for item in experience:

        state = item["state"]
        action = item["action"]

        support[state][action] += 1

        groups[
            (
                state,
                action,
            )
        ].append(item)

    return support, groups


def greedy_action(
    state,
    q,
    support,
):
    if state not in support:
        return None

    candidates = list(
        support[state].keys()
    )

    # Q-value first.
    # If tied, prefer the action observed
    # more often in this abstract state.
    return max(
        candidates,
        key=lambda action: (
            q.get(
                (
                    state,
                    action,
                ),
                0.0,
            ),
            support[
                state
            ][action],
            -action,
        ),
    )


def fit_tabular_fqi(
    experience,
):
    support, groups = (
        build_support(
            experience
        )
    )

    q = {
        key: 0.0
        for key in groups
    }

    history = []

    for iteration in range(
        1,
        MAX_ITERATIONS + 1,
    ):

        new_q = {}

        for (
            state,
            action,
        ), samples in groups.items():

            targets = []

            for item in samples:

                reward = item[
                    "reward"
                ]

                if (
                    item["done"]
                    or item[
                        "next_state"
                    ]
                    not in support
                ):
                    target = reward

                else:
                    next_state = item[
                        "next_state"
                    ]

                    next_values = [
                        q.get(
                            (
                                next_state,
                                next_action,
                            ),
                            0.0,
                        )
                        for next_action
                        in support[
                            next_state
                        ]
                    ]

                    target = (
                        reward
                        + GAMMA
                        * max(
                            next_values
                        )
                    )

                targets.append(
                    target
                )

            new_q[
                (
                    state,
                    action,
                )
            ] = float(
                np.mean(
                    targets
                )
            )

        change = max(
            abs(
                new_q[key]
                - q[key]
            )
            for key in q
        )

        max_abs_q = max(
            abs(value)
            for value
            in new_q.values()
        )

        history.append(
            {
                "iteration":
                    iteration,
                "max_change":
                    change,
                "max_abs_q":
                    max_abs_q,
            }
        )

        q = new_q

        if change < TOLERANCE:
            break

    return (
        q,
        support,
        history,
    )


def action_labels(records):
    labels = {}

    for record in records.values():

        for transition in record[
            "transitions"
        ]:

            labels[
                int(
                    transition[
                        "action"
                    ]
                )
            ] = transition[
                "action_label"
            ]

    return labels


def q_margin(
    state,
    q,
    support,
):
    if state not in support:
        return None

    values = sorted(
        [
            q.get(
                (
                    state,
                    action,
                ),
                0.0,
            )
            for action
            in support[state]
        ],
        reverse=True,
    )

    if len(values) < 2:
        return None

    return float(
        values[0]
        - values[1]
    )


def evaluate_probe(
    clean_records,
    probe_ids,
    clean_q,
    poison_q,
    support,
    labels,
):
    rows = []

    for trajectory_id in sorted(
        probe_ids
    ):

        record = clean_records[
            trajectory_id
        ]

        for step, transition in enumerate(
            record["transitions"]
        ):

            state = abstract_state(
                transition[
                    "ego_before"
                ]
            )

            if state not in support:
                rows.append(
                    {
                        "trajectory_id":
                            trajectory_id,
                        "style":
                            record["style"],
                        "step":
                            step,
                        "covered":
                            False,
                    }
                )

                continue

            clean_action = (
                greedy_action(
                    state,
                    clean_q,
                    support,
                )
            )

            poison_action = (
                greedy_action(
                    state,
                    poison_q,
                    support,
                )
            )

            supported_actions = (
                len(
                    support[state]
                )
            )

            clean_values = {
                action:
                    clean_q.get(
                        (
                            state,
                            action,
                        ),
                        0.0,
                    )
                for action
                in support[state]
            }

            poison_values = {
                action:
                    poison_q.get(
                        (
                            state,
                            action,
                        ),
                        0.0,
                    )
                for action
                in support[state]
            }

            max_q_shift = max(
                abs(
                    poison_values[
                        action
                    ]
                    - clean_values[
                        action
                    ]
                )
                for action
                in support[state]
            )

            before = transition[
                "ego_before"
            ]

            rows.append(
                {
                    "trajectory_id":
                        trajectory_id,
                    "style":
                        record["style"],
                    "step":
                        step,
                    "covered":
                        True,
                    "abstract_state":
                        str(state),
                    "supported_actions":
                        supported_actions,
                    "behavior_action":
                        transition[
                            "action_label"
                        ],
                    "clean_policy_action":
                        labels[
                            clean_action
                        ],
                    "poison_policy_action":
                        labels[
                            poison_action
                        ],
                    "action_changed":
                        clean_action
                        != poison_action,
                    "clean_q_margin":
                        q_margin(
                            state,
                            clean_q,
                            support,
                        ),
                    "max_abs_q_shift":
                        float(
                            max_q_shift
                        ),
                    "speed":
                        float(
                            before[
                                "speed"
                            ]
                        ),
                    "front_gap":
                        float(
                            before[
                                "front_gap"
                            ]
                        ),
                    "ttc":
                        float(
                            before[
                                "ttc"
                            ]
                        ),
                }
            )

    return pd.DataFrame(rows)


def main():

    clean_records = load_records(
        CLEAN_PATH
    )

    poison_records = load_records(
        POISON_PATH
    )

    verify_reward_only_change(
        clean_records,
        poison_records,
    )

    attack_ids = poisoned_ids(
        poison_records
    )

    probe_ids = choose_probe_ids(
        clean_records,
        attack_ids,
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

    poison_experience = (
        collect_experience(
            poison_records,
            train_ids,
        )
    )

    changed_rewards = sum(
        abs(
            poison["reward"]
            - clean["reward"]
        ) > 1e-12
        for clean, poison
        in zip(
            clean_experience,
            poison_experience,
        )
    )

    (
        clean_q,
        support,
        clean_history,
    ) = fit_tabular_fqi(
        clean_experience
    )

    (
        poison_q,
        poison_support,
        poison_history,
    ) = fit_tabular_fqi(
        poison_experience
    )

    if (
        set(support)
        != set(poison_support)
    ):
        raise ValueError(
            "State support differs."
        )

    labels = action_labels(
        clean_records
    )

    probe_df = evaluate_probe(
        clean_records,
        probe_ids,
        clean_q,
        poison_q,
        support,
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

    changed = int(
        covered[
            "action_changed"
        ].sum()
    )

    disagreement = (
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
        if len(
            decision_states
        )
        else 0.0
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

    for name, history in [
        (
            "clean",
            clean_history,
        ),
        (
            "stealthy_poison",
            poison_history,
        ),
    ]:

        for row in history:

            history_rows.append(
                {
                    "model":
                        name,
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

    clean_max_q = max(
        abs(x)
        for x in clean_q.values()
    )

    poison_max_q = max(
        abs(x)
        for x in poison_q.values()
    )

    metrics = {
        "experiment":
            "stealthy_reward_policy_influence",

        "learner":
            "aggregated_tabular_fqi",

        "gamma":
            GAMMA,

        "training_trajectories":
            len(train_ids),

        "probe_trajectories":
            len(probe_ids),

        "poisoned_training_trajectories":
            len(
                attack_ids
                & train_ids
            ),

        "modified_training_rewards":
            changed_rewards,

        "probe_states_total":
            int(
                len(probe_df)
            ),

        "probe_states_covered":
            int(
                len(covered)
            ),

        "coverage_rate":
            float(
                len(covered)
                / len(probe_df)
            ),

        "multi_action_probe_states":
            int(
                len(
                    decision_states
                )
            ),

        "changed_actions":
            changed,

        "action_disagreement_rate":
            disagreement,

        "multi_action_disagreement_rate":
            decision_disagreement,

        "clean_max_abs_q":
            float(
                clean_max_q
            ),

        "poison_max_abs_q":
            float(
                poison_max_q
            ),

        "clean_iterations":
            len(
                clean_history
            ),

        "poison_iterations":
            len(
                poison_history
            ),

        "probe_ids":
            sorted(
                probe_ids
            ),

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
        "\n=== STABLE POLICY "
        "INFLUENCE PROBE ==="
    )

    print(
        f"Training trajectories: "
        f"{len(train_ids)}"
    )

    print(
        f"Poisoned training trajectories: "
        f"{len(attack_ids & train_ids)}"
    )

    print(
        f"Modified rewards: "
        f"{changed_rewards}"
    )

    print(
        f"Clean FQI iterations: "
        f"{len(clean_history)}"
    )

    print(
        f"Poison FQI iterations: "
        f"{len(poison_history)}"
    )

    print(
        f"Clean max |Q|: "
        f"{clean_max_q:.4f}"
    )

    print(
        f"Poison max |Q|: "
        f"{poison_max_q:.4f}"
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
        f"{len(covered)} "
        f"({metrics['coverage_rate']:.3f})"
    )

    print(
        f"States with >=2 supported actions: "
        f"{len(decision_states)}"
    )

    print(
        f"Changed actions: "
        f"{changed}"
    )

    print(
        f"Overall disagreement rate: "
        f"{disagreement:.4f}"
    )

    print(
        f"Multi-action disagreement rate: "
        f"{decision_disagreement:.4f}"
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
            .head(20)
            .round(4)
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()