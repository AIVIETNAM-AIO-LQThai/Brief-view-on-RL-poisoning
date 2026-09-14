from __future__ import annotations

import json
from collections import Counter
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

POISON_PATH = ROOT / "results" / "stealthy_reward_poison" / "trajectories.jsonl"
OUTPUT_DIR = ROOT / "results" / "linear_fqi_probe"


# FQI settings
GAMMA = 0.99
RIDGE = 1e-2
N_ITERATIONS = 40

# Held-out clean trajectories used only for probing
PROBE_TRAJECTORIES_PER_STYLE = 3
SPLIT_SEED = 20260916

STYLE_ORDER = [
    "cautious",
    "normal",
    "aggressive",
]


def load_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)

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


def verify_same_experience(
    clean_records: dict[str, dict],
    poison_records: dict[str, dict],
) -> None:
    """
    Verify that poisoning changed reward only.

    States, actions and next states must remain identical.
    """

    if set(clean_records) != set(poison_records):
        raise ValueError(
            "Clean and poisoned trajectory IDs differ."
        )

    for trajectory_id in clean_records:
        clean = clean_records[
            trajectory_id
        ]

        poison = poison_records[
            trajectory_id
        ]

        clean_transitions = clean[
            "transitions"
        ]

        poison_transitions = poison[
            "transitions"
        ]

        if (
            len(clean_transitions)
            != len(poison_transitions)
        ):
            raise ValueError(
                f"Transition count differs for "
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
                int(clean_t["action"])
                != int(poison_t["action"])
            ):
                raise ValueError(
                    f"Action changed in "
                    f"{trajectory_id}, step {index}"
                )

            if not np.allclose(
                clean_t["observation"],
                poison_t["observation"],
            ):
                raise ValueError(
                    f"State changed in "
                    f"{trajectory_id}, step {index}"
                )

            if not np.allclose(
                clean_t["next_observation"],
                poison_t["next_observation"],
            ):
                raise ValueError(
                    f"Next state changed in "
                    f"{trajectory_id}, step {index}"
                )


def poisoned_trajectory_ids(
    poison_records: dict[str, dict],
) -> set[str]:
    return {
        trajectory_id
        for trajectory_id, record
        in poison_records.items()
        if record.get(
            "data_label"
        ) == "poisoned"
    }


def choose_probe_ids(
    clean_records: dict[str, dict],
    poisoned_ids: set[str],
) -> set[str]:
    """
    Hold out clean, non-poisoned trajectories.

    All six poisoned trajectories remain in training so that
    their modified rewards can influence the learner.
    """

    rng = np.random.default_rng(
        SPLIT_SEED
    )

    probe_ids = set()

    for style in STYLE_ORDER:
        candidates = sorted(
            trajectory_id
            for trajectory_id, record
            in clean_records.items()
            if record["style"] == style
            and trajectory_id
            not in poisoned_ids
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


def flatten_state(
    observation,
) -> np.ndarray:
    return np.asarray(
        observation,
        dtype=np.float64,
    ).reshape(-1)


def collect_transitions(
    records: dict[str, dict],
    trajectory_ids: set[str],
):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for trajectory_id in sorted(
        trajectory_ids
    ):
        record = records[
            trajectory_id
        ]

        for transition in record[
            "transitions"
        ]:
            states.append(
                flatten_state(
                    transition[
                        "observation"
                    ]
                )
            )

            actions.append(
                int(
                    transition[
                        "action"
                    ]
                )
            )

            rewards.append(
                float(
                    transition[
                        "reward"
                    ]
                )
            )

            next_states.append(
                flatten_state(
                    transition[
                        "next_observation"
                    ]
                )
            )

            dones.append(
                bool(
                    transition[
                        "terminated"
                    ]
                    or transition[
                        "truncated"
                    ]
                )
            )

    return (
        np.asarray(
            states,
            dtype=np.float64,
        ),
        np.asarray(
            actions,
            dtype=np.int64,
        ),
        np.asarray(
            rewards,
            dtype=np.float64,
        ),
        np.asarray(
            next_states,
            dtype=np.float64,
        ),
        np.asarray(
            dones,
            dtype=np.float64,
        ),
    )


def fit_scaler(
    states: np.ndarray,
):
    mean = states.mean(
        axis=0
    )

    std = states.std(
        axis=0
    )

    std[
        std < 1e-8
    ] = 1.0

    return mean, std


def transform_states(
    states: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
):
    scaled = (
        states - mean
    ) / std

    # Add bias term.
    return np.column_stack(
        [
            scaled,
            np.ones(
                len(scaled),
                dtype=np.float64,
            ),
        ]
    )


def fit_fqi(
    x: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    x_next: np.ndarray,
    dones: np.ndarray,
    n_actions: int,
):
    """
    Linear fitted Q iteration.

    One linear Q function is fitted for each discrete action.
    """

    n_features = x.shape[1]

    weights = np.zeros(
        (
            n_actions,
            n_features,
        ),
        dtype=np.float64,
    )

    history = []

    identity = np.eye(
        n_features,
        dtype=np.float64,
    )

    # Regularise feature weights, but not the bias strongly.
    identity[-1, -1] = 0.01

    for iteration in range(
        1,
        N_ITERATIONS + 1,
    ):
        q_next = (
            x_next
            @ weights.T
        )

        max_next = q_next.max(
            axis=1
        )

        targets = (
            rewards
            + GAMMA
            * (1.0 - dones)
            * max_next
        )

        new_weights = (
            weights.copy()
        )

        for action in range(
            n_actions
        ):
            mask = (
                actions == action
            )

            if not mask.any():
                continue

            xa = x[mask]
            ya = targets[mask]

            matrix = (
                xa.T @ xa
                + RIDGE
                * identity
            )

            vector = (
                xa.T @ ya
            )

            new_weights[
                action
            ] = np.linalg.solve(
                matrix,
                vector,
            )

        change = float(
            np.max(
                np.abs(
                    new_weights
                    - weights
                )
            )
        )

        if not np.isfinite(
            new_weights
        ).all():
            raise RuntimeError(
                "FQI produced non-finite weights."
            )

        history.append(
            {
                "iteration":
                    iteration,
                "max_weight_change":
                    change,
                "mean_target":
                    float(
                        targets.mean()
                    ),
                "max_target":
                    float(
                        targets.max()
                    ),
            }
        )

        weights = new_weights

    return weights, history


def action_label_map(
    records: dict[str, dict],
) -> dict[int, str]:
    mapping = {}

    for record in records.values():
        for transition in record[
            "transitions"
        ]:
            mapping[
                int(
                    transition["action"]
                )
            ] = transition[
                "action_label"
            ]

    return mapping


def q_values(
    x: np.ndarray,
    weights: np.ndarray,
):
    return x @ weights.T


def q_margin(
    q: np.ndarray,
):
    ordered = np.sort(
        q,
        axis=1,
    )

    return (
        ordered[:, -1]
        - ordered[:, -2]
    )


def build_probe_rows(
    clean_records: dict[str, dict],
    probe_ids: set[str],
    mean: np.ndarray,
    std: np.ndarray,
    clean_weights: np.ndarray,
    poison_weights: np.ndarray,
    labels: dict[int, str],
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
            state = flatten_state(
                transition[
                    "observation"
                ]
            )[None, :]

            x = transform_states(
                state,
                mean,
                std,
            )

            clean_q = q_values(
                x,
                clean_weights,
            )[0]

            poison_q = q_values(
                x,
                poison_weights,
            )[0]

            clean_action = int(
                np.argmax(clean_q)
            )

            poison_action = int(
                np.argmax(poison_q)
            )

            sorted_clean = np.sort(
                clean_q
            )

            clean_margin = float(
                sorted_clean[-1]
                - sorted_clean[-2]
            )

            sorted_poison = np.sort(
                poison_q
            )

            poison_margin = float(
                sorted_poison[-1]
                - sorted_poison[-2]
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

                    "behavior_action":
                        transition[
                            "action_label"
                        ],

                    "clean_policy_action":
                        labels.get(
                            clean_action,
                            str(
                                clean_action
                            ),
                        ),

                    "poison_policy_action":
                        labels.get(
                            poison_action,
                            str(
                                poison_action
                            ),
                        ),

                    "action_changed":
                        clean_action
                        != poison_action,

                    "clean_q_margin":
                        clean_margin,

                    "poison_q_margin":
                        poison_margin,

                    "mean_abs_q_shift":
                        float(
                            np.mean(
                                np.abs(
                                    poison_q
                                    - clean_q
                                )
                            )
                        ),

                    "max_abs_q_shift":
                        float(
                            np.max(
                                np.abs(
                                    poison_q
                                    - clean_q
                                )
                            )
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

    verify_same_experience(
        clean_records,
        poison_records,
    )

    poison_ids = (
        poisoned_trajectory_ids(
            poison_records
        )
    )

    probe_ids = choose_probe_ids(
        clean_records,
        poison_ids,
    )

    all_ids = set(
        clean_records
    )

    train_ids = (
        all_ids
        - probe_ids
    )

    (
        clean_states,
        clean_actions,
        clean_rewards,
        clean_next_states,
        clean_dones,
    ) = collect_transitions(
        clean_records,
        train_ids,
    )

    (
        poison_states,
        poison_actions,
        poison_rewards,
        poison_next_states,
        poison_dones,
    ) = collect_transitions(
        poison_records,
        train_ids,
    )

    # Strong structural check after assembling training arrays.
    if not np.allclose(
        clean_states,
        poison_states,
    ):
        raise ValueError(
            "Training states differ."
        )

    if not np.array_equal(
        clean_actions,
        poison_actions,
    ):
        raise ValueError(
            "Training actions differ."
        )

    if not np.allclose(
        clean_next_states,
        poison_next_states,
    ):
        raise ValueError(
            "Training next states differ."
        )

    if not np.array_equal(
        clean_dones,
        poison_dones,
    ):
        raise ValueError(
            "Training terminal flags differ."
        )

    state_mean, state_std = (
        fit_scaler(
            clean_states
        )
    )

    clean_x = transform_states(
        clean_states,
        state_mean,
        state_std,
    )

    clean_x_next = (
        transform_states(
            clean_next_states,
            state_mean,
            state_std,
        )
    )

    # Same states/scaler for both models.
    poison_x = clean_x
    poison_x_next = clean_x_next

    n_actions = (
        int(
            clean_actions.max()
        )
        + 1
    )

    print(
        "\n=== TRAINING SET ==="
    )

    print(
        f"Training trajectories: "
        f"{len(train_ids)}"
    )

    print(
        f"Held-out probe trajectories: "
        f"{len(probe_ids)}"
    )

    print(
        f"Poisoned training trajectories: "
        f"{len(poison_ids & train_ids)}"
    )

    print(
        f"Training transitions: "
        f"{len(clean_actions)}"
    )

    action_counts = Counter(
        clean_actions.tolist()
    )

    labels = action_label_map(
        clean_records
    )

    print(
        "\nAction counts:"
    )

    for action in range(
        n_actions
    ):
        print(
            f"{action:>2} "
            f"{labels.get(action, str(action)):>12}: "
            f"{action_counts.get(action, 0)}"
        )

    reward_difference = (
        poison_rewards
        - clean_rewards
    )

    changed_rewards = int(
        np.count_nonzero(
            np.abs(
                reward_difference
            )
            > 1e-12
        )
    )

    print(
        f"\nModified training rewards: "
        f"{changed_rewards}"
    )

    print(
        f"Maximum reward change: "
        f"{reward_difference.max():.3f}"
    )

    clean_weights, clean_history = (
        fit_fqi(
            clean_x,
            clean_actions,
            clean_rewards,
            clean_x_next,
            clean_dones,
            n_actions,
        )
    )

    poison_weights, poison_history = (
        fit_fqi(
            poison_x,
            poison_actions,
            poison_rewards,
            poison_x_next,
            poison_dones,
            n_actions,
        )
    )

    probe_df = build_probe_rows(
        clean_records,
        probe_ids,
        state_mean,
        state_std,
        clean_weights,
        poison_weights,
        labels,
    )

    disagreement_rate = float(
        probe_df[
            "action_changed"
        ].mean()
    )

    changed_count = int(
        probe_df[
            "action_changed"
        ].sum()
    )

    low_margin_threshold = float(
        probe_df[
            "clean_q_margin"
        ].quantile(
            0.25
        )
    )

    low_margin = probe_df[
        probe_df[
            "clean_q_margin"
        ]
        <= low_margin_threshold
    ]

    low_margin_disagreement = (
        float(
            low_margin[
                "action_changed"
            ].mean()
        )
        if len(low_margin)
        else 0.0
    )

    finite_ttc = (
        np.isfinite(
            probe_df["ttc"]
        )
    )

    risky_mask = (
        (
            finite_ttc
            & (
                probe_df["ttc"]
                < 3.0
            )
        )
        | (
            probe_df[
                "front_gap"
            ]
            < 12.0
        )
    )

    risky = probe_df[
        risky_mask
    ]

    risky_disagreement = (
        float(
            risky[
                "action_changed"
            ].mean()
        )
        if len(risky)
        else None
    )

    by_style = (
        probe_df
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probe_df.to_csv(
        OUTPUT_DIR
        / "probe_states.csv",
        index=False,
    )

    history_rows = []

    for model_name, history in [
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

    switch_table = pd.crosstab(
        probe_df[
            "clean_policy_action"
        ],
        probe_df[
            "poison_policy_action"
        ],
    )

    switch_table.to_csv(
        OUTPUT_DIR
        / "action_switches.csv"
    )

    metrics = {
        "experiment":
            "stealthy_reward_policy_influence",

        "status":
            "rejected_due_to_q_value_instability",

        "learner":
            "linear_fitted_q_iteration",

        "gamma":
            GAMMA,

        "ridge":
            RIDGE,

        "iterations":
            N_ITERATIONS,

        "training_trajectories":
            len(train_ids),

        "probe_trajectories":
            len(probe_ids),

        "poisoned_training_trajectories":
            len(
                poison_ids
                & train_ids
            ),

        "modified_training_rewards":
            changed_rewards,

        "probe_states":
            len(probe_df),

        "changed_actions":
            changed_count,

        "action_disagreement_rate":
            disagreement_rate,

        "mean_abs_q_shift":
            float(
                probe_df[
                    "mean_abs_q_shift"
                ].mean()
            ),

        "max_abs_q_shift":
            float(
                probe_df[
                    "max_abs_q_shift"
                ].max()
            ),

        "low_margin_threshold":
            low_margin_threshold,

        "low_margin_disagreement_rate":
            low_margin_disagreement,

        "risky_probe_states":
            int(
                len(risky)
            ),

        "risky_state_disagreement_rate":
            risky_disagreement,

        "probe_ids":
            sorted(
                probe_ids
            ),

        "poisoned_ids":
            sorted(
                poison_ids
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
            allow_nan=True,
        )

    print(
        "\n=== POLICY INFLUENCE ON "
        "HELD-OUT CLEAN STATES ==="
    )

    print(
        f"Probe states: "
        f"{len(probe_df)}"
    )

    print(
        f"Changed greedy actions: "
        f"{changed_count}"
    )

    print(
        f"Action disagreement rate: "
        f"{disagreement_rate:.4f}"
    )

    print(
        f"Mean absolute Q shift: "
        f"{metrics['mean_abs_q_shift']:.4f}"
    )

    print(
        f"Maximum absolute Q shift: "
        f"{metrics['max_abs_q_shift']:.4f}"
    )

    print(
        f"Low-margin disagreement rate: "
        f"{low_margin_disagreement:.4f}"
    )

    print(
        "Risky-state disagreement rate: "
        + (
            f"{risky_disagreement:.4f}"
            if risky_disagreement
            is not None
            else "n/a"
        )
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

    print(
        "\n=== MOST AFFECTED "
        "HELD-OUT STATES ==="
    )

    most_affected = (
        probe_df
        .sort_values(
            "max_abs_q_shift",
            ascending=False,
        )
        .head(15)
    )

    print(
        most_affected[
            [
                "trajectory_id",
                "style",
                "step",
                "behavior_action",
                "clean_policy_action",
                "poison_policy_action",
                "action_changed",
                "clean_q_margin",
                "max_abs_q_shift",
                "speed",
                "front_gap",
                "ttc",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()