from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay_drive import (  # noqa: E402
    load_trajectory_records,
    render_replay,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CLEAN_DIR = ROOT / "results" / "clean_drives"
OBVIOUS_DIR = ROOT / "results" / "obvious_reward_poison"
STEALTHY_DIR = ROOT / "results" / "stealthy_reward_poison"
TARGETED_DIR = ROOT / "results" / "decision_targeted_reward_poison"

POLICY_DIR = ROOT / "results" / "policy_influence"
TARGETED_POLICY_DIR = ROOT / "results" / "decision_targeted_policy_influence"
BUDGET_SWEEP_DIR = ROOT / "results" / "targeted_budget_sweep"
RANDOM_CONTROL_DIR = ROOT / "results" / "targeting_random_control"
SPLIT_ROBUSTNESS_DIR = ROOT / "results" / "split_robustness"


# ---------------------------------------------------------------------
# Page setup / styling
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Reward Poisoning Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(120, 120, 120, 0.25);
        border-radius: 18px;
        background:
            radial-gradient(circle at 10% 10%, rgba(48, 120, 255, 0.16), transparent 35%),
            radial-gradient(circle at 90% 0%, rgba(157, 78, 221, 0.13), transparent 32%);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        font-size: 0.82rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        opacity: 0.68;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        font-size: 2.05rem;
        font-weight: 760;
        line-height: 1.08;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        font-size: 1.03rem;
        opacity: 0.78;
        max-width: 1000px;
    }

    .thesis {
        padding: 0.95rem 1.1rem;
        border-left: 4px solid #4f8cff;
        background: rgba(79, 140, 255, 0.08);
        border-radius: 10px;
        margin: 0.8rem 0 1.25rem 0;
        font-size: 1.03rem;
    }

    .mini-card {
        padding: 0.85rem 1rem;
        border: 1px solid rgba(120, 120, 120, 0.22);
        border-radius: 14px;
        min-height: 108px;
    }

    .phase-pill {
        display: inline-block;
        padding: 0.22rem 0.58rem;
        border-radius: 999px;
        border: 1px solid rgba(120, 120, 120, 0.28);
        font-size: 0.78rem;
        margin-right: 0.3rem;
        opacity: 0.83;
    }

    .big-number {
        font-size: 2.0rem;
        font-weight: 760;
        line-height: 1;
    }

    .muted {
        opacity: 0.68;
    }

    .flip-clean {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(41, 171, 135, 0.09);
        border: 1px solid rgba(41, 171, 135, 0.30);
    }

    .flip-poison {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(255, 93, 93, 0.08);
        border: 1px solid rgba(255, 93, 93, 0.30);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def require_path(path: Path, command: str | None = None) -> None:
    if path.exists():
        return

    message = f"Missing required file: `{path.relative_to(ROOT)}`."
    if command:
        message += f"\n\nGenerate it with:\n\n`{command}`"

    st.error(message)
    st.stop()


@st.cache_data(show_spinner=False)
def load_csv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(Path(path_string))


@st.cache_data(show_spinner=False)
def load_json(path_string: str) -> dict:
    with Path(path_string).open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_records(path_string: str) -> dict[str, dict]:
    return load_trajectory_records(Path(path_string))


@st.cache_data(show_spinner=False)
def replay_from_directory(
    directory_string: str,
    trajectory_id: str,
):
    records = load_records(directory_string)
    return render_replay(records[trajectory_id])


def finite_text(value, digits: int = 2) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)

    if not np.isfinite(value):
        return "∞"

    return f"{value:.{digits}f}"


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


def metric_percent(label: str, value: float, delta=None):
    st.metric(
        label,
        f"{100.0 * float(value):.2f}%",
        delta=delta,
    )


def hero(kicker: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def thesis(text: str):
    st.markdown(
        f'<div class="thesis">{text}</div>',
        unsafe_allow_html=True,
    )


def detector_row(df: pd.DataFrame | None, trajectory_id: str):
    if df is None:
        return None

    match = df[
        df["trajectory_id"] == trajectory_id
    ]

    if match.empty:
        return None

    return match.iloc[0]


def render_frame(
    frames,
    index: int,
    caption: str,
):
    index = max(
        0,
        min(
            int(index),
            len(frames) - 1,
        ),
    )

    st.image(
        frames[index],
        caption=caption,
        use_container_width=True,
    )


def action_arrow(clean_action: str, poison_action: str) -> str:
    return (
        f"{clean_action}  →  {poison_action}"
        if clean_action != poison_action
        else clean_action
    )


def phase_bar(active: int):
    labels = [
        "Clean variation",
        "Anomaly failure",
        "Stealth",
        "Targeting",
        "Policy flip",
        "Robustness",
    ]

    pieces = []

    for index, label in enumerate(labels):
        weight = "font-weight:700;" if index == active else ""
        opacity = "opacity:1;" if index == active else "opacity:.48;"
        pieces.append(
            f'<span class="phase-pill" style="{weight}{opacity}">'
            f'{index + 1} · {label}</span>'
        )

    st.markdown(
        " ".join(pieces),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Common data
# ---------------------------------------------------------------------

require_path(
    CLEAN_DIR / "trajectories.jsonl",
    "python -m src.collect_clean_drives",
)

clean_records = load_records(
    str(CLEAN_DIR)
)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧪 Reward Poisoning Lab")
    st.caption(
        "Interactive presentation · Scientific Thinking"
    )

    chapter = st.radio(
        "Presentation chapter",
        [
            "1 · Clean ≠ identical",
            "2 · Anomaly ≠ poison",
            "3 · Obvious → stealthy",
            "4 · Placement matters",
            "5 · Decision flip explorer",
            "6 · Robustness",
        ],
        index=0,
    )

    st.divider()

    st.markdown("**Central finding**")
    st.write(
        "A small reward attack becomes influential when "
        "its edits are placed near decision-sensitive states."
    )

    st.caption(
        "Use the chapters in order during the presentation, "
        "or jump directly to the policy-flip explorer."
    )


# ---------------------------------------------------------------------
# Chapter 1
# ---------------------------------------------------------------------

if chapter.startswith("1"):
    phase_bar(0)

    hero(
        "Chapter 1 · Establish the control",
        "Clean driving is diverse by design",
        (
            "Before discussing poisoning, we need a clean reference "
            "that contains legitimate cautious, normal, and aggressive behavior."
        ),
    )

    thesis(
        "<b>Question:</b> can a clean dataset contain behavior that looks "
        "unusual without being malicious?"
    )

    summary_path = CLEAN_DIR / "summary.csv"
    require_path(
        summary_path,
        "python -m src.collect_clean_drives",
    )

    clean_summary = load_csv(
        str(summary_path)
    )

    style_agg = (
        clean_summary
        .groupby("style")
        .agg(
            trajectories=(
                "trajectory_id",
                "count",
            ),
            crash_rate=(
                "crashed",
                "mean",
            ),
            mean_reward=(
                "mean_reward",
                "mean",
            ),
            mean_speed=(
                "mean_speed",
                "mean",
            ),
            lane_changes=(
                "lane_changes",
                "mean",
            ),
            min_ttc=(
                "min_ttc",
                "median",
            ),
        )
        .reindex(
            [
                "cautious",
                "normal",
                "aggressive",
            ]
        )
        .reset_index()
    )

    cols = st.columns(3)

    for column, (_, row) in zip(
        cols,
        style_agg.iterrows(),
    ):
        with column:
            st.markdown(
                f"### {row['style'].capitalize()}"
            )

            a, b = st.columns(2)
            a.metric(
                "Mean speed",
                f"{row['mean_speed']:.2f} m/s",
            )
            b.metric(
                "Lane changes",
                f"{row['lane_changes']:.2f}",
            )

            c, d = st.columns(2)
            c.metric(
                "Crash rate",
                f"{100 * row['crash_rate']:.1f}%",
            )
            d.metric(
                "Median min TTC",
                finite_text(
                    row["min_ttc"],
                    2,
                ),
            )

    st.divider()
    st.subheader("Replay a clean driving style")

    clean_ids = {
        style: clean_summary[
            clean_summary["style"] == style
        ]["trajectory_id"].tolist()
        for style in [
            "cautious",
            "normal",
            "aggressive",
        ]
    }

    style_choice = st.selectbox(
        "Style",
        list(clean_ids.keys()),
        format_func=lambda x: x.capitalize(),
    )

    trajectory_id = st.selectbox(
        "Trajectory",
        clean_ids[style_choice],
    )

    frames, live_steps = replay_from_directory(
        str(CLEAN_DIR),
        trajectory_id,
    )

    left, right = st.columns(
        [2.1, 1.0],
        gap="large",
    )

    with left:
        image_slot = st.empty()

        controls = st.columns(
            [1.0, 1.0, 2.2]
        )

        play = controls[0].button(
            "▶ Play",
            key="clean_play",
            use_container_width=True,
        )

        speed_label = controls[1].selectbox(
            "Playback",
            ["0.5x", "1x", "2x", "4x"],
            index=1,
            key="clean_speed",
        )

        delays = {
            "0.5x": 0.16,
            "1x": 0.09,
            "2x": 0.05,
            "4x": 0.025,
        }

        step = controls[2].slider(
            "Manual step",
            1,
            len(frames),
            1,
            key="clean_step",
        ) - 1

    with right:
        live_slot = st.empty()

    def draw_clean_step(index: int):
        image_slot.image(
            frames[index],
            caption=(
                f"{trajectory_id} · clean · "
                f"step {index + 1}/{len(frames)}"
            ),
            use_container_width=True,
        )

        info = live_steps[index]

        with live_slot.container():
            st.markdown("#### Live state")

            st.metric(
                "Action",
                info["action"],
            )

            a, b = st.columns(2)

            a.metric(
                "Speed",
                f"{info['speed_mps']:.2f} m/s",
            )

            b.metric(
                "Lane",
                str(info["lane"]),
            )

            c, d = st.columns(2)

            c.metric(
                "Front gap",
                f"{info['front_gap_m']:.2f} m",
            )

            d.metric(
                "TTC",
                finite_text(
                    info["ttc_s"],
                    2,
                ),
            )

            st.metric(
                "Safety band",
                info["safety_band"],
            )

    if play:
        for index in range(
            step,
            len(frames),
        ):
            draw_clean_step(index)
            time.sleep(
                delays[speed_label]
            )
    else:
        draw_clean_step(step)


# ---------------------------------------------------------------------
# Chapter 2
# ---------------------------------------------------------------------

elif chapter.startswith("2"):
    phase_bar(1)

    hero(
        "Chapter 2 · Challenge the first hypothesis",
        "Statistical anomaly is not the same as poisoning",
        (
            "We fit a transparent z-score detector to data that is known "
            "to be 100% clean. Any alarm is therefore a false positive."
        ),
    )

    anomaly_path = (
        CLEAN_DIR
        / "clean_anomaly_ranking.csv"
    )

    require_path(
        anomaly_path,
        "python -m src.inspect_clean_anomalies",
    )

    anomaly = load_csv(
        str(anomaly_path)
    )

    global_flags = anomaly[
        "global_flag"
    ].map(as_bool)

    style_flags = anomaly[
        "style_flag"
    ].map(as_bool)

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Known-clean trajectories",
        len(anomaly),
    )

    m2.metric(
        "Global false positives",
        f"{int(global_flags.sum())}/{len(anomaly)}",
    )

    m3.metric(
        "Style-aware false positives",
        f"{int(style_flags.sum())}/{len(anomaly)}",
    )

    thesis(
        "<b>Revision:</b> unusual behavior can be legitimate. "
        "A detector that equates rarity with poisoning will accuse clean data."
    )

    flagged = (
        anomaly[
            global_flags
        ]
        .sort_values(
            "global_anomaly_score",
            ascending=False,
        )
    )

    if flagged.empty:
        st.warning(
            "No globally flagged clean trajectories were found."
        )
        st.stop()

    selected_id = st.selectbox(
        "Inspect a clean trajectory the detector calls suspicious",
        flagged["trajectory_id"].tolist(),
    )

    row = flagged[
        flagged["trajectory_id"]
        == selected_id
    ].iloc[0]

    frames, live_steps = replay_from_directory(
        str(CLEAN_DIR),
        selected_id,
    )

    top_left, top_right = st.columns(
        [1.9, 1.0],
        gap="large",
    )

    with top_left:
        image_slot = st.empty()

        controls = st.columns(
            [1.0, 1.0, 2.2]
        )

        play = controls[0].button(
            "▶ Play",
            key="false_positive_play",
            use_container_width=True,
        )

        speed_label = controls[1].selectbox(
            "Playback",
            ["0.5x", "1x", "2x", "4x"],
            index=1,
            key="false_positive_speed",
        )

        delays = {
            "0.5x": 0.16,
            "1x": 0.09,
            "2x": 0.05,
            "4x": 0.025,
        }

        step = controls[2].slider(
            "Manual step",
            1,
            len(frames),
            1,
            key="false_positive_step",
        ) - 1

    with top_right:
        live_slot = st.empty()

    def draw_false_positive_step(index: int):
        image_slot.image(
            frames[index],
            caption=(
                f"{selected_id} · CLEAN · "
                f"detector: SUSPICIOUS · "
                f"step {index + 1}/{len(frames)}"
            ),
            use_container_width=True,
        )

        info = live_steps[index]

        with live_slot.container():
            st.metric(
                "Ground truth",
                "CLEAN",
            )

            st.metric(
                "Detector",
                "SUSPICIOUS",
            )

            st.metric(
                "Anomaly score",
                f"{float(row['global_anomaly_score']):.3f}",
            )

            st.metric(
                "Dominant feature",
                str(
                    row[
                        "global_dominant_feature"
                    ]
                ),
            )

            a, b = st.columns(2)

            a.metric(
                "Current action",
                info["action"],
            )

            b.metric(
                "Speed",
                f"{info['speed_mps']:.2f} m/s",
            )

            st.info(
                "The replay is genuine clean data. "
                "The false alarm comes from statistical unusualness, "
                "not corruption."
            )

    if play:
        for index in range(
            step,
            len(frames),
        ):
            draw_false_positive_step(index)
            time.sleep(
                delays[speed_label]
            )
    else:
        draw_false_positive_step(step)

    st.subheader(
        "Highest-scoring clean trajectories"
    )

    display = anomaly[
        [
            "trajectory_id",
            "style",
            "global_anomaly_score",
            "global_flag",
            "global_dominant_feature",
        ]
    ].sort_values(
        "global_anomaly_score",
        ascending=False,
    ).head(12)

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Chapter 3
# ---------------------------------------------------------------------

elif chapter.startswith("3"):
    phase_bar(2)

    hero(
        "Chapter 3 · Reduce the attack magnitude",
        "Obvious poison is easy. Small poison can blend in.",
        (
            "The physical trajectory is held fixed. We alter only stored rewards "
            "and compare clean, obvious, and stealthy versions of the same drive."
        ),
    )

    for path, command in [
        (
            OBVIOUS_DIR / "trajectories.jsonl",
            "python -m src.create_obvious_reward_poison",
        ),
        (
            STEALTHY_DIR / "trajectories.jsonl",
            "python -m src.create_stealthy_reward_poison",
        ),
        (
            OBVIOUS_DIR / "detector_metrics.json",
            "python -m src.evaluate_reward_poison --dataset obvious_reward_poison",
        ),
        (
            STEALTHY_DIR / "detector_metrics.json",
            "python -m src.evaluate_reward_poison --dataset stealthy_reward_poison",
        ),
    ]:
        require_path(
            path,
            command,
        )

    obvious_records = load_records(
        str(OBVIOUS_DIR)
    )

    stealthy_records = load_records(
        str(STEALTHY_DIR)
    )

    common_poisoned_ids = sorted(
        trajectory_id
        for trajectory_id, record
        in stealthy_records.items()
        if record.get("data_label")
        == "poisoned"
        and trajectory_id
        in obvious_records
    )

    obvious_metrics = load_json(
        str(
            OBVIOUS_DIR
            / "detector_metrics.json"
        )
    )

    stealthy_metrics = load_json(
        str(
            STEALTHY_DIR
            / "detector_metrics.json"
        )
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_percent(
            "Obvious-poison recall",
            obvious_metrics[
                "global_detector"
            ]["recall"],
        )

    with c2:
        metric_percent(
            "Stealthy-poison recall",
            stealthy_metrics[
                "global_detector"
            ]["recall"],
        )

    with c3:
        st.metric(
            "Stealthy reward edit",
            "+0.25",
        )

    thesis(
        "<b>Key observation:</b> reducing reward corruption sharply lowers detector recall, "
        "but stealthiness alone does not tell us whether the learner will actually change."
    )

    trajectory_id = st.selectbox(
        "Choose one attacked trajectory",
        common_poisoned_ids,
    )

    clean_record = clean_records[
        trajectory_id
    ]

    obvious_record = obvious_records[
        trajectory_id
    ]

    stealthy_record = stealthy_records[
        trajectory_id
    ]

    frames, _ = replay_from_directory(
        str(CLEAN_DIR),
        trajectory_id,
    )

    rows = []

    for index, (
        clean_t,
        obvious_t,
        stealthy_t,
    ) in enumerate(
        zip(
            clean_record["transitions"],
            obvious_record["transitions"],
            stealthy_record["transitions"],
        )
    ):
        rows.append(
            {
                "Step": index + 1,
                "Action":
                    clean_t["action_label"],
                "Clean reward":
                    float(
                        clean_t["reward"]
                    ),
                "Obvious reward":
                    float(
                        obvious_t["reward"]
                    ),
                "Stealthy reward":
                    float(
                        stealthy_t["reward"]
                    ),
                "Modified":
                    as_bool(
                        stealthy_t.get(
                            "reward_poisoned",
                            False,
                        )
                    ),
                "Speed":
                    float(
                        clean_t[
                            "ego_after"
                        ]["speed"]
                    ),
                "TTC":
                    float(
                        clean_t[
                            "ego_after"
                        ]["ttc"]
                    ),
            }
        )

    reward_df = pd.DataFrame(
        rows
    )

    modified_indices = reward_df.index[
        reward_df["Modified"]
    ].tolist()

    default_index = (
        modified_indices[0]
        if modified_indices
        else 0
    )

    left, right = st.columns(
        [2.0, 1.15],
        gap="large",
    )

    with left:
        image_slot = st.empty()

        controls = st.columns(
            [1.0, 1.0, 2.2]
        )

        play = controls[0].button(
            "▶ Play",
            key=f"reward_play_{trajectory_id}",
            use_container_width=True,
        )

        speed_label = controls[1].selectbox(
            "Playback",
            ["0.5x", "1x", "2x", "4x"],
            index=1,
            key=f"reward_speed_{trajectory_id}",
        )

        delays = {
            "0.5x": 0.16,
            "1x": 0.09,
            "2x": 0.05,
            "4x": 0.025,
        }

        step = controls[2].slider(
            "Manual transition",
            1,
            len(frames),
            default_index + 1,
            key=f"reward_compare_{trajectory_id}",
        ) - 1

    with right:
        live_slot = st.empty()

    def draw_reward_step(index: int):
        image_slot.image(
            frames[index],
            caption=(
                f"{trajectory_id} · same physical transition · "
                f"step {index + 1}/{len(frames)}"
            ),
            use_container_width=True,
        )

        current = reward_df.iloc[
            index
        ]

        with live_slot.container():
            st.metric(
                "Behavior action",
                current["Action"],
            )

            st.markdown(
                "#### Stored reward"
            )

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Clean",
                f"{current['Clean reward']:.3f}",
            )

            r2.metric(
                "Obvious",
                f"{current['Obvious reward']:.3f}",
            )

            r3.metric(
                "Stealthy",
                f"{current['Stealthy reward']:.3f}",
            )

            if current["Modified"]:
                st.warning(
                    "⚠ Reward modified here. "
                    "State, action, and next-state are unchanged."
                )
            else:
                st.success(
                    "Reward unchanged on this transition."
                )

    if play:
        for index in range(
            step,
            len(frames),
        ):
            draw_reward_step(index)
            time.sleep(
                delays[speed_label]
            )
    else:
        draw_reward_step(step)

    chart = reward_df[
        [
            "Step",
            "Clean reward",
            "Obvious reward",
            "Stealthy reward",
        ]
    ].set_index("Step")

    st.subheader(
        "Reward signal across the same drive"
    )

    st.line_chart(
        chart,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Chapter 4
# ---------------------------------------------------------------------

elif chapter.startswith("4"):
    phase_bar(3)

    hero(
        "Chapter 4 · Same magnitude, different placement",
        "Strategic placement dominates attack size",
        (
            "Random stealthy edits changed no held-out actions in the stable learner. "
            "Targeting low-margin runner-up actions changes the outcome dramatically."
        ),
    )

    required = [
        (
            POLICY_DIR / "metrics.json",
            "python -m src.measure_policy_influence",
        ),
        (
            TARGETED_POLICY_DIR / "metrics.json",
            "python -m src.measure_targeted_policy_influence",
        ),
        (
            BUDGET_SWEEP_DIR / "budget_sweep.csv",
            "python -m src.sweep_targeted_attack_budget",
        ),
        (
            RANDOM_CONTROL_DIR / "summary.csv",
            "python -m src.compare_targeting_to_random",
        ),
    ]

    for path, command in required:
        require_path(
            path,
            command,
        )

    random_policy = load_json(
        str(
            POLICY_DIR
            / "metrics.json"
        )
    )

    targeted_policy = load_json(
        str(
            TARGETED_POLICY_DIR
            / "metrics.json"
        )
    )

    budget = load_csv(
        str(
            BUDGET_SWEEP_DIR
            / "budget_sweep.csv"
        )
    )

    control = load_csv(
        str(
            RANDOM_CONTROL_DIR
            / "summary.csv"
        )
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Random edits",
        random_policy[
            "modified_training_rewards"
        ],
    )

    with b:
        metric_percent(
            "Random influence",
            random_policy[
                "action_disagreement_rate"
            ],
        )

    c.metric(
        "Targeted edits",
        targeted_policy[
            "modified_training_rewards"
        ],
    )

    with d:
        metric_percent(
            "Targeted influence",
            targeted_policy[
                "action_disagreement_rate"
            ],
        )

    thesis(
        "<b>Same 180 × +0.25 budget:</b> random placement produced 0% disagreement, "
        "while targeted placement produced 27.45% on the same held-out probe states."
    )

    st.subheader(
        "How much targeting is actually necessary?"
    )

    chart = budget[
        [
            "budget",
            "action_disagreement_rate",
            "global_detector_recall",
        ]
    ].copy()

    chart = chart.rename(
        columns={
            "budget":
                "Reward edits",
            "action_disagreement_rate":
                "Policy disagreement",
            "global_detector_recall":
                "Detector recall",
        }
    ).set_index(
        "Reward edits"
    )

    st.line_chart(
        chart,
        use_container_width=True,
    )

    best_20 = budget[
        budget["budget"] == 20
    ]

    if not best_20.empty:
        row20 = best_20.iloc[0]

        x, y, z = st.columns(3)

        x.metric(
            "20 targeted edits",
            f"{100 * row20['action_disagreement_rate']:.2f}% influence",
        )

        y.metric(
            "Total reward added",
            f"{row20['total_reward_budget']:.2f}",
        )

        z.metric(
            "Global detector recall",
            f"{100 * row20['global_detector_recall']:.2f}%",
        )

    st.subheader(
        "Targeted vs matched-random controls"
    )

    display = control[
        [
            "budget",
            "targeted_rate",
            "random_mean_rate",
            "random_p95_rate",
            "empirical_p_value",
        ]
    ].copy()

    display["targeted_rate"] *= 100
    display["random_mean_rate"] *= 100
    display["random_p95_rate"] *= 100

    display = display.rename(
        columns={
            "budget":
                "Budget",
            "targeted_rate":
                "Targeted %",
            "random_mean_rate":
                "Random mean %",
            "random_p95_rate":
                "Random p95 %",
            "empirical_p_value":
                "Empirical p",
        }
    )

    st.dataframe(
        display.style.format(
            {
                "Targeted %":
                    "{:.2f}",
                "Random mean %":
                    "{:.2f}",
                "Random p95 %":
                    "{:.2f}",
                "Empirical p":
                    "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Matched random controls preserve the number of edits per trajectory; "
        "only transition placement is randomized."
    )


# ---------------------------------------------------------------------
# Chapter 5
# ---------------------------------------------------------------------

elif chapter.startswith("5"):
    phase_bar(4)

    hero(
        "Chapter 5 · Inspect the mechanism",
        "Where does the learned policy actually flip?",
        (
            "Select a held-out clean state where the converged clean policy "
            "and targeted-poison policy recommend different actions."
        ),
    )

    probe_path = (
        TARGETED_POLICY_DIR
        / "probe_states.csv"
    )

    require_path(
        probe_path,
        "python -m src.measure_targeted_policy_influence",
    )

    probe = load_csv(
        str(probe_path)
    )

    changed = probe[
        probe["covered"].map(
            as_bool
        )
        & probe[
            "action_changed"
        ].map(
            as_bool
        )
    ].copy()

    if changed.empty:
        st.warning(
            "No changed held-out decisions were found."
        )
        st.stop()

    changed = changed.sort_values(
        [
            "max_abs_q_shift",
            "clean_q_margin",
        ],
        ascending=[
            False,
            True,
        ],
    )

    style = st.selectbox(
        "Driving style",
        sorted(
            changed["style"].unique()
        ),
        index=0,
    )

    style_part = changed[
        changed["style"] == style
    ]

    options = []

    for _, row in style_part.iterrows():
        options.append(
            (
                row["trajectory_id"],
                int(row["step"]),
                (
                    f"{row['trajectory_id']} · step {int(row['step']) + 1} · "
                    f"{row['clean_policy_action']} → {row['poison_policy_action']} · "
                    f"margin {row['clean_q_margin']:.4f}"
                ),
            )
        )

    selected_label = st.selectbox(
        "Changed decision",
        [item[2] for item in options],
    )

    selected = next(
        item
        for item in options
        if item[2] == selected_label
    )

    trajectory_id = selected[0]
    affected_step = selected[1]

    row = style_part[
        (
            style_part["trajectory_id"]
            == trajectory_id
        )
        & (
            style_part["step"]
            == affected_step
        )
    ].iloc[0]

    frames, _ = replay_from_directory(
        str(CLEAN_DIR),
        trajectory_id,
    )

    thesis(
        "<b>Important:</b> this replay is still the original clean behavior trajectory. "
        "The two actions below are <i>counterfactual greedy recommendations</i> "
        "from the clean and targeted-poison learners at the same observed state."
    )

    left, right = st.columns(
        [2.0, 1.15],
        gap="large",
    )

    low = max(
        0,
        affected_step - 8,
    )

    high = min(
        len(frames) - 1,
        affected_step + 8,
    )

    with left:
        image_slot = st.empty()

        controls = st.columns(
            [1.0, 1.0, 2.2]
        )

        play = controls[0].button(
            "▶ Play window",
            key=(
                f"flip_play_{trajectory_id}_"
                f"{affected_step}"
            ),
            use_container_width=True,
        )

        speed_label = controls[1].selectbox(
            "Playback",
            ["0.5x", "1x", "2x", "4x"],
            index=1,
            key=(
                f"flip_speed_{trajectory_id}_"
                f"{affected_step}"
            ),
        )

        delays = {
            "0.5x": 0.18,
            "1x": 0.10,
            "2x": 0.055,
            "4x": 0.03,
        }

        display_step = controls[2].slider(
            "Nearby replay step",
            low + 1,
            high + 1,
            affected_step + 1,
            key=(
                f"flip_{trajectory_id}_"
                f"{affected_step}"
            ),
        ) - 1

    with right:
        live_slot = st.empty()

    def draw_flip_step(index: int):
        image_slot.image(
            frames[index],
            caption=(
                f"{trajectory_id} · "
                f"affected decision = step {affected_step + 1} · "
                f"viewing step {index + 1}"
            ),
            use_container_width=True,
        )

        exact = (
            index == affected_step
        )

        with live_slot.container():
            if exact:
                st.error(
                    "⚡ EXACT POLICY-FLIP STATE"
                )
            else:
                st.caption(
                    "Context frame near the affected state."
                )

            st.markdown(
                f"""
                <div class="flip-clean">
                    <div class="muted">Clean learned policy</div>
                    <div class="big-number">{row['clean_policy_action']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='height:0.55rem'></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="flip-poison">
                    <div class="muted">Targeted-poison learned policy</div>
                    <div class="big-number">{row['poison_policy_action']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "#### Why this state was sensitive"
            )

            a, b = st.columns(2)

            a.metric(
                "Clean Q-margin",
                f"{row['clean_q_margin']:.4f}",
            )

            b.metric(
                "Max |Q shift|",
                f"{row['max_abs_q_shift']:.3f}",
            )

            c, d = st.columns(2)

            c.metric(
                "Speed",
                f"{row['speed']:.2f} m/s",
            )

            d.metric(
                "Front gap",
                f"{row['front_gap']:.2f} m",
            )

            st.metric(
                "TTC",
                finite_text(
                    row["ttc"],
                    2,
                ),
            )

            st.caption(
                f"Observed behavior action in the dataset: "
                f"{row['behavior_action']}"
            )

    if play:
        for index in range(
            low,
            high + 1,
        ):
            draw_flip_step(index)

            if index == affected_step:
                time.sleep(
                    0.55
                )
            else:
                time.sleep(
                    delays[speed_label]
                )
    else:
        draw_flip_step(
            display_step
        )

    st.subheader(
        "Most common action switches in held-out states"
    )

    switch_path = (
        TARGETED_POLICY_DIR
        / "action_switches.csv"
    )

    if switch_path.exists():
        switches = load_csv(
            str(switch_path)
        )

        st.dataframe(
            switches,
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# Chapter 6
# ---------------------------------------------------------------------

elif chapter.startswith("6"):
    phase_bar(5)

    hero(
        "Chapter 6 · Robustness",
        "The targeting advantage survives new train–probe splits",
        (
            "We recompute the targeted 20-edit attack from scratch on ten different "
            "training splits and compare it with the much larger 180-edit random attack."
        ),
    )

    split_path = (
        SPLIT_ROBUSTNESS_DIR
        / "split_results.csv"
    )

    summary_path = (
        SPLIT_ROBUSTNESS_DIR
        / "summary.json"
    )

    require_path(
        split_path,
        "python -m src.validate_targeting_across_splits",
    )

    require_path(
        summary_path,
        "python -m src.validate_targeting_across_splits",
    )

    split = load_csv(
        str(split_path)
    )

    summary = load_json(
        str(summary_path)
    )

    a, b, c, d = st.columns(4)

    with a:
        metric_percent(
            "Random 180-edit mean",
            summary[
                "random_mean_rate"
            ],
        )

    with b:
        metric_percent(
            "Targeted 20-edit mean",
            summary[
                "targeted_mean_rate"
            ],
        )

    c.metric(
        "Targeted wins",
        (
            f"{summary['targeted_wins']}/"
            f"{summary['splits']}"
        ),
    )

    d.metric(
        "Sign-test p",
        f"{summary['one_sided_sign_test_p']:.6f}",
    )

    thesis(
        "<b>Cross-split result:</b> 20 targeted edits beat 180 random edits "
        "in every tested split."
    )

    chart = split[
        [
            "split_seed",
            "random_disagreement_rate",
            "targeted_disagreement_rate",
        ]
    ].copy()

    chart = chart.rename(
        columns={
            "split_seed":
                "Split seed",
            "random_disagreement_rate":
                "Random · 180 edits",
            "targeted_disagreement_rate":
                "Targeted · 20 edits",
        }
    ).set_index(
        "Split seed"
    )

    st.line_chart(
        chart,
        use_container_width=True,
    )

    display = split[
        [
            "split_seed",
            "random_changed_actions",
            "random_disagreement_rate",
            "targeted_changed_actions",
            "targeted_disagreement_rate",
            "paired_difference",
            "coverage_rate",
        ]
    ].copy()

    for column in [
        "random_disagreement_rate",
        "targeted_disagreement_rate",
        "paired_difference",
        "coverage_rate",
    ]:
        display[column] *= 100

    display = display.rename(
        columns={
            "split_seed":
                "Split",
            "random_changed_actions":
                "Random changed",
            "random_disagreement_rate":
                "Random %",
            "targeted_changed_actions":
                "Targeted changed",
            "targeted_disagreement_rate":
                "Targeted %",
            "paired_difference":
                "Advantage pp",
            "coverage_rate":
                "Coverage %",
        }
    )

    st.dataframe(
        display.style.format(
            {
                "Random %":
                    "{:.2f}",
                "Targeted %":
                    "{:.2f}",
                "Advantage pp":
                    "{:.2f}",
                "Coverage %":
                    "{:.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.success(
        "Experimental story complete: the effect is not tied to one probe split."
    )

    st.markdown("### Final takeaway")

    st.markdown(
        """
        **1.** Clean data can look anomalous.  
        **2.** Small random reward poisoning can be stealthy yet ineffective.  
        **3.** Moving the same type of reward edits toward low-margin decisions changes the learned policy.  
        **4.** Only 20 targeted edits recover almost the full 180-edit effect.  
        **5.** The advantage survives matched-random controls and 10 new train–probe splits.
        """
    )
