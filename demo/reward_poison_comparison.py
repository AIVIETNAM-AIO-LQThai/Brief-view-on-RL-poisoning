from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay_drive import (  # noqa: E402
    load_trajectory_records,
    render_replay,
)


CLEAN_DIR = (
    ROOT
    / "results"
    / "clean_drives"
)

OBVIOUS_DIR = (
    ROOT
    / "results"
    / "obvious_reward_poison"
)

STEALTHY_DIR = (
    ROOT
    / "results"
    / "stealthy_reward_poison"
)


st.set_page_config(
    page_title="Reward Poisoning Comparison",
    layout="wide",
)

st.title("Reward Poisoning Comparison")

st.caption(
    "The car follows the same trajectory in all three versions. "
    "Only the stored reward signal changes."
)


@st.cache_data(show_spinner=False)
def load_records(dataset_dir: str):
    return load_trajectory_records(
        Path(dataset_dir)
    )


@st.cache_data(show_spinner=False)
def load_detector_results(
    dataset_dir: str,
):
    path = (
        Path(dataset_dir)
        / "detector_results.csv"
    )

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_clean_detector():
    path = (
        CLEAN_DIR
        / "clean_anomaly_ranking.csv"
    )

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_detector_metrics(
    dataset_dir: str,
):
    path = (
        Path(dataset_dir)
        / "detector_metrics.json"
    )

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def replay_clean(
    trajectory_id: str,
):
    clean_records = load_records(
        str(CLEAN_DIR)
    )

    return render_replay(
        clean_records[trajectory_id]
    )


clean_records = load_records(
    str(CLEAN_DIR)
)

obvious_records = load_records(
    str(OBVIOUS_DIR)
)

stealthy_records = load_records(
    str(STEALTHY_DIR)
)


poisoned_ids = sorted(
    trajectory_id
    for trajectory_id, record
    in stealthy_records.items()
    if record.get("data_label")
    == "poisoned"
)


trajectory_id = st.selectbox(
    "Choose one of the poisoned trajectories",
    poisoned_ids,
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


frames, live_steps = replay_clean(
    trajectory_id
)


def build_reward_table():
    rows = []

    clean_transitions = (
        clean_record["transitions"]
    )

    obvious_transitions = (
        obvious_record["transitions"]
    )

    stealthy_transitions = (
        stealthy_record["transitions"]
    )

    for index, (
        clean_t,
        obvious_t,
        stealthy_t,
    ) in enumerate(
        zip(
            clean_transitions,
            obvious_transitions,
            stealthy_transitions,
        )
    ):

        rows.append(
            {
                "Step": index + 1,

                "Action":
                    clean_t[
                        "action_label"
                    ],

                "Speed (m/s)":
                    clean_t[
                        "ego_after"
                    ]["speed"],

                "Front gap (m)":
                    clean_t[
                        "ego_after"
                    ]["front_gap"],

                "TTC (s)":
                    clean_t[
                        "ego_after"
                    ]["ttc"],

                "Clean reward":
                    clean_t[
                        "reward"
                    ],

                "Obvious reward":
                    obvious_t[
                        "reward"
                    ],

                "Stealthy reward":
                    stealthy_t[
                        "reward"
                    ],

                "Reward modified":
                    bool(
                        stealthy_t.get(
                            "reward_poisoned",
                            False,
                        )
                    ),
            }
        )

    return pd.DataFrame(rows)


reward_table = build_reward_table()


clean_detector = (
    load_clean_detector()
)

obvious_detector = (
    load_detector_results(
        str(OBVIOUS_DIR)
    )
)

stealthy_detector = (
    load_detector_results(
        str(STEALTHY_DIR)
    )
)


def detector_row(
    dataframe,
    trajectory_id,
):
    if dataframe is None:
        return None

    match = dataframe[
        dataframe["trajectory_id"]
        == trajectory_id
    ]

    if match.empty:
        return None

    return match.iloc[0]


clean_detection = detector_row(
    clean_detector,
    trajectory_id,
)

obvious_detection = detector_row(
    obvious_detector,
    trajectory_id,
)

stealthy_detection = detector_row(
    stealthy_detector,
    trajectory_id,
)


obvious_metrics = (
    load_detector_metrics(
        str(OBVIOUS_DIR)
    )
)

stealthy_metrics = (
    load_detector_metrics(
        str(STEALTHY_DIR)
    )
)


st.info(
    "The position, speed, lane changes, actions and traffic "
    "are identical across Clean, Obvious Poison and Stealthy Poison. "
    "Only reward values differ."
)


if (
    obvious_metrics is not None
    and stealthy_metrics is not None
):
    m1, m2 = st.columns(2)

    obvious_recall = (
        obvious_metrics[
            "global_detector"
        ]["recall"]
    )

    stealthy_recall = (
        stealthy_metrics[
            "global_detector"
        ]["recall"]
    )

    m1.metric(
        "Obvious-poison detector recall",
        f"{100 * obvious_recall:.1f}%",
    )

    m2.metric(
        "Stealthy-poison detector recall",
        f"{100 * stealthy_recall:.1f}%",
    )


def detector_summary():
    rows = []

    if clean_detection is not None:
        rows.append(
            {
                "Version": "Clean",
                "Ground truth": "Clean",
                "Mean reward":
                    clean_record[
                        "summary"
                    ]["mean_reward"],
                "Anomaly score":
                    clean_detection[
                        "global_anomaly_score"
                    ],
                "Detector":
                    (
                        "Suspicious"
                        if bool(
                            clean_detection[
                                "global_flag"
                            ]
                        )
                        else "Normal"
                    ),
                "Dominant feature":
                    clean_detection[
                        "global_dominant_feature"
                    ],
            }
        )

    if obvious_detection is not None:
        rows.append(
            {
                "Version":
                    "Obvious poison",

                "Ground truth":
                    "Poisoned",

                "Mean reward":
                    obvious_record[
                        "summary"
                    ]["mean_reward"],

                "Anomaly score":
                    obvious_detection[
                        "global_anomaly_score"
                    ],

                "Detector":
                    (
                        "Suspicious"
                        if bool(
                            obvious_detection[
                                "global_flag"
                            ]
                        )
                        else "Normal"
                    ),

                "Dominant feature":
                    obvious_detection[
                        "global_dominant_feature"
                    ],
            }
        )

    if stealthy_detection is not None:
        rows.append(
            {
                "Version":
                    "Stealthy poison",

                "Ground truth":
                    "Poisoned",

                "Mean reward":
                    stealthy_record[
                        "summary"
                    ]["mean_reward"],

                "Anomaly score":
                    stealthy_detection[
                        "global_anomaly_score"
                    ],

                "Detector":
                    (
                        "Suspicious"
                        if bool(
                            stealthy_detection[
                                "global_flag"
                            ]
                        )
                        else "Normal"
                    ),

                "Dominant feature":
                    stealthy_detection[
                        "global_dominant_feature"
                    ],
            }
        )

    return pd.DataFrame(rows)


st.subheader(
    "Same trajectory, different reward data"
)

detector_table = detector_summary()

st.dataframe(
    detector_table.style.format(
        {
            "Mean reward": "{:.3f}",
            "Anomaly score": "{:.3f}",
        }
    ),
    use_container_width=True,
)


left, right = st.columns(
    [2.0, 1.2],
    gap="large",
)


with left:
    st.subheader(
        "Driving simulation"
    )

    image_placeholder = st.empty()

    controls = st.columns(
        [1, 1, 2]
    )

    play = controls[0].button(
        "Play",
        use_container_width=True,
    )

    speed_label = (
        controls[1].selectbox(
            "Playback",
            [
                "1x",
                "2x",
                "4x",
            ],
            index=1,
        )
    )

    delays = {
        "1x": 0.20,
        "2x": 0.10,
        "4x": 0.05,
    }

    manual_step = (
        controls[2].slider(
            "Step",
            min_value=1,
            max_value=len(frames),
            value=1,
        )
        - 1
    )


with right:
    st.subheader(
        "Current transition"
    )

    live_placeholder = st.empty()


def draw_step(index: int):
    image_placeholder.image(
        frames[index],
        caption=(
            f"{trajectory_id} — "
            f"step {index + 1}/"
            f"{len(frames)}"
        ),
        use_container_width=True,
    )

    row = reward_table.iloc[
        index
    ]

    with live_placeholder.container():

        st.metric(
            "Action",
            row["Action"],
        )

        a, b = st.columns(2)

        a.metric(
            "Speed",
            f"{row['Speed (m/s)']:.2f} m/s",
        )

        b.metric(
            "TTC",
            f"{row['TTC (s)']:.2f} s",
        )

        st.markdown(
            "**Stored reward**"
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Clean",
            f"{row['Clean reward']:.3f}",
        )

        r2.metric(
            "Obvious",
            f"{row['Obvious reward']:.3f}",
        )

        r3.metric(
            "Stealthy",
            f"{row['Stealthy reward']:.3f}",
        )

        if row[
            "Reward modified"
        ]:
            st.warning(
                "This transition was poisoned."
            )
        else:
            st.success(
                "Reward unchanged on this transition."
            )


if play:
    for index in range(
        len(frames)
    ):
        draw_step(index)

        time.sleep(
            delays[speed_label]
        )
else:
    draw_step(
        manual_step
    )


st.divider()

st.subheader(
    "Reward signal over the drive"
)

reward_chart = (
    reward_table[
        [
            "Step",
            "Clean reward",
            "Obvious reward",
            "Stealthy reward",
        ]
    ]
    .set_index("Step")
)

st.line_chart(
    reward_chart
)


with st.expander(
    "Inspect every transition"
):
    st.dataframe(
        reward_table,
        use_container_width=True,
    )