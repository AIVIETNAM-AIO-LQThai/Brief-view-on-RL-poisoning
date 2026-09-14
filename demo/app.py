from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay_drive import (  # noqa: E402
    audience_narrative,
    get_record,
    load_demo_cases,
    render_replay,
    step_dataframe,
    summary_table,
)

st.set_page_config(page_title="Highway Trajectory Demo", layout="wide")
st.title("Highway Trajectory Demo")
st.caption("A visual replay of clean driving trajectories, with live statistics that match the on-screen behavior.")


@st.cache_data(show_spinner=False)
def cached_demo_cases():
    return load_demo_cases()


@st.cache_data(show_spinner=False)
def cached_record(trajectory_id: str):
    return get_record(trajectory_id)


@st.cache_data(show_spinner=False)
def cached_replay(trajectory_id: str):
    record = cached_record(trajectory_id)
    return render_replay(record)


def metric_value(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


try:
    cases = cached_demo_cases()
except FileNotFoundError:
    st.error("No demo cases found. Run: python -m src.choose_demo_cases")
    st.stop()

case_map = {case["title"]: case for case in cases}
selected_title = st.selectbox("Select a trajectory to replay", list(case_map.keys()))
selected_case = case_map[selected_title]

trajectory_id = selected_case["trajectory_id"]
record = cached_record(trajectory_id)
frames, live_steps = cached_replay(trajectory_id)
summary = summary_table(record)
step_df = step_dataframe(live_steps)

left, right = st.columns([2.1, 1.2], gap="large")

with left:
    st.subheader(selected_case["title"])
    st.write(selected_case["subtitle"])
    st.info(selected_case["why_selected"])

    image_placeholder = st.empty()

    controls = st.columns([1, 1, 2])
    play = controls[0].button("Play replay", use_container_width=True)
    speed_label = controls[1].selectbox("Playback speed", ["0.25x", "0.5x", "1x", "2x"], index=2)
    speed_map = {"0.25x": 0.20, "0.5x": 0.12, "1x": 0.06, "2x": 0.03}
    delay = speed_map[speed_label]

    max_step = max(1, len(frames))
    step_index = controls[2].slider("Manual step", min_value=1, max_value=max_step, value=1) - 1

with right:
    st.subheader("Trajectory explanation")
    st.write(audience_narrative(record))

    badge_cols = st.columns(2)
    badge_cols[0].metric("Style", selected_case["style"].capitalize())
    badge_cols[1].metric("Data label", selected_case["data_label"].capitalize())

    live_box = st.container(border=True)
    summary_box = st.container(border=True)

def draw_live(index: int):
    image_placeholder.image(frames[index], caption=f"Step {index + 1}/{len(frames)}", use_container_width=True)

    step = live_steps[index]
    with live_box:
        st.markdown("**Live statistics**")
        a, b = st.columns(2)
        c, d = st.columns(2)
        e, f = st.columns(2)

        a.metric("Action", step["action"])
        b.metric("Lane", step["lane"])
        c.metric("Speed (m/s)", metric_value(step["speed_mps"], 2))
        d.metric("Front gap (m)", metric_value(step["front_gap_m"], 2))
        e.metric("TTC (s)", metric_value(step["ttc_s"], 2))
        f.metric("Safety band", step["safety_band"])

        st.metric("Crash status", "Crashed" if step["crashed"] else "Safe so far")

with summary_box:
    st.markdown("**Trajectory summary**")
    s1, s2 = st.columns(2)
    s3, s4 = st.columns(2)
    s5, s6 = st.columns(2)

    s1.metric("Crashed", summary["Crashed"])
    s2.metric("Steps", summary["Steps"])
    s3.metric("Mean speed (m/s)", metric_value(summary["Mean speed (m/s)"], 2))
    s4.metric("Lane changes", summary["Lane changes"])
    s5.metric("Min front gap (m)", metric_value(summary["Min front gap (m)"], 2))
    s6.metric("Min TTC (s)", metric_value(summary["Min TTC (s)"], 2))
    st.metric("Action entropy", metric_value(summary["Action entropy"], 3))
    st.metric("Return", metric_value(summary["Return"], 3))

if play:
    for i in range(len(frames)):
        draw_live(i)
        time.sleep(delay)
else:
    draw_live(step_index)

st.divider()
st.subheader("How the statistics evolve over time")
chart_df = step_df[["Step", "Speed (m/s)", "Front gap (m)", "TTC (s)"]].set_index("Step")
st.line_chart(chart_df)

with st.expander("Step-by-step table"):
    st.dataframe(step_df, use_container_width=True)
