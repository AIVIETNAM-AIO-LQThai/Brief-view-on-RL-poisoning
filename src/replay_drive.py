from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import pandas as pd

from .experiment_config import ENV_CONFIG, ENV_ID


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "clean_drives"


STYLE_EXPLANATION = {
    "cautious": "Drives slower, changes lanes rarely, and tends to keep larger safety margins.",
    "normal": "Maintains balanced speed and occasionally overtakes when there is enough room.",
    "aggressive": "Drives faster, overtakes more often, and accepts smaller safety margins.",
}


def load_trajectory_records(results_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    results_dir = results_dir or DEFAULT_RESULTS
    path = results_dir / "trajectories.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run python -m src.collect_clean_drives first."
        )

    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records[record["trajectory_id"]] = record
    return records


def load_demo_cases(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or ROOT
    path = root / "demo" / "demo_cases.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run python -m src.choose_demo_cases first."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_record(trajectory_id: str, results_dir: Path | None = None) -> dict[str, Any]:
    return load_trajectory_records(results_dir)[trajectory_id]


def safety_band(ttc: float | None, front_gap: float | None) -> str:
    if ttc is None or front_gap is None:
        return "Unknown"
    if not np.isfinite(ttc) or not np.isfinite(front_gap):
        return "Comfortable"
    if ttc < 1.5 or front_gap < 6:
        return "Critical"
    if ttc < 2.5 or front_gap < 10:
        return "Tight"
    return "Comfortable"


def format_live_step(transition: dict[str, Any], style: str, data_label: str) -> dict[str, Any]:
    after = transition["ego_after"]
    return {
        "step": int(transition["t"]) + 1,
        "action": transition["action_label"],
        "style": style.capitalize(),
        "data_label": data_label.capitalize(),
        "speed_mps": float(after["speed"]),
        "lane": int(after["lane_id"]),
        "front_gap_m": float(after["front_gap"]),
        "ttc_s": float(after["ttc"]),
        "crashed": bool(after["crashed"]),
        "safety_band": safety_band(after["ttc"], after["front_gap"]),
    }


def render_replay(record: dict[str, Any], render_mode: str = "rgb_array"):
    env_id = record.get("environment_id", ENV_ID)
    env_config = record.get("environment_config", ENV_CONFIG)

    env = gym.make(env_id, config=env_config, render_mode=render_mode)
    try:
        env.reset(seed=int(record["env_seed"]))
        frames = []
        steps = []

        style = record["style"]
        data_label = record.get("data_label", "clean")

        for transition in record["transitions"]:
            action = int(transition["action"])
            env.step(action)
            frame = env.render()
            frames.append(np.asarray(frame))
            steps.append(format_live_step(transition, style, data_label))

        return frames, steps
    finally:
        env.close()


def summary_table(record: dict[str, Any]) -> dict[str, Any]:
    s = record["summary"]
    return {
        "Trajectory ID": record["trajectory_id"],
        "Driving style": record["style"].capitalize(),
        "Data label": record.get("data_label", "clean").capitalize(),
        "Crashed": "Yes" if s["crashed"] else "No",
        "Steps": int(s["steps"]),
        "Return": round(float(s["return"]), 3),
        "Mean speed (m/s)": round(float(s["mean_speed"]), 3),
        "Max speed (m/s)": round(float(s["max_speed"]), 3),
        "Lane changes": int(s["lane_changes"]),
        "Min front gap (m)": round(float(s["min_front_gap"]), 3),
        "Median front gap (m)": round(float(s["median_front_gap"]), 3),
        "Min TTC (s)": round(float(s["min_ttc"]), 3),
        "Action entropy": round(float(s["action_entropy_bits"]), 3),
    }


def step_dataframe(steps):
    df = pd.DataFrame(steps)
    if df.empty:
        return df
    df = df.rename(
        columns={
            "step": "Step",
            "action": "Action",
            "speed_mps": "Speed (m/s)",
            "lane": "Lane",
            "front_gap_m": "Front gap (m)",
            "ttc_s": "TTC (s)",
            "crashed": "Crashed",
            "safety_band": "Safety band",
        }
    )
    return df


def audience_narrative(record: dict[str, Any]) -> str:
    s = record["summary"]
    style = record["style"]
    explanation = STYLE_EXPLANATION.get(style, "")

    parts = [explanation]

    if s["crashed"]:
        parts.append("This drive ends in a collision, but it is still clean data.")
    else:
        parts.append("This drive finishes without a collision.")

    if s["lane_changes"] <= 1:
        parts.append("The vehicle stays mostly in its lane.")
    elif s["lane_changes"] <= 3:
        parts.append("The vehicle changes lanes occasionally.")
    else:
        parts.append("The vehicle changes lanes frequently.")

    if float(s["min_ttc"]) < 1.5:
        parts.append("At some point the time-to-collision becomes critically small.")
    elif float(s["min_ttc"]) < 2.5:
        parts.append("At some point the time-to-collision becomes tight.")
    else:
        parts.append("The vehicle mostly keeps comfortable longitudinal margins.")

    return " ".join(parts)
