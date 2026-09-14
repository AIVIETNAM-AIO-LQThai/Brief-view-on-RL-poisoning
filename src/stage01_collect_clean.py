from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401  -- registers the environments
import numpy as np
import pandas as pd

from .config import (
    BASE_SEED,
    ENV_CONFIG,
    ENV_ID,
    EPISODES_PER_STYLE,
    STYLE_ORDER,
)
from .policies import choose_action

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ACTION_FALLBACK = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def entropy_bits(values: list[int]) -> float:
    if not values:
        return 0.0
    counts = np.array(list(Counter(values).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def min_distance_to_other_vehicle(env) -> float:
    u = env.unwrapped
    ego = u.vehicle
    distances = []
    for vehicle in u.road.vehicles:
        if vehicle is ego:
            continue
        distances.append(float(np.linalg.norm(vehicle.position - ego.position)))
    return min(distances) if distances else math.inf


def current_front_gap(env) -> float:
    u = env.unwrapped
    ego = u.vehicle
    front, _ = u.road.neighbour_vehicles(ego, ego.lane_index)
    if front is None:
        return math.inf
    return max(0.0, float(front.position[0]) - float(ego.position[0]))


def action_label(env, action: int) -> str:
    mapping = getattr(env.unwrapped.action_type, "actions", None)
    if isinstance(mapping, dict) and action in mapping:
        return str(mapping[action])
    reverse = {
        int(v): str(k)
        for k, v in env.unwrapped.action_type.actions_indexes.items()
    }
    return reverse.get(int(action), ACTION_FALLBACK.get(int(action), str(action)))


def collect_episode(style: str, episode_index: int, env_seed: int, policy_seed: int):
    rng = np.random.default_rng(policy_seed)
    env = gym.make(
        ENV_ID,
        config=ENV_CONFIG,
        render_mode="rgb_array",
    )

    obs, info = env.reset(seed=env_seed)
    transitions = []
    actions = []
    speeds = []
    min_distances = []
    lane_ids = []
    rewards = []

    terminated = False
    truncated = False
    t = 0

    while not (terminated or truncated):
        u = env.unwrapped
        ego = u.vehicle

        before = {
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "speed": float(ego.speed),
            "heading": float(ego.heading),
            "lane_id": int(ego.lane_index[2]),
            "crashed": bool(ego.crashed),
            "front_gap": float(current_front_gap(env)),
            "min_vehicle_distance": float(min_distance_to_other_vehicle(env)),
        }

        action = choose_action(env, style, rng)
        label = action_label(env, action)
        obs_next, reward, terminated, truncated, info = env.step(action)

        ego_after = env.unwrapped.vehicle
        after = {
            "x": float(ego_after.position[0]),
            "y": float(ego_after.position[1]),
            "speed": float(ego_after.speed),
            "heading": float(ego_after.heading),
            "lane_id": int(ego_after.lane_index[2]),
            "crashed": bool(ego_after.crashed),
            "front_gap": float(current_front_gap(env)),
            "min_vehicle_distance": float(min_distance_to_other_vehicle(env)),
        }

        transitions.append(
            {
                "t": t,
                "action": int(action),
                "action_label": label,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "ego_before": before,
                "ego_after": after,
                # Kinematic observation: rows are ego + nearby vehicles.
                "observation": np.asarray(obs, dtype=float).tolist(),
                "next_observation": np.asarray(obs_next, dtype=float).tolist(),
            }
        )

        actions.append(int(action))
        speeds.append(after["speed"])
        min_distances.append(after["min_vehicle_distance"])
        lane_ids.append(after["lane_id"])
        rewards.append(float(reward))

        obs = obs_next
        t += 1

    crashed = bool(env.unwrapped.vehicle.crashed)
    lane_changes = sum(
        1 for a, b in zip(lane_ids, lane_ids[1:]) if a != b
    )

    summary = {
        "trajectory_id": f"{style}_{episode_index:03d}",
        "style": style,
        "episode_index": episode_index,
        "env_seed": env_seed,
        "policy_seed": policy_seed,
        "steps": len(transitions),
        "return": float(np.sum(rewards)),
        "crashed": crashed,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "max_speed": float(np.max(speeds)) if speeds else 0.0,
        "min_vehicle_distance": float(np.min(min_distances)) if min_distances else math.inf,
        "mean_vehicle_distance": float(np.mean(min_distances)) if min_distances else math.inf,
        "lane_changes": int(lane_changes),
        "action_entropy_bits": entropy_bits(actions),
        "actions": actions,
    }

    record = {
        "trajectory_id": summary["trajectory_id"],
        "style": style,
        "env_seed": env_seed,
        "policy_seed": policy_seed,
        "summary": summary,
        "transitions": transitions,
    }

    env.close()
    return summary, record


def select_representatives(df: pd.DataFrame) -> list[dict]:
    """Choose a median-return clean episode for each style.

    These episode seeds + action sequences become stable demo candidates.
    """
    reps = []
    for style in STYLE_ORDER:
        part = df[df["style"] == style].sort_values("return").reset_index(drop=True)
        row = part.iloc[len(part) // 2]
        reps.append(row.to_dict())
    return reps


def safe_json_value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def main():
    summaries = []
    records = []

    for style_index, style in enumerate(STYLE_ORDER):
        for episode_index in range(EPISODES_PER_STYLE):
            env_seed = BASE_SEED + style_index * 10_000 + episode_index
            policy_seed = BASE_SEED + 100_000 + style_index * 10_000 + episode_index
            summary, record = collect_episode(
                style=style,
                episode_index=episode_index,
                env_seed=env_seed,
                policy_seed=policy_seed,
            )
            summaries.append(summary)
            records.append(record)
            print(
                f"{summary['trajectory_id']:>18} | "
                f"R={summary['return']:.3f} | "
                f"speed={summary['mean_speed']:.2f} | "
                f"changes={summary['lane_changes']} | "
                f"crash={summary['crashed']}"
            )

    df = pd.DataFrame(summaries)
    df.to_csv(OUT / "clean_summary.csv", index=False)

    with (OUT / "clean_trajectories.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=True) + "\n")

    by_style = (
        df.groupby("style", sort=False)
        .agg(
            n=("trajectory_id", "count"),
            crash_rate=("crashed", "mean"),
            return_mean=("return", "mean"),
            return_std=("return", "std"),
            speed_mean=("mean_speed", "mean"),
            speed_std=("mean_speed", "std"),
            lane_changes_mean=("lane_changes", "mean"),
            min_distance_median=("min_vehicle_distance", "median"),
            action_entropy_mean=("action_entropy_bits", "mean"),
        )
        .reset_index()
    )

    metrics = {
        "stage": 1,
        "status": "complete",
        "environment": ENV_ID,
        "base_seed": BASE_SEED,
        "episodes_per_style": EPISODES_PER_STYLE,
        "n_trajectories": int(len(df)),
        "overall": {
            "crash_rate": float(df["crashed"].mean()),
            "return_mean": float(df["return"].mean()),
            "return_std": float(df["return"].std(ddof=1)),
            "speed_mean": float(df["mean_speed"].mean()),
            "lane_changes_mean": float(df["lane_changes"].mean()),
        },
        "by_style": [
            {k: safe_json_value(v) for k, v in row.items()}
            for row in by_style.to_dict(orient="records")
        ],
    }

    with (OUT / "stage01_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)

    representatives = select_representatives(df)
    with (OUT / "stage01_representatives.json").open("w", encoding="utf-8") as f:
        json.dump(
            [{k: safe_json_value(v) for k, v in r.items()} for r in representatives],
            f,
            indent=2,
            allow_nan=True,
        )

    checkpoint = f"""# Scientific Thinking Highway Project — Checkpoint

## Current stage
**Stage 1 — Clean heterogeneous trajectory baseline COMPLETE**

## Scientific question
How much variation exists among trajectories that are all legitimate and unpoisoned?

## Fixed environment
- Environment: `{ENV_ID}`
- Base seed: `{BASE_SEED}`
- Driving styles: cautious, normal, aggressive
- Episodes per style: `{EPISODES_PER_STYLE}`
- Total clean trajectories: {len(df)}

## Key result
This stage intentionally contains no attack. Any extreme trajectory is still clean.

### Overall
- Crash rate: {metrics['overall']['crash_rate']:.4f}
- Mean return: {metrics['overall']['return_mean']:.4f}
- Return std: {metrics['overall']['return_std']:.4f}
- Mean speed: {metrics['overall']['speed_mean']:.4f} m/s
- Mean lane changes: {metrics['overall']['lane_changes_mean']:.4f}

## Next stage
**Stage 2 — Naive anomaly inspection on CLEAN data only.**

We will compute simple trajectory-level anomaly scores from:
- return
- mean speed
- maximum speed
- lane changes
- minimum vehicle distance
- action entropy

The detector will rank suspicious trajectories even though every trajectory is clean.
That false-positive behaviour is the first deliberate failure in the scientific-thinking narrative.
"""
    (ROOT / "CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    project_state = {
        "project": "Scientific Thinking — Highway Trajectory Poisoning",
        "current_stage": 1,
        "stage_name": "clean_heterogeneous_baseline",
        "status": "complete",
        "next_stage": 2,
        "next_stage_name": "naive_clean_anomaly_inspection",
        "environment": ENV_ID,
        "base_seed": BASE_SEED,
        "representative_trajectory_ids": [
            r["trajectory_id"] for r in representatives
        ],
    }
    (ROOT / "project_state.json").write_text(
        json.dumps(project_state, indent=2),
        encoding="utf-8",
    )

    print("\n=== BY STYLE ===")
    print(by_style.round(4).to_string(index=False))
    print("\nCheckpoint written to CHECKPOINT.md")


if __name__ == "__main__":
    main()
