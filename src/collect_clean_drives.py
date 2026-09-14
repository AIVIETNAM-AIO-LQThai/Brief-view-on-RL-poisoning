from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import pandas as pd

from .experiment_config import (
    ACCEPTANCE,
    BASE_SEED,
    ENV_CONFIG,
    ENV_ID,
    EPISODES_PER_STYLE,
    STYLE_ORDER,
)
from .driving_policy import choose_action

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "clean_drives"
OUT.mkdir(parents=True, exist_ok=True)


def entropy_bits(values):
    if not values:
        return 0.0
    counts = np.array(list(Counter(values).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def front_metrics(env):
    u = env.unwrapped
    ego = u.vehicle
    front, _ = u.road.neighbour_vehicles(ego, ego.lane_index)
    if front is None:
        return math.inf, math.inf

    gap = max(0.0, float(front.position[0]) - float(ego.position[0]))
    closing_speed = float(ego.speed) - float(front.speed)
    ttc = math.inf if closing_speed <= 1e-9 else gap / closing_speed
    return gap, ttc


def min_euclidean_vehicle_distance(env):
    u = env.unwrapped
    ego = u.vehicle
    ds = [
        float(np.linalg.norm(v.position - ego.position))
        for v in u.road.vehicles
        if v is not ego
    ]
    return min(ds) if ds else math.inf


def action_label(env, action):
    reverse = {
        int(v): str(k)
        for k, v in env.unwrapped.action_type.actions_indexes.items()
    }
    return reverse.get(int(action), str(action))


def collect_episode(style, episode_index, env_seed, policy_seed):
    rng = np.random.default_rng(policy_seed)
    env = gym.make(ENV_ID, config=ENV_CONFIG, render_mode="rgb_array")
    obs, info = env.reset(seed=env_seed)

    transitions = []
    rewards, speeds, lane_ids, actions = [], [], [], []
    front_gaps, ttcs, euclidean_mins = [], [], []

    terminated = truncated = False
    t = 0

    while not (terminated or truncated):
        ego = env.unwrapped.vehicle
        gap_before, ttc_before = front_metrics(env)

        before = {
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "speed": float(ego.speed),
            "lane_id": int(ego.lane_index[2]),
            "front_gap": float(gap_before),
            "ttc": float(ttc_before),
            "crashed": bool(ego.crashed),
        }

        action = choose_action(env, style, rng)
        obs_next, reward, terminated, truncated, info = env.step(action)

        ego_after = env.unwrapped.vehicle
        gap_after, ttc_after = front_metrics(env)
        min_euclid = min_euclidean_vehicle_distance(env)

        after = {
            "x": float(ego_after.position[0]),
            "y": float(ego_after.position[1]),
            "speed": float(ego_after.speed),
            "lane_id": int(ego_after.lane_index[2]),
            "front_gap": float(gap_after),
            "ttc": float(ttc_after),
            "crashed": bool(ego_after.crashed),
        }

        transitions.append({
            "t": t,
            "action": int(action),
            "action_label": action_label(env, action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "ego_before": before,
            "ego_after": after,
            "observation": np.asarray(obs, dtype=float).tolist(),
            "next_observation": np.asarray(obs_next, dtype=float).tolist(),
        })

        rewards.append(float(reward))
        speeds.append(float(ego_after.speed))
        lane_ids.append(int(ego_after.lane_index[2]))
        actions.append(int(action))
        front_gaps.append(float(gap_after))
        ttcs.append(float(ttc_after))
        euclidean_mins.append(float(min_euclid))

        obs = obs_next
        t += 1

    finite_gaps = [x for x in front_gaps if np.isfinite(x)]
    finite_ttcs = [x for x in ttcs if np.isfinite(x)]

    lane_changes = sum(a != b for a, b in zip(lane_ids, lane_ids[1:]))

    summary = {
        "trajectory_id": f"{style}_{episode_index:03d}",
        "style": style,
        "episode_index": episode_index,
        "env_seed": env_seed,
        "policy_seed": policy_seed,
        "steps": len(transitions),
        "return": float(np.sum(rewards)),
        "crashed": bool(env.unwrapped.vehicle.crashed),
        "mean_speed": float(np.mean(speeds)),
        "max_speed": float(np.max(speeds)),
        "lane_changes": int(lane_changes),
        "action_entropy_bits": entropy_bits(actions),
        "min_front_gap": float(min(finite_gaps)) if finite_gaps else math.inf,
        "median_front_gap": float(np.median(finite_gaps)) if finite_gaps else math.inf,
        "min_ttc": float(min(finite_ttcs)) if finite_ttcs else math.inf,
        "min_euclidean_vehicle_distance": float(min(euclidean_mins)),
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


def acceptance_report(by_style):
    rows = {r["style"]: r for r in by_style.to_dict(orient="records")}
    checks = {}

    for style, limit in ACCEPTANCE["max_crash_rate"].items():
        actual = float(rows[style]["crash_rate"])
        checks[f"{style}_crash_rate"] = {
            "pass": actual <= limit,
            "actual": actual,
            "limit": limit,
        }

    speeds = [float(rows[s]["speed_mean"]) for s in STYLE_ORDER]
    min_gap = float(ACCEPTANCE["min_speed_gap_ms"])
    checks["speed_separation"] = {
        "pass": all((b - a) >= min_gap for a, b in zip(speeds, speeds[1:])),
        "means": dict(zip(STYLE_ORDER, speeds)),
        "minimum_required_gap": min_gap,
    }

    changes = [float(rows[s]["lane_changes_mean"]) for s in STYLE_ORDER]
    checks["lane_change_order"] = {
        "pass": changes[0] < changes[1] < changes[2],
        "means": dict(zip(STYLE_ORDER, changes)),
    }

    return {
        "pass": all(v["pass"] for v in checks.values()),
        "checks": checks,
    }


def main():
    summaries, records = [], []

    for style_index, style in enumerate(STYLE_ORDER):
        for episode_index in range(EPISODES_PER_STYLE):
            env_seed = BASE_SEED + style_index * 10_000 + episode_index
            policy_seed = BASE_SEED + 100_000 + style_index * 10_000 + episode_index

            s, r = collect_episode(
                style, episode_index, env_seed, policy_seed
            )
            summaries.append(s)
            records.append(r)

            print(
                f"{s['trajectory_id']:>18} | "
                f"R={s['return']:.3f} | "
                f"speed={s['mean_speed']:.2f} | "
                f"changes={s['lane_changes']} | "
                f"minTTC={s['min_ttc']:.2f} | "
                f"crash={s['crashed']}"
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
            min_front_gap_median=("min_front_gap", "median"),
            min_ttc_median=("min_ttc", "median"),
            action_entropy_mean=("action_entropy_bits", "mean"),
        )
        .reset_index()
    )

    report = acceptance_report(by_style)

    result = {
        "exp": "clean_driving_baseline",
        "status": "accepted" if report["pass"] else "needs_revision",
        "n_trajectories": int(len(df)),
        "by_style": by_style.to_dict(orient="records"),
        "acceptance": report,
    }

    with (OUT / "stage01b_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    print("\n=== STAGE 1B BY STYLE ===")
    print(by_style.round(4).to_string(index=False))

    print("\n=== ACCEPTANCE ===")
    print("PASS" if report["pass"] else "NEEDS TUNING")
    for name, detail in report["checks"].items():
        print(f"{name}: {'PASS' if detail['pass'] else 'FAIL'}")

    checkpoint = f"""# Checkpoint 1B — Tuned clean baseline

Status: **{'ACCEPTED' if report['pass'] else 'NEEDS MORE TUNING'}**

No poisoning was present.

The full measured metrics are saved in:
`outputs/stage01b_tuned/stage01b_metrics.json`

Do not proceed to Stage 2 unless the acceptance result is PASS.
"""
    (ROOT / "CHECKPOINT_STAGE01B.md").write_text(checkpoint, encoding="utf-8")


if __name__ == "__main__":
    main()
