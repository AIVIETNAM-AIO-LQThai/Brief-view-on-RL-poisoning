from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import STYLE_CONFIG


@dataclass
class LaneGaps:
    front: float
    rear: float


def _lane_gaps(env, lane_id: int) -> LaneGaps:
    """Return front/rear gaps on a candidate lane.

    highway-v0 is a straight road, so x-coordinate differences are sufficient
    and easy to explain in the paper.
    """
    u = env.unwrapped
    ego = u.vehicle
    start, end, _ = ego.lane_index
    candidate = (start, end, int(lane_id))

    try:
        front, rear = u.road.neighbour_vehicles(ego, candidate)
    except Exception:
        return LaneGaps(front=-np.inf, rear=-np.inf)

    ex = float(ego.position[0])
    front_gap = np.inf if front is None else max(0.0, float(front.position[0]) - ex)
    rear_gap = np.inf if rear is None else max(0.0, ex - float(rear.position[0]))
    return LaneGaps(front=front_gap, rear=rear_gap)


def _safe_lane_candidates(env, cfg: dict) -> list[int]:
    ego = env.unwrapped.vehicle
    lane_id = int(ego.lane_index[2])
    lanes_count = int(env.unwrapped.config["lanes_count"])
    candidates = []

    for cand in (lane_id - 1, lane_id + 1):
        if not (0 <= cand < lanes_count):
            continue
        gaps = _lane_gaps(env, cand)
        if (
            gaps.front >= cfg["candidate_front_gap"]
            and gaps.rear >= cfg["candidate_rear_gap"]
        ):
            candidates.append(cand)
    return candidates


def _current_front_gap(env) -> float:
    u = env.unwrapped
    ego = u.vehicle
    front, _ = u.road.neighbour_vehicles(ego, ego.lane_index)
    if front is None:
        return np.inf
    return max(0.0, float(front.position[0]) - float(ego.position[0]))


def choose_action(env, style: str, rng: np.random.Generator) -> int:
    """A transparent heuristic controller for three legitimate driving styles.

    This is intentionally not a learned RL agent yet. The purpose of Stage 1
    is to create heterogeneous *clean trajectories* whose origin we fully know.
    """
    if style not in STYLE_CONFIG:
        raise ValueError(f"Unknown style: {style}")

    cfg = STYLE_CONFIG[style]
    u = env.unwrapped
    ego = u.vehicle
    actions = u.action_type.actions_indexes

    # Tiny amount of benign randomness creates within-style variation.
    if rng.random() < cfg["random_action_probability"]:
        available = list(u.get_available_actions())
        return int(rng.choice(available))

    speed = float(ego.speed)
    front_gap = _current_front_gap(env)

    # If blocked, consider a safe overtake before changing speed.
    if front_gap < cfg["front_gap_trigger"]:
        candidates = _safe_lane_candidates(env, cfg)
        if candidates and rng.random() < cfg["lane_change_probability"]:
            current_lane = int(ego.lane_index[2])

            # Prefer the lane with the larger front gap.
            best = max(candidates, key=lambda lid: _lane_gaps(env, lid).front)
            if best < current_lane and "LANE_LEFT" in actions:
                return int(actions["LANE_LEFT"])
            if best > current_lane and "LANE_RIGHT" in actions:
                return int(actions["LANE_RIGHT"])

        if speed > max(18.0, cfg["target_speed"] - 2.0):
            return int(actions["SLOWER"])

    # Speed regulation gives the styles their main behavioural distinction.
    if speed < cfg["target_speed"] - 1.0:
        return int(actions["FASTER"])
    if speed > cfg["target_speed"] + 1.5:
        return int(actions["SLOWER"])

    return int(actions["IDLE"])
