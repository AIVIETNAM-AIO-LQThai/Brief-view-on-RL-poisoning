from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .experiment_config import STYLE_CONFIG


@dataclass
class LaneSafety:
    front_gap: float
    rear_gap: float
    front_speed: float | None
    rear_speed: float | None


def _lane_safety(env, lane_id: int) -> LaneSafety:
    u = env.unwrapped
    ego = u.vehicle
    start, end, _ = ego.lane_index
    candidate = (start, end, int(lane_id))
    front, rear = u.road.neighbour_vehicles(ego, candidate)

    ex = float(ego.position[0])
    front_gap = np.inf if front is None else max(0.0, float(front.position[0]) - ex)
    rear_gap = np.inf if rear is None else max(0.0, ex - float(rear.position[0]))

    return LaneSafety(
        front_gap=front_gap,
        rear_gap=rear_gap,
        front_speed=None if front is None else float(front.speed),
        rear_speed=None if rear is None else float(rear.speed),
    )


def _candidate_is_safe(env, lane_id: int, cfg: dict) -> bool:
    ego_speed = float(env.unwrapped.vehicle.speed)
    lane = _lane_safety(env, lane_id)

    # If ego is closing quickly on the candidate-lane front car, require
    # additional front space. If the rear car is closing quickly on ego,
    # require additional rear space.
    front_closing = 0.0 if lane.front_speed is None else max(0.0, ego_speed - lane.front_speed)
    rear_closing = 0.0 if lane.rear_speed is None else max(0.0, lane.rear_speed - ego_speed)

    required_front = cfg["base_front_gap"] + 1.5 * front_closing
    required_rear = cfg["base_rear_gap"] + 2.0 * rear_closing

    return lane.front_gap >= required_front and lane.rear_gap >= required_rear


def _safe_lane_candidates(env, cfg: dict) -> list[int]:
    ego = env.unwrapped.vehicle
    lane_id = int(ego.lane_index[2])
    lanes_count = int(env.unwrapped.config["lanes_count"])

    return [
        cand
        for cand in (lane_id - 1, lane_id + 1)
        if 0 <= cand < lanes_count and _candidate_is_safe(env, cand, cfg)
    ]


def _current_front(env):
    u = env.unwrapped
    ego = u.vehicle
    front, _ = u.road.neighbour_vehicles(ego, ego.lane_index)
    if front is None:
        return np.inf, None
    gap = max(0.0, float(front.position[0]) - float(ego.position[0]))
    return gap, front


def choose_action(env, style: str, rng: np.random.Generator) -> int:
    cfg = STYLE_CONFIG[style]
    u = env.unwrapped
    ego = u.vehicle
    actions = u.action_type.actions_indexes
    available = set(int(x) for x in u.get_available_actions())

    # Small benign within-style randomness.
    if rng.random() < cfg["random_action_probability"]:
        return int(rng.choice(list(available)))

    speed = float(ego.speed)
    front_gap, front_vehicle = _current_front(env)

    if front_gap < cfg["front_gap_trigger"]:
        candidates = _safe_lane_candidates(env, cfg)

        if candidates and rng.random() < cfg["lane_change_probability"]:
            current_lane = int(ego.lane_index[2])
            best = max(candidates, key=lambda lid: _lane_safety(env, lid).front_gap)

            if best < current_lane:
                a = int(actions["LANE_LEFT"])
                if a in available:
                    return a
            elif best > current_lane:
                a = int(actions["LANE_RIGHT"])
                if a in available:
                    return a

        # If we cannot overtake safely, prioritise braking when closing on
        # traffic instead of forcing a risky lane change.
        closing = (
            front_vehicle is not None
            and speed > float(front_vehicle.speed)
        )
        if closing or front_gap < 12.0:
            a = int(actions["SLOWER"])
            if a in available:
                return a

    if speed < cfg["target_speed"] - 1.0:
        a = int(actions["FASTER"])
        if a in available:
            return a

    if speed > cfg["target_speed"] + 1.5:
        a = int(actions["SLOWER"])
        if a in available:
            return a

    return int(actions["IDLE"])
