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

def _probability_per_decision(
    probability_per_second: float,
    policy_frequency: float,
) -> float:
    """
    Convert a probability defined per second into the equivalent
    probability for one policy decision.

    This keeps behaviour approximately comparable when policy_frequency
    changes.
    """
    p = float(np.clip(probability_per_second, 0.0, 1.0))
    frequency = max(float(policy_frequency), 1.0)

    return 1.0 - (1.0 - p) ** (1.0 / frequency)


def _nearest_target_speed_index(env, desired_speed: float) -> int:
    """
    Map the human-readable desired speed to HighwayEnv's discrete
    cruise-control speed setpoints.
    """
    target_speeds = np.asarray(
        env.unwrapped.action_type.target_speeds,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(target_speeds - float(desired_speed))
        )
    )


def _target_speed_index_not_above(
    env,
    desired_speed: float,
) -> int:
    """
    Choose the highest HighwayEnv speed setpoint that does not exceed
    the requested safety speed.
    """
    target_speeds = np.asarray(
        env.unwrapped.action_type.target_speeds,
        dtype=float,
    )

    valid = np.flatnonzero(
        target_speeds <= float(desired_speed)
    )

    if len(valid) == 0:
        return 0

    return int(valid[-1])

def choose_action(
    env,
    style: str,
    rng: np.random.Generator,
) -> int:
    cfg = STYLE_CONFIG[style]

    u = env.unwrapped
    ego = u.vehicle

    actions = u.action_type.actions_indexes
    available = set(
        int(x)
        for x in u.get_available_actions()
    )

    policy_frequency = float(
        u.config["policy_frequency"]
    )

    # Convert per-second benign randomness into a per-decision probability.
    random_probability = _probability_per_decision(
        cfg["random_action_probability_per_second"],
        policy_frequency,
    )

    if rng.random() < random_probability:
        return int(
            rng.choice(list(available))
        )

    front_gap, front_vehicle = _current_front(env)

    # HighwayEnv's FASTER/SLOWER actions change a target-speed index.
    # Therefore compare target indices rather than repeatedly reacting
    # to physical speed while the low-level controller is still catching up.
    desired_speed_index = _nearest_target_speed_index(
        env,
        cfg["target_speed"],
    )

    target_lane = getattr(
        ego,
        "target_lane_index",
        None,
    )

    lane_change_in_progress = (
        target_lane is not None
        and target_lane != ego.lane_index
    )

    if front_gap < cfg["front_gap_trigger"]:

        # Do not request another lane change while the previous one is
        # still being executed.
        candidates = (
            []
            if lane_change_in_progress
            else _safe_lane_candidates(env, cfg)
        )

        lane_change_probability = (
            _probability_per_decision(
                cfg[
                    "lane_change_probability_per_second"
                ],
                policy_frequency,
            )
        )

        if (
            candidates
            and rng.random()
            < lane_change_probability
        ):
            current_lane = int(
                ego.lane_index[2]
            )

            best = max(
                candidates,
                key=lambda lane_id: _lane_safety(
                    env,
                    lane_id,
                ).front_gap,
            )

            if best < current_lane:
                action = int(
                    actions["LANE_LEFT"]
                )

                if action in available:
                    return action

            elif best > current_lane:
                action = int(
                    actions["LANE_RIGHT"]
                )

                if action in available:
                    return action

        closing = (
            front_vehicle is not None
            and float(ego.speed)
            > float(front_vehicle.speed)
        )

        # Preserve the existing safety idea, but express it as a bounded
        # target-speed choice instead of blindly issuing SLOWER repeatedly.
        if (
            front_vehicle is not None
            and (
                closing
                or front_gap < 12.0
            )
        ):
            safe_reference_speed = float(
                front_vehicle.speed
            )

            # When extremely close, request a speed below the front car
            # instead of merely matching it.
            if front_gap < 12.0:
                safe_reference_speed -= 2.0

            safe_speed_index = (
                _target_speed_index_not_above(
                    env,
                    safe_reference_speed,
                )
            )

            desired_speed_index = min(
                desired_speed_index,
                safe_speed_index,
            )

    current_speed_index = int(
        ego.speed_index
    )

    if (
        current_speed_index
        < desired_speed_index
    ):
        action = int(
            actions["FASTER"]
        )

        if action in available:
            return action

    if (
        current_speed_index
        > desired_speed_index
    ):
        action = int(
            actions["SLOWER"]
        )

        if action in available:
            return action

    return int(actions["IDLE"])