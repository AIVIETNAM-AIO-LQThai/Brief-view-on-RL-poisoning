ENV_ID = "highway-v0"
BASE_SEED = 20260915
EPISODES_PER_STYLE = 20

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 12,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": False,
        "order": "sorted",
        "see_behind": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 3,

    # Reduced from 25 after the initial baseline showed that crash outcome
# was too strongly associated with driving style.
    "vehicles_count": 20,

    "duration": 30,
    "simulation_frequency": 15,
    "policy_frequency": 5,

    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "show_trajectories": False,
}

STYLE_ORDER = ["cautious", "normal", "aggressive"]

# base_*_gap values are augmented dynamically when relative velocity makes a
# lane change more dangerous.
STYLE_CONFIG = {
    "cautious": {
        "target_speed": 22.0,
        "front_gap_trigger": 18.0,
        "base_front_gap": 22.0,
        "base_rear_gap": 16.0,
        "lane_change_probability": 0.30,
        "random_action_probability": 0.00,
    },
    "normal": {
        "target_speed": 25.5,
        "front_gap_trigger": 22.0,
        "base_front_gap": 19.0,
        "base_rear_gap": 14.0,
        "lane_change_probability": 0.55,
        "random_action_probability": 0.01,
    },
    "aggressive": {
        "target_speed": 28.5,
        "front_gap_trigger": 25.0,
        "base_front_gap": 16.0,
        "base_rear_gap": 12.0,
        "lane_change_probability": 0.75,
        "random_action_probability": 0.02,
    },
}

# We do not demand zero crashes. A risky but legitimate trajectory may crash.
# We only require that failure outcome does not overwhelm style.
ACCEPTANCE = {
    "max_crash_rate": {
        "cautious": 0.15,
        "normal": 0.30,
        "aggressive": 0.50,
    },
    "min_speed_gap_ms": 1.5,
    "require_lane_change_order": True,
}
