"""Fixed experiment configuration.

Changing this file after Stage 1 invalidates later comparisons unless a new
experiment version/checkpoint is created.
"""

ENV_ID = "highway-v0"
BASE_SEED = 20260915
EPISODES_PER_STYLE = 20

# Explicit HighwayEnv configuration.
# We use non-normalized relative kinematic observations so the recorded
# numbers are directly interpretable in metres and metres/second.
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
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 3,
    "vehicles_count": 25,
    "duration": 30,              # seconds
    "simulation_frequency": 15,  # internal physics Hz
    "policy_frequency": 1,       # one decision per second
    "collision_reward": -1,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
    "show_trajectories": False,
}

STYLE_ORDER = ["cautious", "normal", "aggressive"]

STYLE_CONFIG = {
    "cautious": {
        "target_speed": 22.0,
        "front_gap_trigger": 16.0,
        "candidate_front_gap": 20.0,
        "candidate_rear_gap": 14.0,
        "lane_change_probability": 0.35,
        "random_action_probability": 0.01,
    },
    "normal": {
        "target_speed": 26.0,
        "front_gap_trigger": 20.0,
        "candidate_front_gap": 16.0,
        "candidate_rear_gap": 11.0,
        "lane_change_probability": 0.65,
        "random_action_probability": 0.02,
    },
    "aggressive": {
        "target_speed": 30.0,
        "front_gap_trigger": 27.0,
        "candidate_front_gap": 11.0,
        "candidate_rear_gap": 8.0,
        "lane_change_probability": 0.90,
        "random_action_probability": 0.04,
    },
}
