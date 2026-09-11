from dataclasses import dataclass


@dataclass(frozen=True)
class BrainConfig:
    neuron_count: int = 500
    sensory_count: int = 18
    motor_count: int = 3
    connection_probability: float = 0.025
    inhibitory_fraction: float = 0.2
    leak: float = 0.92
    threshold: float = 1.0
    reset_voltage: float = 0.0
    refractory_steps: int = 2
    input_gain: float = 1.5
    recurrent_gain: float = 0.42
    trace_decay: float = 0.9
    learning_rate: float = 0.0015
    max_abs_weight: float = 1.5
    plasticity_enabled: bool = True


@dataclass(frozen=True)
class EnvConfig:
    width: int = 720
    height: int = 520
    agent_radius: float = 9.0
    food_radius: float = 7.0
    sensor_range: float = 190.0
    max_speed: float = 3.0
    turn_rate: float = 0.18
    food_count: int = 0
    step_penalty: float = -0.001
    wall_penalty: float = -0.02
    food_reward: float = 1.0
    exit_reward: float = 8.0
    waypoint_reward: float = 0.12
    all_food_bonus: float = 3.0
    locked_exit_penalty: float = -0.1
    progress_reward_scale: float = 0.015
    exit_progress_reward_scale: float = 0.03
    maze_enabled: bool = True
    base_maze_columns: int = 9
    base_maze_rows: int = 7
    novelty_reward: float = 0.03
    revisit_penalty: float = 0.001


@dataclass(frozen=True)
class ReinforcementConfig:
    gamma: float = 0.98
    actor_learning_rate: float = 0.0001
    critic_learning_rate: float = 0.002
    exploration_start: float = 0.04
    exploration_end: float = 0.01
    exploration_decay_steps: int = 30_000


@dataclass(frozen=True)
class RenderConfig:
    sidebar_width: int = 520
    fps: int = 60
    activity_columns: int = 25
