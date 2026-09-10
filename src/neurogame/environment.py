from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np

from neurogame.config import EnvConfig


@dataclass
class AgentState:
    x: float
    y: float
    heading: float
    speed: float = 0.0


@dataclass
class EnvStep:
    observation: np.ndarray
    reward: float
    ate_food: bool
    distance_to_nearest_food: float


class FoodWorld:
    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.agent = AgentState(self.config.width / 2, self.config.height / 2, 0.0)
        self.food = np.zeros((self.config.food_count, 2), dtype=np.float32)
        self.total_reward = 0.0
        self.food_eaten = 0
        self.steps = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.agent = AgentState(
            x=float(self.config.width / 2),
            y=float(self.config.height / 2),
            heading=float(self.rng.uniform(-pi, pi)),
            speed=0.0,
        )
        self.food_eaten = 0
        self.total_reward = 0.0
        self.steps = 0
        for idx in range(self.config.food_count):
            self.food[idx] = self._random_food_position()
        return self.observe()

    def _random_food_position(self) -> np.ndarray:
        margin = 28
        return np.array(
            [
                self.rng.uniform(margin, self.config.width - margin),
                self.rng.uniform(margin, self.config.height - margin),
            ],
            dtype=np.float32,
        )

    def step(self, turn: float, thrust: float) -> EnvStep:
        cfg = self.config
        turn = float(np.clip(turn, -1.0, 1.0))
        thrust = float(np.clip(thrust, 0.0, 1.0))

        self.agent.heading = wrap_angle(self.agent.heading + turn * cfg.turn_rate)
        target_speed = thrust * cfg.max_speed
        self.agent.speed = 0.82 * self.agent.speed + 0.18 * target_speed
        self.agent.x += cos(self.agent.heading) * self.agent.speed
        self.agent.y += sin(self.agent.heading) * self.agent.speed

        reward = cfg.step_penalty
        hit_wall = False
        if self.agent.x < cfg.agent_radius:
            self.agent.x = cfg.agent_radius
            hit_wall = True
        elif self.agent.x > cfg.width - cfg.agent_radius:
            self.agent.x = cfg.width - cfg.agent_radius
            hit_wall = True
        if self.agent.y < cfg.agent_radius:
            self.agent.y = cfg.agent_radius
            hit_wall = True
        elif self.agent.y > cfg.height - cfg.agent_radius:
            self.agent.y = cfg.height - cfg.agent_radius
            hit_wall = True

        if hit_wall:
            reward += cfg.wall_penalty
            self.agent.heading = wrap_angle(self.agent.heading + self.rng.uniform(-0.5, 0.5))

        ate_food = False
        agent_pos = np.array([self.agent.x, self.agent.y], dtype=np.float32)
        distances = np.linalg.norm(self.food - agent_pos, axis=1)
        nearest_idx = int(np.argmin(distances))
        if distances[nearest_idx] <= cfg.agent_radius + cfg.food_radius:
            reward += cfg.food_reward
            ate_food = True
            self.food_eaten += 1
            self.food[nearest_idx] = self._random_food_position()

        self.steps += 1
        self.total_reward += reward
        observation = self.observe()
        new_distances = np.linalg.norm(self.food - agent_pos, axis=1)
        return EnvStep(
            observation=observation,
            reward=float(reward),
            ate_food=ate_food,
            distance_to_nearest_food=float(np.min(new_distances)),
        )

    def observe(self) -> np.ndarray:
        cfg = self.config
        agent_pos = np.array([self.agent.x, self.agent.y], dtype=np.float32)
        rel = self.food - agent_pos
        distances = np.linalg.norm(rel, axis=1)
        angles = np.array([wrap_angle(atan2(y, x) - self.agent.heading) for x, y in rel], dtype=np.float32)

        ray_angles = np.linspace(-pi * 0.75, pi * 0.75, 8, dtype=np.float32)
        food_rays = np.zeros(8, dtype=np.float32)
        for idx, ray in enumerate(ray_angles):
            angular_gain = np.exp(-((wrap_angle(angles - ray)) ** 2) / 0.22)
            distance_gain = np.clip(1.0 - distances / cfg.sensor_range, 0.0, 1.0)
            food_rays[idx] = float(np.max(angular_gain * distance_gain))

        front_wall = distance_to_wall(self.agent, cfg, 0.0) / cfg.sensor_range
        left_wall = distance_to_wall(self.agent, cfg, -pi / 3) / cfg.sensor_range
        right_wall = distance_to_wall(self.agent, cfg, pi / 3) / cfg.sensor_range
        nearest_food = np.min(distances) / cfg.sensor_range

        wall_sensors = 1.0 - np.clip(np.array([left_wall, front_wall, right_wall]), 0.0, 1.0)
        food_distance = np.array([1.0 - np.clip(nearest_food, 0.0, 1.0)], dtype=np.float32)
        return np.concatenate([food_rays, wall_sensors.astype(np.float32), food_distance]).astype(np.float32)


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + pi) % (2 * pi) - pi


def distance_to_wall(agent: AgentState, cfg: EnvConfig, relative_angle: float) -> float:
    angle = agent.heading + relative_angle
    dx = cos(angle)
    dy = sin(angle)
    candidates: list[float] = []

    if abs(dx) > 1e-6:
        candidates.append((cfg.agent_radius - agent.x) / dx)
        candidates.append((cfg.width - cfg.agent_radius - agent.x) / dx)
    if abs(dy) > 1e-6:
        candidates.append((cfg.agent_radius - agent.y) / dy)
        candidates.append((cfg.height - cfg.agent_radius - agent.y) / dy)

    positive = [value for value in candidates if value > 0]
    return min(positive) if positive else cfg.sensor_range
