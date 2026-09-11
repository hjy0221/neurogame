from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurogame.config import ReinforcementConfig


@dataclass
class LearningSnapshot:
    td_error: float
    value: float
    exploration: float
    updates: int


class ActorCriticController:
    """Online neural actor-critic using sensory, spiking, and short-term action state."""

    def __init__(self, config: ReinforcementConfig | None = None, seed: int | None = None):
        self.config = config or ReinforcementConfig()
        self.rng = np.random.default_rng(seed)
        self.feature_count = 29
        self.hidden_count = 64
        self.deep_count = 32
        scale = np.sqrt(2.0 / self.feature_count)
        self.actor_hidden = self.rng.normal(0.0, scale, (64, 29)).astype(np.float32)
        self.actor_deep = self.rng.normal(0.0, np.sqrt(2.0 / 65), (32, 65)).astype(np.float32)
        self.actor_output = np.zeros((2, 33), dtype=np.float32)
        self.actor_skip = np.zeros((2, 29), dtype=np.float32)
        # Neural priors: face sensed food, avoid walls, and keep moving.
        self.actor_skip[0, 27] = 4.0
        self.actor_skip[0, 25] = 0.8
        self.actor_skip[0, 21] = 0.35
        self.actor_skip[1, 24] = 0.5
        self.actor_skip[1, 26] = 1.0
        self.actor_skip[1, 28] = 0.35
        self.critic_hidden = self.rng.normal(0.0, scale, (64, 29)).astype(np.float32)
        self.critic_deep = self.rng.normal(0.0, np.sqrt(2.0 / 65), (32, 65)).astype(np.float32)
        self.critic_output = np.zeros(33, dtype=np.float32)
        self.previous_action = np.array([0.0, 0.7], dtype=np.float32)
        self.exploration_state = np.zeros(2, dtype=np.float32)
        self.updates = 0
        self.last_td_error = 0.0
        self.last_value = 0.0

    def features(self, observation: np.ndarray, motor_values: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        motor_values = np.asarray(motor_values, dtype=np.float32)
        if observation.shape != (18,) or motor_values.shape != (3,):
            raise ValueError("Actor-critic expects 18 sensors and 3 motor values.")
        ray_angles = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        food_strength = float(np.sum(observation[:8]))
        food_turn = float(np.dot(observation[:8], ray_angles) / max(food_strength, 1e-5))
        food_visibility = float(np.max(observation[:8]))
        wall_turn = float(observation[8] - observation[10])
        front_clear = float(1.0 - observation[9])
        exit_turn = float(observation[12] * 2.0 - 1.0)
        nearest_food_turn = float(observation[15] * 2.0 - 1.0)
        food_remaining = float(observation[17])
        if food_remaining > 0.0:
            target_turn = nearest_food_turn
        else:
            target_turn = exit_turn
        return np.concatenate((
            observation, motor_values, self.previous_action,
            [food_turn, food_visibility, wall_turn, front_clear, target_turn, 1.0],
        )).astype(np.float32)

    @property
    def exploration(self) -> float:
        cfg = self.config
        fraction = min(1.0, self.updates / max(1, cfg.exploration_decay_steps))
        return cfg.exploration_start + fraction * (cfg.exploration_end - cfg.exploration_start)

    def _actor(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hidden = np.tanh(self.actor_hidden @ features)
        hidden_state = np.concatenate((hidden, [1.0])).astype(np.float32)
        deep = np.tanh(self.actor_deep @ hidden_state)
        deep_state = np.concatenate((deep, [1.0])).astype(np.float32)
        raw = np.tanh(0.2 * (self.actor_output @ deep_state) + self.actor_skip @ features)
        means = np.array([raw[0], (raw[1] + 1.0) * 0.5], dtype=np.float32)
        return means, raw, hidden_state, deep_state

    def _critic(self, features: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        hidden = np.tanh(self.critic_hidden @ features)
        hidden_state = np.concatenate((hidden, [1.0])).astype(np.float32)
        deep = np.tanh(self.critic_deep @ hidden_state)
        deep_state = np.concatenate((deep, [1.0])).astype(np.float32)
        return float(self.critic_output @ deep_state), hidden_state, deep_state

    def act(self, features: np.ndarray, training: bool = True) -> tuple[float, float, np.ndarray]:
        means, _, _, _ = self._actor(features)
        action = means.copy()
        if training:
            push = self.rng.normal(0.0, self.exploration, size=2).astype(np.float32)
            self.exploration_state = 0.92 * self.exploration_state + 0.35 * push
            action += self.exploration_state
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1], 0.12, 1.0)
        self.previous_action = action.copy()
        return float(action[0]), float(action[1]), action

    def learn(self, features: np.ndarray, action: np.ndarray, reward: float,
              next_features: np.ndarray) -> LearningSnapshot:
        cfg = self.config
        value, critic_hidden_state, critic_deep_state = self._critic(features)
        next_value, _, _ = self._critic(next_features)
        td_error = float(np.clip(reward + cfg.gamma * next_value - value, -2.0, 2.0))
        old_critic_output = self.critic_output.copy()
        self.critic_output += cfg.critic_learning_rate * td_error * critic_deep_state
        deep_signal = old_critic_output[:-1] * (1.0 - critic_deep_state[:-1] ** 2)
        old_critic_deep = self.critic_deep.copy()
        self.critic_deep += cfg.critic_learning_rate * td_error * deep_signal[:, None] * critic_hidden_state[None, :]
        hidden_signal = old_critic_deep[:, :-1].T @ deep_signal
        hidden_signal *= 1.0 - critic_hidden_state[:-1] ** 2
        self.critic_hidden += cfg.critic_learning_rate * td_error * hidden_signal[:, None] * features[None, :]

        means, raw_means, actor_hidden_state, actor_deep_state = self._actor(features)
        output_gradient = np.array([1.0, 0.5], dtype=np.float32) * (1.0 - raw_means**2)
        score = ((action - means) / max(self.exploration**2, 0.01)) * output_gradient
        old_actor_output = self.actor_output.copy()
        policy_signal = cfg.actor_learning_rate * td_error * score
        self.actor_output += 0.2 * policy_signal[:, None] * actor_deep_state[None, :]
        self.actor_skip += 0.25 * policy_signal[:, None] * features[None, :]
        deep_actor_signal = 0.2 * (old_actor_output[:, :-1].T @ policy_signal)
        deep_actor_signal *= 1.0 - actor_deep_state[:-1] ** 2
        old_actor_deep = self.actor_deep.copy()
        self.actor_deep += deep_actor_signal[:, None] * actor_hidden_state[None, :]
        hidden_actor_signal = old_actor_deep[:, :-1].T @ deep_actor_signal
        hidden_actor_signal *= 1.0 - actor_hidden_state[:-1] ** 2
        self.actor_hidden += hidden_actor_signal[:, None] * features[None, :]
        for weights in (self.actor_hidden, self.actor_deep, self.actor_output, self.actor_skip,
                        self.critic_hidden, self.critic_deep, self.critic_output):
            np.clip(weights, -3.0, 3.0, out=weights)
        self.updates += 1
        self.last_td_error = td_error
        self.last_value = value
        return self.snapshot()

    def snapshot(self) -> LearningSnapshot:
        return LearningSnapshot(self.last_td_error, self.last_value, self.exploration, self.updates)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            actor_hidden=self.actor_hidden,
            actor_deep=self.actor_deep,
            actor_output=self.actor_output,
            actor_skip=self.actor_skip,
            critic_hidden=self.critic_hidden,
            critic_deep=self.critic_deep,
            critic_output=self.critic_output,
            updates=np.array(self.updates),
        )

    def load(self, path: str | Path) -> None:
        with np.load(path) as data:
            for name in (
                "actor_hidden", "actor_deep", "actor_output", "actor_skip",
                "critic_hidden", "critic_deep", "critic_output",
            ):
                target = getattr(self, name)
                loaded = data[name].astype(np.float32)
                if loaded.shape != target.shape:
                    raise ValueError(f"Incompatible model parameter {name}: {loaded.shape} != {target.shape}")
                target[...] = loaded
            self.updates = int(data["updates"])
