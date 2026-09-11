from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neurogame.config import ReinforcementConfig


@dataclass
class LearningSnapshot:
    td_error: float
    value: float
    exploration: float
    updates: int


class RecurrentPolicy(nn.Module):
    def __init__(self, feature_count: int = 29, hidden_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_count, 128), nn.Tanh())
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, 2)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.tensor([-1.8, -2.2]))

    def forward(self, features: torch.Tensor, state):
        encoded = self.encoder(features)
        recurrent, next_state = self.lstm(encoded, state)
        raw = self.actor(recurrent)
        # Sensor-derived residuals are not a map: they only stabilize local steering.
        turn = torch.tanh(raw[..., 0] + 0.65 * features[..., 27] + 0.8 * features[..., 25])
        thrust = torch.sigmoid(raw[..., 1] + 1.0)
        means = torch.stack((turn, thrust), dim=-1)
        value = self.critic(recurrent).squeeze(-1)
        return means, value, next_state


class ActorCriticController:
    """LSTM policy trained with clipped PPO from sensor-only observations."""

    feature_count = 29
    hidden_size = 128
    rollout_size = 128

    def __init__(self, config: ReinforcementConfig | None = None, seed: int | None = None):
        self.config = config or ReinforcementConfig()
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(0 if seed is None else seed)
        self.network = RecurrentPolicy(self.feature_count, self.hidden_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.previous_action = np.array([0.0, 0.7], dtype=np.float32)
        self.escape_steps = 0
        self.escape_direction = 1.0
        self.updates = 0
        self.last_td_error = 0.0
        self.last_value = 0.0
        self._buffer: list[dict[str, object]] = []
        self._rollout_state = None
        self._pending: dict[str, object] | None = None
        self.reset_memory()

    def reset_memory(self) -> None:
        self._buffer.clear()
        self._pending = None
        self._rollout_state = None
        self.state = (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        )
        self.previous_action[:] = (0.0, 0.7)
        self.escape_steps = 0

    def features(self, observation: np.ndarray, motor_values: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        motor_values = np.asarray(motor_values, dtype=np.float32)
        if observation.shape != (18,) or motor_values.shape != (3,):
            raise ValueError("LSTM policy expects 18 sensors and 3 motor values.")
        left_obstacle = float(np.max(observation[2:4]))
        right_obstacle = float(np.max(observation[5:7]))
        wall_turn = left_obstacle - right_obstacle
        front_clear = float(1.0 - max(observation[3], observation[4], observation[5]))
        exit_sine = float(observation[12] * 2.0 - 1.0)
        exit_cosine = float(observation[13] * 2.0 - 1.0)
        if exit_cosine >= 0.0:
            target_turn = exit_sine
        elif abs(exit_sine) > 0.03:
            target_turn = 1.0 if exit_sine > 0.0 else -1.0
        else:
            target_turn = 1.0 if self.previous_action[0] >= 0.0 else -1.0
        return np.concatenate((
            observation,
            motor_values,
            self.previous_action,
            [0.0, 0.0, wall_turn, front_clear, target_turn, 1.0],
        )).astype(np.float32)

    @property
    def exploration(self) -> float:
        return float(torch.exp(self.network.log_std).mean().detach())

    def act(self, features: np.ndarray, training: bool = True) -> tuple[float, float, np.ndarray]:
        tensor = torch.from_numpy(features).view(1, 1, -1)
        if not self._buffer:
            self._rollout_state = tuple(item.detach().clone() for item in self.state)
        with torch.no_grad():
            means, value, next_state = self.network(tensor, self.state)
            means = means[0, 0]
            std = torch.exp(self.network.log_std).clamp(0.02, 0.35)
            distribution = torch.distributions.Normal(means, std)
            action_tensor = distribution.sample() if training else means
            log_prob = distribution.log_prob(action_tensor).sum()
        self.state = tuple(item.detach() for item in next_state)

        action = action_tensor.numpy().astype(np.float32)
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1], 0.05, 1.0)
        self.previous_action = action.copy()
        self.last_value = float(value.item())
        self._pending = {
            "features": features.copy(), "action": action_tensor.numpy().copy(),
            "log_prob": float(log_prob.item()), "value": self.last_value,
        }
        return float(action[0]), float(action[1]), action

    def learn(self, features: np.ndarray, action: np.ndarray, reward: float,
              next_features: np.ndarray, terminal: bool = False) -> LearningSnapshot:
        if self._pending is None:
            return self.snapshot()
        transition = dict(self._pending)
        transition["reward"] = float(reward)
        self._buffer.append(transition)
        self._pending = None
        self.updates += 1
        self.last_td_error = float(reward - self.last_value)
        if terminal or len(self._buffer) >= self.rollout_size:
            self._ppo_update(next_features, terminal=terminal)
        return self.snapshot()

    def end_episode(self, final_features: np.ndarray) -> None:
        if self._buffer:
            self._ppo_update(final_features, terminal=True)
        self.reset_memory()

    def _ppo_update(self, next_features: np.ndarray, terminal: bool = False) -> None:
        cfg = self.config
        running_return = 0.0
        if not terminal:
            with torch.no_grad():
                next_tensor = torch.from_numpy(next_features).view(1, 1, -1)
                _, bootstrap, _ = self.network(next_tensor, self.state)
                running_return = float(bootstrap.item())
        advantages_list = []
        advantage = 0.0
        for item in reversed(self._buffer):
            delta = float(item["reward"]) + cfg.gamma * running_return - float(item["value"])
            advantage = delta + cfg.gamma * 0.95 * advantage
            advantages_list.append(advantage)
            running_return = float(item["value"])
        features = torch.tensor(np.stack([item["features"] for item in self._buffer]))[None, ...]
        actions = torch.tensor(np.stack([item["action"] for item in self._buffer]))
        old_log_probs = torch.tensor([item["log_prob"] for item in self._buffer])
        old_values = torch.tensor([item["value"] for item in self._buffer])
        advantages = torch.tensor(list(reversed(advantages_list)), dtype=torch.float32)
        returns_tensor = advantages + old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
        initial_state = tuple(item.detach() for item in self._rollout_state)

        for _ in range(4):
            means, values, _ = self.network(features, initial_state)
            means = means[0]
            std = torch.exp(self.network.log_std).clamp(0.02, 0.35)
            distribution = torch.distributions.Normal(means, std)
            log_probs = distribution.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantages
            actor_loss = -torch.min(ratio * advantages, clipped).mean()
            critic_loss = 0.5 * (returns_tensor - values[0]).pow(2).mean()
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        self._buffer.clear()
        self._rollout_state = None

    def snapshot(self) -> LearningSnapshot:
        return LearningSnapshot(self.last_td_error, self.last_value, self.exploration, self.updates)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": 2,
            "model": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 2:
            raise ValueError("Train a version 2 PPO checkpoint; old action probabilities were inconsistent.")
        self.network.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.updates = int(checkpoint.get("updates", 0))
