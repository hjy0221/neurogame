from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurogame.config import BrainConfig


@dataclass
class BrainSnapshot:
    voltage: np.ndarray
    spikes: np.ndarray
    activity_trace: np.ndarray
    motor_values: np.ndarray
    mean_rate: float


class SpikingBrain:
    """A compact recurrent spiking-style neural controller."""

    def __init__(self, config: BrainConfig | None = None, seed: int | None = None):
        self.config = config or BrainConfig()
        if self.config.sensory_count + self.config.motor_count >= self.config.neuron_count:
            raise ValueError("Brain needs more neurons than sensory + motor groups.")

        self.rng = np.random.default_rng(seed)
        self.n = self.config.neuron_count
        self.sensory_idx = np.arange(self.config.sensory_count)
        self.motor_idx = np.arange(self.n - self.config.motor_count, self.n)
        self.hidden_idx = np.arange(self.config.sensory_count, self.n - self.config.motor_count)

        self.neuron_sign = np.ones(self.n, dtype=np.float32)
        inhibitory_count = int(self.n * self.config.inhibitory_fraction)
        inhibitory_pool = self.hidden_idx.copy()
        self.rng.shuffle(inhibitory_pool)
        self.neuron_sign[inhibitory_pool[:inhibitory_count]] = -1.0

        self.weights = self._make_sparse_weights()
        self.input_weights = self._make_input_weights()
        self.motor_weights = self._make_motor_weights()

        self.voltage = np.zeros(self.n, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.activity_trace = np.zeros(self.n, dtype=np.float32)
        self.recent_reward = 0.0

    def _make_sparse_weights(self) -> np.ndarray:
        cfg = self.config
        mask = self.rng.random((self.n, self.n)) < cfg.connection_probability
        np.fill_diagonal(mask, False)
        mask[:, self.sensory_idx] = False

        base = self.rng.gamma(shape=1.7, scale=0.35, size=(self.n, self.n)).astype(np.float32)
        signed = base * self.neuron_sign[:, None]
        weights = np.where(mask, signed, 0.0).astype(np.float32)
        return weights * cfg.recurrent_gain

    def _make_input_weights(self) -> np.ndarray:
        weights = np.zeros((self.config.sensory_count, self.n), dtype=np.float32)
        weights[:, self.sensory_idx] = np.eye(self.config.sensory_count, dtype=np.float32) * 1.5

        fanout = min(80, len(self.hidden_idx))
        for sensor in range(self.config.sensory_count):
            targets = self.rng.choice(self.hidden_idx, size=fanout, replace=False)
            weights[sensor, targets] = self.rng.uniform(0.05, 0.55, size=fanout)

        return weights * self.config.input_gain

    def _make_motor_weights(self) -> np.ndarray:
        weights = np.zeros((self.n, self.config.motor_count), dtype=np.float32)
        for motor in range(self.config.motor_count):
            sources = self.rng.choice(self.hidden_idx, size=min(120, len(self.hidden_idx)), replace=False)
            weights[sources, motor] = self.rng.normal(0.0, 0.35, size=len(sources))

        # A small innate bias: left sensors lean left, right sensors lean right,
        # and frontal food signals lean forward.
        weights[0:4, 0] += np.array([0.2, 0.45, 0.25, 0.05], dtype=np.float32)
        weights[4:8, 2] += np.array([0.05, 0.25, 0.45, 0.2], dtype=np.float32)
        weights[8:12, 1] += np.array([0.35, 0.55, 0.55, 0.35], dtype=np.float32)
        return weights

    def reset_state(self) -> None:
        self.voltage.fill(0.0)
        self.spikes.fill(0.0)
        self.refractory.fill(0)
        self.activity_trace.fill(0.0)
        self.recent_reward = 0.0

    def step(self, sensory: np.ndarray, reward: float = 0.0) -> BrainSnapshot:
        cfg = self.config
        sensory = np.asarray(sensory, dtype=np.float32)
        if sensory.shape != (cfg.sensory_count,):
            raise ValueError(f"Expected sensory shape {(cfg.sensory_count,)}, got {sensory.shape}.")

        recurrent_current = self.spikes @ self.weights
        sensory_current = sensory @ self.input_weights
        current = recurrent_current + sensory_current

        active = self.refractory <= 0
        self.voltage[active] = self.voltage[active] * cfg.leak + current[active]
        self.voltage[~active] = cfg.reset_voltage
        self.refractory[self.refractory > 0] -= 1

        new_spikes = (self.voltage > cfg.threshold).astype(np.float32)
        spiked = new_spikes > 0
        self.voltage[spiked] = cfg.reset_voltage
        self.refractory[spiked] = cfg.refractory_steps
        self.spikes = new_spikes
        self.activity_trace = cfg.trace_decay * self.activity_trace + self.spikes

        motor_raw = self.activity_trace @ self.motor_weights
        motor_values = np.tanh(motor_raw / 12.0).astype(np.float32)

        self.recent_reward = 0.95 * self.recent_reward + reward
        if cfg.plasticity_enabled and reward != 0.0:
            self._apply_reward_plasticity(reward)

        return BrainSnapshot(
            voltage=self.voltage.copy(),
            spikes=self.spikes.copy(),
            activity_trace=self.activity_trace.copy(),
            motor_values=motor_values,
            mean_rate=float(np.mean(self.spikes)),
        )

    def _apply_reward_plasticity(self, reward: float) -> None:
        cfg = self.config
        pre = self.activity_trace[:, None]
        post = self.spikes[None, :]
        delta = cfg.learning_rate * reward * (pre @ post)
        signed_delta = delta * np.sign(self.weights + 1e-6)
        self.weights += signed_delta.astype(np.float32)
        np.clip(self.weights, -cfg.max_abs_weight, cfg.max_abs_weight, out=self.weights)

    def decode_action(self, motor_values: np.ndarray) -> tuple[float, float]:
        left, forward, right = motor_values
        turn = float(right - left)
        thrust = float((forward + 1.0) * 0.5)
        return turn, thrust
