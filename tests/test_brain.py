import numpy as np

from neurogame.brain import SpikingBrain
from neurogame.config import BrainConfig


def test_brain_step_returns_motor_values_and_spikes():
    brain = SpikingBrain(BrainConfig(neuron_count=80, sensory_count=12, motor_count=3), seed=1)
    sensory = np.linspace(0, 1, 12, dtype=np.float32)

    snapshot = brain.step(sensory)

    assert snapshot.spikes.shape == (80,)
    assert snapshot.voltage.shape == (80,)
    assert snapshot.motor_values.shape == (3,)
    assert np.all(snapshot.motor_values >= -1.0)
    assert np.all(snapshot.motor_values <= 1.0)


def test_brain_rejects_wrong_observation_shape():
    brain = SpikingBrain(BrainConfig(neuron_count=80, sensory_count=12, motor_count=3), seed=1)

    try:
        brain.step(np.zeros(10, dtype=np.float32))
    except ValueError as exc:
        assert "Expected sensory shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError.")
