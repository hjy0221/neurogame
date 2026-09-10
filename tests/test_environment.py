import numpy as np

from neurogame.config import EnvConfig
from neurogame.environment import FoodWorld


def test_environment_observation_shape_and_range():
    world = FoodWorld(EnvConfig(food_count=3), seed=2)

    observation = world.observe()

    assert observation.shape == (12,)
    assert np.all(observation >= 0.0)
    assert np.all(observation <= 1.0)


def test_environment_step_moves_agent():
    world = FoodWorld(EnvConfig(food_count=3), seed=2)
    before = (world.agent.x, world.agent.y)

    result = world.step(turn=0.0, thrust=1.0)

    after = (world.agent.x, world.agent.y)
    assert before != after
    assert result.observation.shape == (12,)
    assert isinstance(result.reward, float)
