import numpy as np

from neurogame.config import EnvConfig
from neurogame.environment import FoodWorld, circle_hits_wall


def test_environment_observation_shape_and_range():
    world = FoodWorld(EnvConfig(food_count=3), seed=2)

    observation = world.observe()

    assert observation.shape == (18,)
    assert np.all(observation >= 0.0)
    assert np.all(observation <= 1.0)


def test_environment_step_moves_agent():
    world = FoodWorld(EnvConfig(food_count=3), seed=2)
    before = (world.agent.x, world.agent.y)

    result = world.step(turn=0.0, thrust=1.0)

    after = (world.agent.x, world.agent.y)
    assert before != after
    assert result.observation.shape == (18,)
    assert isinstance(result.reward, float)


def test_maze_walls_affect_sensors_and_food_stays_clear():
    world = FoodWorld(EnvConfig(food_count=20), seed=8)
    world.agent.x = 72.0
    world.agent.y = 60.0
    world.agent.heading = -np.pi / 2

    assert len(world.walls) > 30
    assert world.observe()[9] > 0.0
    assert all(
        not circle_hits_wall(float(food[0]), float(food[1]), world.config.food_radius, wall)
        for food in world.food
        for wall in world.walls
    )


def test_reaching_exit_rewards_and_returns_agent_to_entrance():
    world = FoodWorld(EnvConfig(), seed=9)
    world.food_active.fill(False)
    world.agent.x = 666.0
    world.agent.y = 260.0
    world.agent.heading = 0.0

    result = world.step(turn=0.0, thrust=0.0)

    assert result.reached_exit
    assert result.reward >= world.config.exit_reward + world.config.step_penalty
    assert world.exits_completed == 1
    assert world.maze_level == 2
    assert np.isclose(world.agent.x, world.entrance_position[0])


def test_each_exit_increases_maze_grid_and_regenerates_walls():
    world = FoodWorld(EnvConfig(), seed=10)
    world.food_active.fill(False)
    first_walls = world.walls
    first_columns = world.maze_columns
    world.agent.x = 666.0
    world.agent.y = float(sum(world.exit_bounds) / 2)

    result = world.step(turn=0.0, thrust=0.0)

    assert result.reached_exit
    assert world.maze_columns == first_columns + 1
    assert world.walls != first_walls
    assert all(
        not circle_hits_wall(float(food[0]), float(food[1]), world.config.food_radius, wall)
        for food in world.food
        for wall in world.walls
    )


def test_exit_stays_locked_until_all_food_is_eaten():
    world = FoodWorld(EnvConfig(food_count=14), seed=11)
    world.agent.x = 666.0
    world.agent.y = float(sum(world.exit_bounds) / 2)

    result = world.step(turn=0.0, thrust=0.0)

    assert not result.reached_exit
    assert world.maze_level == 1
    assert world.agent.x == 650.0


def test_different_environment_seeds_create_different_mazes():
    first = FoodWorld(EnvConfig(), seed=101)
    second = FoodWorld(EnvConfig(), seed=10_001)

    assert first.walls != second.walls
