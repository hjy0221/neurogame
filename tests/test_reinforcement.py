import numpy as np

from neurogame.reinforcement import ActorCriticController


def test_actor_critic_action_range_and_learning_update():
    learner = ActorCriticController(seed=4)
    observation = np.linspace(0.0, 1.0, 18, dtype=np.float32)
    motor = np.array([-0.2, 0.4, 0.1], dtype=np.float32)
    features = learner.features(observation, motor)
    before = learner.actor_output.copy()

    turn, thrust, action = learner.act(features)
    result = learner.learn(features, action, 1.0, features)

    assert -1.0 <= turn <= 1.0
    assert 0.0 <= thrust <= 1.0
    assert result.updates == 1
    assert not np.array_equal(before, learner.actor_output)
