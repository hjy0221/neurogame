import numpy as np
import torch

from neurogame.reinforcement import ActorCriticController


def test_evaluation_changes_memory_but_not_parameters():
    learner = ActorCriticController(seed=41)
    before = {k: v.clone() for k, v in learner.network.state_dict().items()}
    observation = np.full(18, 0.5, dtype=np.float32)
    for _ in range(5):
        learner.act(learner.features(observation, np.zeros(3, dtype=np.float32)), training=False)
    assert learner.state[0].abs().sum() > 0
    assert all(torch.equal(v, learner.network.state_dict()[k]) for k, v in before.items())
    assert learner.updates == 0
    learner.reset_memory()
    assert learner.state[0].abs().sum() == 0


def test_rollout_likelihood_matches_policy_and_single_terminal_update_is_finite(tmp_path):
    learner = ActorCriticController(seed=18)
    features = learner.features(np.full(18, 0.5, dtype=np.float32), np.zeros(3, dtype=np.float32))
    state = tuple(x.clone() for x in learner.state)
    _, _, action = learner.act(features)
    pending = learner._pending
    with torch.no_grad():
        means, _, _ = learner.network(torch.from_numpy(features)[None, None], state)
        distribution = torch.distributions.Normal(means[0, 0], learner.network.log_std.exp().clamp(0.02, 0.35))
        log_prob = distribution.log_prob(torch.tensor(pending['action'])).sum().item()
    assert abs(log_prob - pending['log_prob']) < 1e-5
    before = learner.network.actor.weight.detach().clone()
    learner.learn(features, action, 8.0, features, terminal=True)
    assert all(torch.isfinite(p).all() for p in learner.network.parameters())
    assert not torch.equal(before, learner.network.actor.weight)
    path = tmp_path / 'policy.pt'
    learner.save(path)
    restored = ActorCriticController(seed=19)
    restored.load(path)
    assert all(torch.equal(a, b) for a, b in zip(learner.network.parameters(), restored.network.parameters()))


def test_actor_critic_action_range_and_learning_update():
    learner = ActorCriticController(seed=4)
    observation = np.linspace(0.0, 1.0, 18, dtype=np.float32)
    motor = np.array([-0.2, 0.4, 0.1], dtype=np.float32)
    features = learner.features(observation, motor)
    before = learner.network.actor.weight.detach().clone()

    turn, thrust, action = learner.act(features)
    result = learner.learn(features, action, 1.0, features)

    assert -1.0 <= turn <= 1.0
    assert 0.0 <= thrust <= 1.0
    assert result.updates == 1
    assert learner.updates == 1
    assert learner.state[0].shape == (1, 1, 128)
