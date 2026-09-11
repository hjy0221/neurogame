from neurogame.app import build_parser, run_simulation


def test_headless_run_smoke():
    parser = build_parser()
    args = parser.parse_args(["--headless", "--steps", "5", "--neurons", "80", "--seed", "3"])

    stats = run_simulation(args)

    assert stats["steps"] == 5.0
    assert "total_reward" in stats


def test_evaluation_freezes_learning_across_unseen_environments():
    parser = build_parser()
    args = parser.parse_args([
        "--headless", "--eval", "--steps", "3", "--episodes", "2",
        "--neurons", "80", "--seed", "10001", "--model", "missing.npz",
    ])

    stats = run_simulation(args)

    assert stats["steps"] == 6.0
    assert stats["episodes"] == 2.0
    assert stats["rl_updates"] == 0.0
    assert stats["updates_this_run"] == 0.0
