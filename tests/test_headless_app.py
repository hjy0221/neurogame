from neurogame.app import build_parser, run_simulation


def test_headless_run_smoke():
    parser = build_parser()
    args = parser.parse_args(["--headless", "--steps", "5", "--neurons", "80", "--seed", "3"])

    stats = run_simulation(args)

    assert stats["steps"] == 5.0
    assert "total_reward" in stats
