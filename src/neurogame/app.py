from __future__ import annotations

import argparse
import os
import sys

from neurogame.brain import SpikingBrain
from neurogame.config import BrainConfig, EnvConfig, RenderConfig
from neurogame.environment import FoodWorld


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NeuroGame v0.1.")
    parser.add_argument("--neurons", type=int, default=500, help="Number of recurrent/spiking-style neurons.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a Pygame window.")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps. 0 means keep running.")
    parser.add_argument("--no-plasticity", action="store_true", help="Disable reward-gated plasticity.")
    return parser


def run_simulation(args: argparse.Namespace) -> dict[str, float]:
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame

    pygame.init()
    brain_config = BrainConfig(
        neuron_count=args.neurons,
        plasticity_enabled=not args.no_plasticity,
    )
    env_config = EnvConfig()
    render_config = RenderConfig()
    brain = SpikingBrain(brain_config, seed=args.seed)
    world = FoodWorld(env_config, seed=args.seed + 1)
    observation = world.observe()
    snapshot = brain.step(observation, reward=0.0)

    renderer = None
    if not args.headless:
        from neurogame.rendering import PygameRenderer

        renderer = PygameRenderer(pygame, env_config, render_config, brain)

    clock = pygame.time.Clock()
    paused = False
    running = True
    steps = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    observation = world.reset()
                    brain.reset_state()
                    snapshot = brain.step(observation, reward=0.0)
                elif event.key == pygame.K_p:
                    brain.config = BrainConfig(
                        neuron_count=brain.config.neuron_count,
                        sensory_count=brain.config.sensory_count,
                        motor_count=brain.config.motor_count,
                        connection_probability=brain.config.connection_probability,
                        inhibitory_fraction=brain.config.inhibitory_fraction,
                        leak=brain.config.leak,
                        threshold=brain.config.threshold,
                        reset_voltage=brain.config.reset_voltage,
                        refractory_steps=brain.config.refractory_steps,
                        input_gain=brain.config.input_gain,
                        recurrent_gain=brain.config.recurrent_gain,
                        trace_decay=brain.config.trace_decay,
                        learning_rate=brain.config.learning_rate,
                        max_abs_weight=brain.config.max_abs_weight,
                        plasticity_enabled=not brain.config.plasticity_enabled,
                    )

        if not paused:
            turn, thrust = brain.decode_action(snapshot.motor_values)
            env_step = world.step(turn, thrust)
            snapshot = brain.step(env_step.observation, reward=env_step.reward)
            observation = env_step.observation
            steps += 1

        if renderer is not None:
            renderer.draw(world, snapshot, observation, paused, brain.config.plasticity_enabled)

        if args.steps and steps >= args.steps:
            running = False

        clock.tick(render_config.fps if renderer is not None else 240)

    pygame.quit()
    return {
        "steps": float(steps),
        "total_reward": float(world.total_reward),
        "food_eaten": float(world.food_eaten),
        "mean_spike_rate": float(snapshot.mean_rate),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stats = run_simulation(args)
    if args.headless:
        print(
            "Headless run complete: "
            f"steps={int(stats['steps'])}, "
            f"reward={stats['total_reward']:.3f}, "
            f"food={int(stats['food_eaten'])}, "
            f"spike_rate={stats['mean_spike_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
