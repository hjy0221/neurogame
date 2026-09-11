from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from neurogame.brain import SpikingBrain
from neurogame.config import BrainConfig, EnvConfig, RenderConfig
from neurogame.environment import FoodWorld
from neurogame.reinforcement import ActorCriticController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NeuroGame v0.1.")
    parser.add_argument("--neurons", type=int, default=500, help="Number of recurrent/spiking-style neurons.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a Pygame window.")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps. 0 means keep running.")
    parser.add_argument("--no-plasticity", action="store_true", help="Disable reward-gated plasticity.")
    parser.add_argument("--no-rl", action="store_true", help="Disable actor-critic reinforcement learning.")
    parser.add_argument("--model", default="models/forager_exit.npz", help="RL model to load when present.")
    parser.add_argument("--save-model", default="", help="Save trained RL weights on exit.")
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
    learner = ActorCriticController(seed=args.seed + 2)
    if args.model and Path(args.model).is_file():
        try:
            learner.load(args.model)
        except ValueError:
            # Input layouts can change between versions; train a fresh compatible model.
            pass
    rl_enabled = not args.no_rl
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
                elif event.key == pygame.K_l:
                    rl_enabled = not rl_enabled

        if not paused:
            features = learner.features(observation, snapshot.motor_values)
            if rl_enabled:
                turn, thrust, action = learner.act(features, training=True)
            else:
                turn, thrust = brain.decode_action(snapshot.motor_values)
                action = None
            env_step = world.step(turn, thrust)
            snapshot = brain.step(env_step.observation, reward=env_step.reward)
            if rl_enabled and action is not None:
                next_features = learner.features(env_step.observation, snapshot.motor_values)
                learner.learn(features, action, env_step.reward, next_features)
            observation = env_step.observation
            steps += 1

        if renderer is not None:
            renderer.draw(
                world,
                snapshot,
                observation,
                paused,
                brain.config.plasticity_enabled,
                rl_enabled,
                learner.snapshot(),
            )

        if args.steps and steps >= args.steps:
            running = False

        clock.tick(render_config.fps if renderer is not None else 240)

    pygame.quit()
    if args.save_model and rl_enabled:
        learner.save(args.save_model)
    return {
        "steps": float(steps),
        "total_reward": float(world.total_reward),
        "food_eaten": float(world.food_eaten),
        "exits_completed": float(world.exits_completed),
        "mean_spike_rate": float(snapshot.mean_rate),
        "rl_updates": float(learner.updates),
        "exploration": float(learner.exploration),
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
            f"exits={int(stats['exits_completed'])}, "
            f"spike_rate={stats['mean_spike_rate']:.3f}, "
            f"rl_updates={int(stats['rl_updates'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
