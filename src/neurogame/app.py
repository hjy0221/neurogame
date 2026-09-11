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
    parser.add_argument("--model", default="models/lstm_ppo_v2.pt", help="LSTM-PPO model to load when present.")
    parser.add_argument("--save-model", default="", help="Save trained RL weights on exit.")
    parser.add_argument("--eval", action="store_true", help="Deterministic evaluation: freeze all weights and noise.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of different seeded environments.")
    parser.add_argument("--food-count", type=int, default=0, help="Optional food count; the default task is exit-only.")
    parser.add_argument("--open-world", action="store_true", help="Train without internal maze walls.")
    parser.add_argument("--maze-columns", type=int, default=9, help="Starting maze columns.")
    parser.add_argument("--maze-rows", type=int, default=7, help="Starting maze rows.")
    return parser


def run_simulation(args: argparse.Namespace) -> dict[str, float]:
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import pygame

    pygame.init()
    brain_config = BrainConfig(
        neuron_count=args.neurons,
        plasticity_enabled=not args.no_plasticity and not args.eval,
    )
    env_config = EnvConfig(
        food_count=args.food_count,
        maze_enabled=not args.open_world,
        base_maze_columns=args.maze_columns,
        base_maze_rows=args.maze_rows,
    )
    render_config = RenderConfig()
    brain = SpikingBrain(brain_config, seed=args.seed)
    learner = ActorCriticController(seed=args.seed + 2)
    if args.model and Path(args.model).is_file():
        learner.load(args.model)
    starting_updates = learner.updates
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
    episode = 1
    completed_reward = 0.0
    completed_food = 0
    completed_exits = 0
    total_step_limit = args.steps * args.episodes if args.steps else 0

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
                    learner.reset_memory()
                    observation = world.reset()
                    brain.reset_state()
                    snapshot = brain.step(observation, reward=0.0)
                elif event.key == pygame.K_p and not args.eval:
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
                turn, thrust, action = learner.act(features, training=not args.eval)
            else:
                turn, thrust = brain.decode_action(snapshot.motor_values)
                action = None
            env_step = world.step(turn, thrust)
            snapshot = brain.step(env_step.observation, reward=env_step.reward)
            if rl_enabled and action is not None and not args.eval:
                next_features = learner.features(env_step.observation, snapshot.motor_values)
                learner.learn(features, action, env_step.reward, next_features, terminal=env_step.reached_exit)
                if env_step.reached_exit:
                    learner.end_episode(next_features)
            elif env_step.reached_exit:
                learner.reset_memory()
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
                not args.eval,
                learner.snapshot(),
            )

        if args.headless and args.steps and steps % args.steps == 0 and episode < args.episodes:
            completed_reward += world.total_reward
            completed_food += world.food_eaten
            completed_exits += world.exits_completed
            final_features = learner.features(observation, snapshot.motor_values)
            if rl_enabled and not args.eval:
                learner.end_episode(final_features)
            else:
                learner.reset_memory()
            episode += 1
            world = FoodWorld(env_config, seed=args.seed + episode)
            brain.reset_state()
            observation = world.observe()
            snapshot = brain.step(observation, reward=0.0)

        if total_step_limit and steps >= total_step_limit:
            running = False

        clock.tick(render_config.fps if renderer is not None else 0)

    pygame.quit()
    if args.save_model and rl_enabled and not args.eval:
        learner.end_episode(learner.features(observation, snapshot.motor_values))
        learner.save(args.save_model)
    return {
        "steps": float(steps),
        "episodes": float(episode),
        "total_reward": float(completed_reward + world.total_reward),
        "food_eaten": float(completed_food + world.food_eaten),
        "exits_completed": float(completed_exits + world.exits_completed),
        "mean_spike_rate": float(snapshot.mean_rate),
        "rl_updates": float(learner.updates),
        "updates_this_run": float(learner.updates - starting_updates),
        "exploration": float(learner.exploration),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stats = run_simulation(args)
    if args.headless:
        print(
            "Headless run complete: "
            f"mode={'eval' if args.eval else 'train'}, "
            f"episodes={int(stats['episodes'])}, "
            f"steps={int(stats['steps'])}, "
            f"reward={stats['total_reward']:.3f}, "
            f"food={int(stats['food_eaten'])}, "
            f"exits={int(stats['exits_completed'])}, "
            f"spike_rate={stats['mean_spike_rate']:.3f}, "
            f"rl_updates={int(stats['rl_updates'])}"
            f", updates_this_run={int(stats['updates_this_run'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
