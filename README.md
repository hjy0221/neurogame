# NeuroGame

[日本語](README.ja.md) · [English](README.md) · [한국어](README.ko.md)

NeuroGame is an experimental project that challenges biologically inspired neural networks to find the exit of a procedurally generated maze. A recurrent spiking neural network receives wall and exit sensors as input and outputs movement commands.

On the right side of the screen, the training neural network structure: input layer → recurrent middle layer → output layer and the amount of activity in each layer are displayed.

## Included features

- Project structure based on Python 3.12
- About 500 neurons implemented directly in NumPy
- Sparse circular connections including excitatory and inhibitory neurons
- Leaky membrane potential, spikes, refractory period, traces of activity
- Reward-based Hebbian plasticity
- Dense 128 → LSTM 128 recurrent PPO reinforcement learning that uses sensory and spiking activities as input
- 2D Pygame environment for procedural generation focusing on exit navigation
- Entrance/exit DFS maze reflected in collision and wall sensors
- Multiple crossroads and dead ends, single connectable route that can be completed
- Exit arrival compensation and entrance restart loop
- Procedurally generated maze where levels, grids, and dead ends increase with each exit you pass.
- Curriculum that can be learned even in long mazes with exit route checkpoint rewards
-Storable learning weights using exit and wall sensors
- 360-degree 8-way wall distance LiDAR and front, left, and right proximity sensors
- New cell search rewards and repeated visit deductions
- Exit search single reinforcement learning goal
- Input layer → circulation layer → output layer real-time visualization
- Tests run deterministically

## Installation

```bash
cd neurogame
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

In environments without Python 3.12, it may run on lower Python for basic verification, but the official baseline is Python 3.12.

## run

```bash
neurogame
```

Or run `python -m neurogame.app`.

```bash
python -m neurogame.app --neurons 500 --seed 7
python -m neurogame.app --no-plasticity
python -m neurogame.app --no-rl
python -m neurogame.app --headless --steps 300
python -m neurogame.app --headless --steps 5000 --episodes 20 --seed 1 --model NUL --save-model models/forager_exit.npz
python -m neurogame.app --headless --eval --steps 5000 --episodes 10 --seed 10001
python -m neurogame.app --headless --steps 10000 --episodes 10 --seed 1 --model NUL --save-model models/forager_exit.npz
```

## Controls

- `Space`: Pause / Continue
- `R`: Reset environment and brain state
- `P`: Turn plasticity on/off
- `L`: Turn reinforcement learning on/off
- `Esc`: Closed

Training uses multiple environments starting from seed 1, and evaluation is performed starting from seed 10001, which does not overlap. Exploration noise, Actor–Critic updates, and Hebbian plasticity are all disabled in `--eval`. The execution policy does not take BFS path input and uses only wall/exit sensors and sensor-based short-term avoidance memory.

The agent's movements are determined by a neural network, not the keyboard.

## test

```bash
pytest
```

In my current environment I don't have Python 3.12 installed, so I tested with whatever Python was available. Five tests and headless execution passed successfully.

## Next expansion direction

- Replace synthetic cyclic graphs with parts of real connectomes
- Rich visual input based on screen pixels
- Gymnasium environment support
- Save execution records and graphs
- Evolutionary learning or reinforcement learning of sensor-behavior connection
