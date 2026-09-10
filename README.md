# NeuroGame

NeuroGame is a v0.1 prototype of a biologically inspired neural controller playing a small 2D game. A recurrent, spiking-style artificial neural network receives distance/vision sensors from the game world, drives motor outputs, and shows live neural activity while the agent searches for food.

This is not a full fly-brain or DOOM connectome simulation. It is a deliberately small foundation that keeps the same loop:

```text
environment -> sensory encoding -> recurrent neural dynamics -> motor decoding -> action
```

## Features

- Python 3.12 project structure.
- NumPy-based neural dynamics with about 500 neurons by default.
- Sparse recurrent graph with excitatory and inhibitory neurons.
- Leaky membrane potential, spikes, refractory state, and activity traces.
- Reward-gated Hebbian plasticity.
- 2D food-seeking Pygame environment.
- Distance and directional food sensors.
- Live visualization of neural spikes, voltage traces, and basic metrics.
- Small deterministic test suite.

## Install

```bash
cd neurogame
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

If Python 3.12 is not installed, the core code is intentionally conservative and may run on older Python versions for local experimentation, but the project target is Python 3.12.

## Run

```bash
neurogame
```

or:

```bash
python -m neurogame.app
```

Useful options:

```bash
python -m neurogame.app --neurons 500 --seed 7
python -m neurogame.app --no-plasticity
python -m neurogame.app --headless --steps 300
```

## Controls

- `Space`: pause/resume
- `R`: reset environment and brain state
- `P`: toggle plasticity
- `Esc`: quit

The agent is controlled by the neural network, not the keyboard.

## Tests

```bash
pytest
```

In this workspace I verified the simulation in headless mode and ran the test suite with the available system Python. The local machine did not have Python 3.12 installed, so the project metadata still targets 3.12 while the implementation remains compatible enough for basic checks here.

## Future Extensions

- Replace the synthetic recurrent graph with a loaded connectome subgraph.
- Add richer visual encoding from rendered pixels.
- Support Gymnasium-style environments.
- Add saved run logs and plots.
- Add evolved or learned sensor/motor mappings.
