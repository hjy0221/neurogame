# NeuroGame Agent Notes

NeuroGame v0.1 is intentionally a compact research toy, not a faithful fly-brain simulation.

## Project Shape

- Keep neural dynamics in `src/neurogame/brain.py`.
- Keep game physics and observations in `src/neurogame/environment.py`.
- Keep Pygame rendering and event handling in `src/neurogame/app.py`.
- Keep tests deterministic and fast; prefer seeded random generators.

## Design Direction

- The controller should remain NumPy-first.
- Avoid pulling in machine-learning frameworks unless a future version explicitly needs them.
- Preserve the boundary between environment observations, neural dynamics, and action decoding so a future connectome graph loader can replace the synthetic recurrent graph.

## v0.1 Scope

- Roughly 500 recurrent/spiking-style artificial neurons.
- 2D Pygame food-seeking environment.
- Visual/distance sensor inputs.
- Realtime visualization of neuron activity.
- Lightweight reward-gated plasticity.
