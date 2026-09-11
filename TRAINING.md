# Destination learning, version 2

Measured in this run (80 spiking neurons, frozen plasticity): open-world warm-up, 8,000 steps, 25 exits; 3x3-start training, 12,000 steps, 8 exits. Held-out 3x3-start evaluation, seeds 20051-20053, 3,000 steps each, 7 exits total and zero updates. Held-out default 9x7 evaluation, seeds 30051-30053, 3,000 steps each, zero exits and zero updates. Exits include automatic level advancement; these counts are not episode success percentages. The checkpoint has 20,000 training transitions. Large-maze navigation remains unsolved.

The controller uses Dense(128), LSTM(128), actor and value heads. Training combines recurrent clipped PPO, GAE(lambda=0.95), entropy regularization, gradient clipping, and a curriculum of open arenas and progressively larger mazes. There is no BFS guidance in policy observations or rewards. The policy has local sensor steering biases, but no action-overriding escape controller.

PPO stores the sampled Gaussian action and its matching log probability. The environment receives a clipped version; optimization uses the original sample. Terminal rewards are processed before clearing recurrent memory. Single-transition updates remain finite. Old checkpoints are incompatible and retained separately.

Novelty is rewarded once per newly visited cell. Revisit costs occur only when entering a previously visited cell, not for every frame spent traversing one.

From the repository directory in PowerShell:

```powershell
.\.venv312\Scripts\python.exe -m neurogame.app --headless --open-world --steps 2000 --episodes 4 --neurons 80 --no-plasticity --seed 50 --model '' --save-model models/lstm_ppo_v2.pt
.\.venv312\Scripts\python.exe -m neurogame.app --headless --steps 3000 --episodes 4 --maze-columns 3 --maze-rows 3 --neurons 80 --no-plasticity --seed 150 --model models/lstm_ppo_v2.pt --save-model models/lstm_ppo_v2.pt
.\.venv312\Scripts\python.exe -m neurogame.app --headless --eval --steps 3000 --episodes 3 --maze-columns 3 --maze-rows 3 --neurons 80 --seed 20050 --model models/lstm_ppo_v2.pt
.\.venv312\Scripts\python.exe -m neurogame.app --eval --neurons 80 --seed 20050 --model models/lstm_ppo_v2.pt
```

Environment seeds are CLI seed + episode index. Training and evaluation must use disjoint ranges. Use the same neuron count, sensor configuration and frozen spiking weights for comparisons. Evaluate with no updates or exploration noise, reporting exits as well as episode count and step budget. Small-maze success does not establish success on the default 9x7 maze. Do not select or tune checkpoints using final test seeds.
