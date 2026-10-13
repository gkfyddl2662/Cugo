# Torch50 paired evaluator

`python/eval50.py` compares two saved policies on identical deals with a deterministic, paired seat swap.

For every seed, the evaluator creates two copies of the same initial game. Policy A controls player 0 in the first copy and player 1 in the second; policy B controls the opposite seat. Actions are masked greedy argmax actions, so evaluation consumes no policy RNG. Reported uncertainty uses the paired per-deal score, not two games as independent samples.

Games that are already terminal before the first policy decision (for example an initial Chongtong resolution) are excluded from W/L and confidence statistics. Their two seat-swapped outcomes are policy-independent and would cancel in paired score, while the current Torch environment observation API does not expose their terminal reward directly.

Supported checkpoints are PPO checkpoints from `train_ppo50.py`. `--*-policy ppo` (or `auto`) uses the actor EMA when the checkpoint has one; `ppo_raw` uses the trained actor.

Example:

```powershell
.\.venv-torch\Scripts\python.exe python\eval50.py `
  --a-checkpoint build\checkpoints\ppo_h\iter_001500.pt `
  --b-checkpoint build\checkpoints\ppo_h\iter_001000.pt `
  --pairs 131072
```

For more than two checkpoints, `python/league50.py` plays every pair the same way.
