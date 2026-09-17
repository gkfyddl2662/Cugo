# Torch50 paired evaluator

`python/eval50.py` compares two saved policies on identical deals with a deterministic, paired seat swap.

For every seed, the evaluator creates two copies of the same initial game. Policy A controls player 0 in the first copy and player 1 in the second; policy B controls the opposite seat. Actions are masked greedy argmax actions, so evaluation consumes no policy RNG. Reported uncertainty uses the paired per-deal score, not two games as independent samples.

Games that are already terminal before the first policy decision (for example an initial Chongtong resolution) are excluded from W/L and confidence statistics. Their two seat-swapped outcomes are policy-independent and would cancel in paired score, while the current Torch environment observation API does not expose their terminal reward directly.

Supported checkpoints:

- legacy `train50.py`: `--*-policy legacy` or `auto`
- NFSP BR network: `--*-policy br`
- NFSP historical average network: `--*-policy average` or `auto`

Example:

```powershell
.\.venv-torch\Scripts\python.exe python\eval50.py `
  --a-checkpoint build\checkpoints\nfsp16\iter_000010.pt `
  --a-policy average `
  --b-checkpoint build\checkpoints\nfsp64\iter_000010.pt `
  --b-policy average `
  --pairs 131072
```

`python/eval50_smoke.py` checks two invariants: a policy against itself must cancel exactly under seat swap, and an NFSP average policy initialized from the BR policy must also cancel exactly before training.
