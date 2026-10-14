# Cugo

**GPU-resident self-play for Hangame Shin Matgo** — a rules-exact two-player
Go-Stop engine in C++20/CUDA, a batched PyTorch environment built on it, and the
reinforcement-learning stack that trains Shin Matgo agents with it.

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)
![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C)

## Highlights

- **Faithful to the Hangame client.** The 50-card Shin Matgo rules follow the
  client's own rule server: the deal and floor refill, ppuk chains, last-turn
  exceptions, automatic pi steals, Gukjin prompts, missions and the full settlement.
  80,000 games recorded from that server replay move-for-move and point-for-point
  identically (`tools/hg_replay50.cpp`).
- **One engine, CPU and GPU.** Header-only C++20 that compiles for the host and for
  CUDA. A CPU/GPU differential test suite checks every layer, from card masks to
  whole games.
- **Self-play without leaving the GPU.** Game states live in device memory. A PyTorch
  CUDA extension observes, steps, clones and branches them in batches: about 100k
  complete games/s with a small policy, and over 100M state branches/s on one RTX 5080.
- **Two kinds of policy.**
  - Residual-MLP PPO over a 720-feature observation, with an oracle critic.
  - Event-sequence transformers that read the game as a token stream, Mortal-style.
    Their self-play runs on a KV cache inside CUDA graphs, with a Triton decode
    kernel.
- **Honest evaluation.** Paired, seat-swapped matches on identical deals; league
  tables; a live training dashboard with a per-reference checkpoint ranking.

## Status

As of October 2026:

| Policy | Model | vs. `ppo_h1500_v3` (points per deal pair) |
| --- | --- | --- |
| `ppo_h1500_v3` | 3×512 residual MLP, PPO | reference |
| `worlds_j` (in training from 2100) | 4×2048 residual MLP, PPO with redealt-world Q | +1.8 over `ppo_j`/2000 head to head |
| `ppo_j` (stopped at 2000) | 4×2048 residual MLP, PPO | +4.2 |
| `seq_hybrid` (paused at iteration 3000) | 4-layer transformer, events + 720 features | −3.8 |
| `seq_events` (retired at iteration 300) | 4-layer transformer, events only | −218 |

League of 131,072 deal pairs per match. Head to head, `ppo_j` at 400 beats
`seq_hybrid` at 3000 by 5.2 points per pair [3.3, 7.1]. The sequence model, trained
from scratch, closed most of its gap (−49 at 650) and is paused while the MLP
trains; the events-only variant plateaued well below the hybrid one.

## Requirements

- An NVIDIA GPU, and the CUDA Toolkit 13.x for the CUDA targets. The CPU tests build
  without CUDA.
- CMake 3.24 or later, Ninja, and a C++20 compiler (recent MSVC, GCC or Clang).
- Python 3.12+ (developed on 3.14) with PyTorch 2.14 (`cu132` wheels), for training and evaluation.
- Optional: Triton (`triton-windows` on Windows), for the sequence-model decode
  kernel. Without it, a plain matrix-product path is used.

## Quick start

Build and run the engine tests:

```bash
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON -DCUGO_ENABLE_CUDA=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows, `scripts/windows_bootstrap.ps1` finds the MSVC/CMake/Ninja toolchain,
builds and runs the suite.

Set up Python. The CUDA extension compiles on first use, into
`build/torch_extensions/`:

```bash
python -m venv .venv-torch
.venv-torch/Scripts/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
.venv-torch/Scripts/python python/torch50_smoke.py --batch 32768 --warmup 1 --iters 3
```

Train an MLP policy with PPO self-play, or a sequence policy:

```bash
python python/train_ppo50.py --checkpoint-dir build/checkpoints/my_run \
    --reference build/checkpoints/hg_init/ppo_h1500_v3.pt
python python/train_seq50.py --mode hybrid --checkpoint-dir build/checkpoints/seq_hybrid \
    --wave 4096 --length-bucket 8 --reference build/checkpoints/hg_init/ppo_h1500_v3.pt
```

To continue an MLP run, pass `train_ppo50.py --resume <checkpoint>`. The sequence
trainer resumes with `--init <checkpoint>`.

Evaluate and compare checkpoints:

```bash
python python/eval50.py --a-checkpoint A.pt --b-checkpoint B.pt --pairs 131072
python python/league50.py A.pt B.pt C.pt --pairs 65536
python python/dashboard50.py        # http://127.0.0.1:8765 (training, ranking, evals)
```

Run the Python scripts with `PYTHONPATH=python`, or from inside `python/`.

## How it fits together

```
include/cugo/core      cards, masks, RNG
include/cugo/game      deal → turn → game → special → terminal → settlement
                       rollout / policy / post-chance decisions
                       torch50.h: 720-feature observation, 179-action space, event log
python/csrc            CUDA extension: create / observe / step / clone / branch / events
python/                PPO (MLP), sequence PPO, self-play, evaluation, dashboard
tests/                 CPU tests and CPU/GPU differential tests
tools/                 hg_replay50: replays recorded Hangame server games
```

A decision is one of: a primary action (play, bonus, shake, bomb, Go/Stop,
Chongtong), a floor choice after a stock flip, or a Gukjin prompt. All of them share
one 179-way action head with a legal-action mask.

## Documentation

| Topic | Document |
| --- | --- |
| Rules taken from the Hangame client | [docs/hangame50.md](docs/hangame50.md) |
| Rules pinned from the public guide, card ids | [docs/rules.md](docs/rules.md) |
| Special plays, terminal states, last turns, settlement | [special50](docs/special50.md), [terminal50](docs/terminal50.md), [last_card50](docs/last_card50.md), [settlement50](docs/settlement50.md) |
| Rollouts, policy actions, post-chance decisions | [rollout50](docs/rollout50.md), [policy50](docs/policy50.md), [postchance50](docs/postchance50.md) |
| Observation and action contract (v4) | [docs/torch50.md](docs/torch50.md) |
| CUDA extension | [docs/torch_extension.md](docs/torch_extension.md) |
| Event-sequence policies and GPU-resident self-play | [docs/seq50.md](docs/seq50.md) |
| Paired evaluation | [docs/eval50.md](docs/eval50.md) |
| Survey of training methods for imperfect-information games (Korean) | [reports/](reports/) |

## Scope

Cugo is an independent research project. It is not affiliated with or endorsed by
NHN or Hangame.

- The rule details were read from the game's offline rule server. The repository
  holds no client code or assets.
- Cugo does not connect to, read from or automate the live game client.
- Money caps, fees and pushing stakes between games are not modelled.

## License

No license has been chosen yet; all rights reserved by the author.
