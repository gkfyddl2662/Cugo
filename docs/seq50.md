# Event-sequence policies

`python/cugo_seq50.py` reads a decision as a token sequence instead of a single feature vector:

- one opening token: own hand, floor, both captured piles, mission and who moves first;
- one token per committed action so far, from the observer's side: cards played, own drawn cards (the opponent's only as a count), stock flips, cards gained and lost by each side, shake reveals, action kind and actor;
- one query token for the decision at hand.

A pre-norm transformer (default d=256, 4 layers, 8 heads) reads the query position for the 179 logits. The critic is the same network with the opponent's hidden hand added to the query token.

Two variants differ only in the query token:

| Mode | Query token |
| --- | --- |
| `hybrid` | all 720 Torch50 features + legal mask |
| `events` | decision context (pending public cards, floor-choice cards, decision kind, first-player bit, opponent hand count, choice scalars, Gukjin flag), own hand, capture counts and round multiplier + legal mask; the rest of the table must be inferred from the events |

`python/train_seq50.py` trains either variant with the same PPO objective as `train_ppo50.py` (oracle critic, GAE, actor EMA, snapshot pool). Every iteration keeps each game's event log; a decision row stores how many events it saw, so the update rebuilds exactly the sequence the policy acted on. Event tokens attend causally (opening and earlier events only), so the update can run each (game, seat) once with all of its decisions as query tokens (`SeqNet.forward_games`) and get the same outputs as the per-decision pass used while playing; minibatches are whole (game, seat) sequences.

```powershell
.\.venv-torch\Scripts\python.exe python\train_seq50.py --mode events --checkpoint-dir build\checkpoints\seq_events `
  --reference build\checkpoints\hg_init\ppo_h1500_v3.pt
```

Checkpoints carry `config.algorithm = "seq-ppo-v1"`; `eval50.py` and `league50.py` load them next to MLP checkpoints and record event logs only when a sequence policy plays. v3 checkpoints are upgraded on load (`cugo_contract50.py`): the events-mode query layer gains zero-weight columns for its new inputs, together with its Adam moments.

The snapshot pool holds the actor EMA (the release policy), both when it is filled during a run and when `--init` rebuilds it from the checkpoints next to the one being resumed. `--compile` compiles the encoder stack (`SeqNet._encode`) used by the update.

## GPU-resident self-play

`python/cugo_seq_selfplay50.py` plays in waves of `--wave` games (default 4,096). Opening and event tokens only attend backwards, so their per-layer keys/values never change: each (game, seat) keeps a KV cache, every committed event adds one token per seat and every decision runs one query token (`SeqNet.cached_step`); attention for that single query is a Triton decode kernel that reads only the cached slots each row attends (`python/cugo_seq_attention50.py`). All tensors in a decision step have fixed shapes (finished games stay in the batch, masked), so the step - observe, policy query, Gumbel-max sampling, engine step, cache append - is captured once as a CUDA graph and replayed. The host checks every four steps whether the wave has finished and compacts the recorded decisions once per wave. Pool games are the first 20% of each wave; the opponent's weights are copied into a fixed module so the graph stays valid. The query runs only on the moving seat's cache row (the kernel takes a row index per query); event tokens are appended to both seats' rows. A wave allows 64 decision steps (games so far needed at most 43); events are separately checked against the 48 cache slots.

The update reads every minibatch's padding sizes in one device-to-host copy, computes losses with validity masks instead of boolean indexing, runs GAE as a backward scan over decision columns and checks the KL early stop one update late, so its inner loop never waits on the host. `--length-bucket N` sorts each window of N minibatches by game length to cut padding. `--collector eager` keeps the previous per-decision implementation as a reference.

`--update-graphs` replays the minibatch step (both forward passes, backward, gradient clipping and the Adam step) as CUDA graphs. Eagerly the CPU spent longer launching a step's kernels than the GPU spent running them (20 ms vs 15 ms per step with `--compile`). Minibatch shapes are rounded up to a multiple of 4 events and 4 decisions, one graph per shape (7 or 8 in practice), all sharing one memory pool. Each iteration's sequence table, advantages and returns are copied into fixed buffers the graphs read. A new shape is warmed up eagerly and that update is then undone in place, so every minibatch is applied exactly once.

Hybrid, 65,536 games per iteration, RTX 5080 alone:

| Update | Self-play | Update | Iteration | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| eager | 2.5 s | 8.0 s | 10.5 s | 9.4 GB |
| `--compile` | 2.6 s | 6.9 s | 9.5 s | 9.7 GB |
| `--compile --update-graphs` | 2.4 s | 6.0 s | 8.5 s | 11.7 GB |

On a 16 GB card under Windows keep the reserved peak near 11-12 GB (`--memory-fraction 0.80`): at 0.85 the run reserved 13.7 GB, the desktop needs the rest, and once the driver began paging GPU memory to system memory the update slowed from 6 s to 9 s per iteration. `--wave 4096` halves the self-play recording buffers and KV caches (about 0.9 GB) for 0.1-0.3 s more self-play per iteration.

Weights after a full update differ between graphed and eager runs by as much as two eager runs differ from each other (atomic adds in the backward pass make even eager runs differ).

## Training options

All are off by default; checkpoints written with them resume and evaluate like any other (only the actor is ever played).

- `--value-bins N` (with `--value-top`, default 640 scaled points): the critic predicts a distribution over N bins spaced evenly in symlog space and is trained with cross-entropy against a two-hot target that preserves the return's mean exactly; its value is the distribution's mean. Returns are heavy-tailed (median 36 points, 99.9th percentile 8,704, largest seen 48,128), which a single regression output with a smooth L1 loss fits poorly.
- `--critic-stock`: the critic's query also gets the next 8 stock draws (`stock_indexed`), on top of the opponent's hand. With both, the critic sees everything that decides the rest of the deal except the players' future choices.
- `--belief-coef C`: an auxiliary head on the actor's final hidden state predicts the opponent's hand (binary cross-entropy per card, weight C). It is trained, not played; it shapes the shared trunk to track hidden cards.
- `--critic-warmup N`: the first N iterations of an invocation hold the actor (learning rate 0) while the critic and belief head train. Adding stock inputs or bins to a trained critic keeps its trunk (new input columns start at zero, together with their Adam moments) but restarts its output layer.
- `--gostop-baseline`: measures Go/Stop advantages against the points stopping now would settle instead of the critic's value. Measured on 135,906 Go/Stop decisions at iteration 550, the per-sample gradient signal had the same mean and a slightly larger spread (0.153 against 0.150 with the critic: the policy goes 72% of the time, so the critic's value already sits close to the value of going). Kept for experiments, not used.
- `--attention math`: spells attention out so `torch.compile` fuses it; 5% faster in an encoder micro-benchmark but it keeps the score matrices, about 1 GB more at 65,536 games, so it is off.

A/B from the hybrid run's iteration 550, 100 iterations each, same seeds (65,536 games per iteration):

| | Baseline | `--value-bins 151 --critic-stock --belief-coef 0.1 --critic-warmup 10` |
| --- | ---: | ---: |
| vs. `ppo_h1500_v3` at 600 | −55.8 [−60.2, −51.3] | −52.7 [−56.7, −48.6] |
| vs. `ppo_h1500_v3` at 650 | −51.1 [−55.2, −46.9] | −49.5 [−53.6, −45.3] |
| critic explained variance, iterations 601-650 | 0.327 | 0.394 |
| update time per iteration | 6.28 s | 5.64 s |

Head to head at 650 (131,072 deal pairs) the new options scored −0.17 [−1.96, +1.63] points per pair: no difference in strength yet, with a clearly better critic, and the run continues with them.

The MLP trainer (`train_ppo50.py`) takes the same `--value-bins`, `--value-top`, `--critic-stock`, `--belief-coef` (with `--belief-width`, default 512) and `--critic-warmup`. A/B from `ppo_j` at iteration 200 (4×2048 MLP, 65,536 games per iteration), 200 iterations each:

| | Baseline | `--value-bins 151 --critic-stock --belief-coef 0.1 --critic-warmup 10` |
| --- | ---: | ---: |
| vs. `ppo_h1500_v3` at 400 (league) | +2.25 [+0.62, +3.89] | +2.61 [+0.90, +4.32] |
| critic explained variance, iterations 351-400 | 0.329 | 0.364 |
| update time per iteration | 6.58 s | 6.82 s |
| peak reserved memory | 10.9 GB | 12.9 GB |

Head to head at 400 the new options scored −1.35 [−2.84, +0.15] points per pair. Again no strength gain, at 2 GB more memory, so `ppo_j` continues from the baseline arm.

### All-action training (MLP)

`train_ppo50.py --all-actions` branches every legal action of every learner decision one step (`ext.branch_indexed`; 5.6 legal actions per decision on average, 7.9 M children per iteration) and scores each child with the oracle critic. The branch replays the real stock, so the child already holds the flip that action gets: Q(a) = ±V(child), or the final score when the action ends the deal. The policy loss is the clipped surrogate taken in expectation over all actions under the behaviour policy, with A(a) = Q(a) − Σ π_old(b) Q(b), plus the sampled action's (return − Q) term (`--mc-coef`, default 1), which keeps the gradient unbiased whatever the critic's error. Not for `--exploiter`: there the critic only knows the learner's side.

A/B from `ppo_j` at 900, 200 iterations each (league, 131,072 deal pairs):

| | Baseline | `--all-actions` |
| --- | ---: | ---: |
| vs. `ppo_h1500_v3` at 1100 | +3.47 [+1.88, +5.05] | +4.57 [+2.79, +6.36] |
| vs. `ppo_j` at 900 | −0.90 [−2.39, +0.60] | −0.04 [−1.48, +1.40] |
| self-play / update per iteration | 1.4 s / 7.4 s | 7.7 s / 7.5 s |
| explained variance: V(h) / Q(taken action) | 0.33 / – | 0.33 / 0.37 |

Head to head at 1100 the all-action arm scored +0.79 [−0.81, +2.38] points per pair: ahead on every comparison but not significant, at 1.7 times the wall-clock. `ppo_j` continues with it from 1100. Pool snapshots wait in host memory and only the iteration's opponent visits the GPU; 16 snapshots of a 2048×4 actor (140 MB each) would not fit beside the 1.2 GB of all-action buffers.

Branching only actions that can move the policy (`--branch-min-prob 0.01 --branch-max-top 0.99`, the default since 0d9b573) cut self-play from 7.7 s to 3.2 s per iteration (10.9 s in all, against 8.8 s without all-action and 15.3 s branching everything). At `ppo_j`/1100 the policy put ≥ 1% on 1.9 of 5.6 legal actions and half the decisions had a top action ≥ 99%, while an all-action gradient term scales with the action's probability.

`--policy-target improve` (cross-entropy toward π_old·exp(A/τ), τ set so KL(target ‖ π_old) averages `--improve-kl`, every action branched) failed: from `ppo_j`/1300 with `--improve-kl 0.1`, entropy doubled (0.32 to 0.64) within 100 iterations and at 1500 it lost 25.1 [23.1, 27.1] points per pair to the `pg` arm. The critic's Q is too noisy (explained variance ~0.37) for an exponential reweighting: noise lifts low-probability actions it cannot also push down. The `pg` arm itself gained nothing over 1300 in those 200 iterations (+0.02 [−1.42, +1.45]).

### Entropy and learning-rate sweep (MLP)

From `ppo_j`/1600, 150 iterations per arm with `--all-actions`, same seeds; league of 131,072 deal pairs per match:

| Arm | Final entropy | vs. `ppo_j`/1600 | vs. `ppo_h1500_v3` |
| --- | ---: | ---: | ---: |
| entropy 0.02, lr 4e-5 (baseline) | 0.318 | −0.27 [−1.53, +1.00] | +4.78 |
| entropy 0.01 | 0.238 | +0.08 [−1.24, +1.39] | +4.61 |
| entropy 0.005 | 0.195 | −0.50 [−1.83, +0.83] | +4.32 |
| lr 1.5e-5 | 0.322 | −0.14 [−1.37, +1.09] | +5.02 |

Every pair of arms is within ±0.6 points of each other: none of these knobs moves the plateau at this horizon. The reference no longer separates checkpoints (`ppo_j` 1100 through 2000 all score +4.2 to +4.6 against it); head to head, 2000 beats 1600 by +0.46 [−1.06, +1.98], 1100 by +1.33 [−0.38, +3.03] and 600 by +2.66 [+0.99, +4.33], so progress continues at roughly +0.3 per 100 iterations. `ppo_j` continues past 2000 with plain PPO (no all-action, 8.8 s per iteration), since nothing measured favoured the slower variant.

### Gradient signal

Head to head, `ppo_j` at 2000 does not beat 1400 (1400 scores +0.26 [−1.28, +1.80]). To see why, the actor gradient was computed at ratio 1 on independent rollouts of 65,536 games (1.4 M decisions) from `ppo_j`/2000 and compared pairwise (8 rollouts, 28 pairs):

| Policy-gradient estimate | Mean cosine between rollouts |
| --- | ---: |
| GAE(0.95) advantage, oracle critic | 0.027 |
| All-action, critic Q (4 rollouts) | 0.024 |
| All-action, Q from greedy play to the end of each branch, with the (return − Q) term | 0.021 |
| Same, Q term only | 0.034 |

A full iteration's gradient is ~97% noise, and the minibatches of 32,768 decisions are 40 times noisier. Exploiters trained at lr 1e-4 against a fixed greedy target, even the weak `ppo_h1500_v3` and even with Monte-Carlo advantages, lost ground in their own training score (−8.8 to −15 points per game over 350 iterations against `ppo_j`/2000) and were flat at lr 4e-5: a larger step only drifts. Playing branches to the end with shared luck (the cloned state keeps the stock order) did not help, at 3.4 times the self-play cost (21.7 s), so the noise is not the luck of future flips but how much an action's value depends on the hidden cards. The rollout variant was measured and not kept. An exploiter that finds nothing at this signal level says nothing about exploitability.

### Redealt-world Q (`--q-source worlds`)

Since the noise is the hidden cards, Q is averaged over worlds the learner cannot tell apart: at each branched decision the opponent's unrevealed hand and the stock are redealt uniformly among the cards the learner cannot see (cards shown by a shake stay put; `held` follows the new hand) and the stock order is reseeded. A world is kept only if the learner's 720 features and legal actions are bit-identical to the real state's: 96.9% are, the rest (mostly a pending flip that already revealed a stock card) keep the real deal. Each branch is played to the end greedily by both seats, who act on their own information sets, so nothing peeks. Measured at `ppo_j`/2000, 8 rollouts of 65,536 games:

| Q for the all-action term (`--mc-coef 0`) | Self-play per iteration | Batch-gradient cosine |
| --- | ---: | ---: |
| GAE baseline (no all-action) | 1.4 s | 0.007 to 0.027 |
| 4 worlds, play to the end | 93.9 s | 0.18 |
| 2 worlds, 2 more where they disagree on the best action (`--worlds-first 2`) | 73.6 s | 0.28 |
| 4 worlds, one step then the oracle critic (`--world-depth 1`) | 8.5 s | 0.85 |
| same, adaptive | 6.5 s | 0.81 |

With the (return − Q) term (`--mc-coef 1`) the cosine falls back to 0.01: the sampled return reintroduces the noise.

Training, 100 iterations from `ppo_j`/2000 at 32,768 games per iteration (head to head, 131,072 deal pairs):

| Arm | vs. `ppo_j`/2000 | Final entropy |
| --- | ---: | ---: |
| 4 worlds, play to the end | **+1.82 [+0.41, +3.23]** | 0.35 |
| depth 1 + adaptive | −0.74 [−2.35, +0.86] | 0.60 |

Play-to-the-end gained more in 100 iterations (72 minutes) than plain PPO did in the 600 before (1400 to 2000: +0.26 for 1400). Depth 1's high cosine was the critic's consistent bias: its action values were too flat to sharpen the policy and entropy nearly doubled; it lost 3.05 [1.45, 4.66] to the play-to-the-end arm. `worlds_j` continues from 2100 with play-to-the-end and `--worlds-first 2`, evaluated against `ppo_j`/2000 (the old reference no longer separates checkpoints).

### Exploiter

`--exploiter --opponent T --init T --reference T` trains a copy of T against T alone; its margin over T is a lower bound on T's exploitability. Pass `--pool-greedy` so T plays greedy, as it is deployed: against a sampling T (about 8 points weaker) the exploiter learned that weaker policy's leaks and lost 2.3 points to greedy T after 200 iterations. With `--pool-greedy` and lr 4e-5, 200 iterations against `ppo_j` at 900 peaked at +0.56 [−0.71, +1.82] and ended at −0.17 [−1.59, +1.24]: no weakness found at that budget, which is a weak bound, not proof of none.

Learning rates are GPU tensors, so the replayed Adam steps of `--update-graphs` read the current value and the warm-up switch needs no recapture. The optimizer has two groups (actor; critic and belief head) and gradients are clipped per network. Older single-group checkpoints are mapped by parameter name.

The critic's value pre-pass runs in length-sorted batches (1.20 s to 0.93 s per iteration); it is GPU-bound, so capturing it as CUDA graphs gained nothing.
