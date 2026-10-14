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

Weights after a full update differ between graphed and eager runs by as much as two eager runs differ from each other (atomic adds in the backward pass make even eager runs differ).
