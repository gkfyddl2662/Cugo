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
| `events` | decision context only (pending public cards, selected pi, decision kind, choice scalars, Gukjin flag) + legal mask; the table state must be inferred from the events |

`python/train_seq50.py` trains either variant with the same PPO objective as `train_ppo50.py` (oracle critic, GAE, actor EMA, snapshot pool). Every iteration keeps each game's event log; a decision row stores how many events it saw, so the update rebuilds exactly the sequence the policy acted on. Event tokens attend causally (opening and earlier events only), so the update can run each (game, seat) once with all of its decisions as query tokens (`SeqNet.forward_games`) and get the same outputs as the per-decision pass used while playing; minibatches are whole (game, seat) sequences.

```powershell
.\.venv-torch\Scripts\python.exe python\train_seq50.py --mode events --checkpoint-dir build\checkpoints\seq_events `
  --reference build\checkpoints\hg_init\ppo_h1500_v3.pt
```

Checkpoints carry `config.algorithm = "seq-ppo-v1"`; `eval50.py` and `league50.py` load them next to MLP checkpoints and record event logs only when a sequence policy plays.
