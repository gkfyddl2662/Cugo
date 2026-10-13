# Evaluation (strength/exploitability) and Search-Augmented Methods for Large Imperfect-Information 2p0s Games, with fit to 2-player Shin Matgo

Context used for the fit judgments: 2-player Matgo is imperfect-information with heavy chance, about 177 masked actions and about 33 decisions per game, and a score-based zero-sum reward. Per-pair score std is about 12 points and typical mean differences are 0.1-1.4 points. The simulator is CUDA-batched at about 700k games/s with a clone/branch API at about 65M branches/s on one RTX 5080. Current evaluation plays seat-swapped pairs on identical deals and reports a 95% CI on the mean paired score.

---

## A1. Approximate exploitability measures for large games (LBR, ABR/ISMCTS-BR, NashConv proxies, BR budgets)

### Takeaway
Exact exploitability is out of reach for full-size games. The standard substitutes are lower bounds from an explicit approximate best response (BR): LBR, a cheap greedy one-step lookahead that needs the policy's action probabilities and Bayesian range tracking, and ABR/ISMCTS-BR, a full RL+search best response trained against the frozen policy. Both give only lower bounds, so a weak BR proves nothing. The landmark search-free systems (DeepNash, Ataraxos) reported no exploitability number for the full game at all; they relied on head-to-head play against humans and bots.

### Cited Findings
**Local Best Response (LBR), Lisý & Bowling, arXiv Dec 2016 / AAAI-17 Computer Poker workshop**
- LBR is a "simple, computationally inexpensive" method that gives an approximate lower bound on best-response value in heads-up no-limit hold'em (HUNL). Using it, the authors found that abstraction-based poker bots were "remarkably poor Nash equilibrium approximations" — [arXiv 1612.07547](https://arxiv.org/abs/1612.07547)
- Mechanics: LBR queries the opponent's strategy for action probabilities and tracks a Bayesian range over the opponent's hands (uniform prior, updated by Bayes' rule after each opponent action, no abstraction). For each candidate action it estimates EV under the assumption that the opponent either folds (probability taken from the range) or checks/calls to showdown, then plays greedily — [ar5iv 1612.07547](https://ar5iv.labs.arxiv.org/html/1612.07547)
- Cost: one LBR hand costs at most about (n|H|+1) times a normal hand, with n bet sizes and |H|=1326. Runs were 10-20 instances per 24-core node in batches of 1000 hands; a batch took 0.5 h (fold/call) to 8 h (56 bet sizes) — [ar5iv 1612.07547](https://ar5iv.labs.arxiv.org/html/1612.07547)
- Results: folding every hand loses 750 mbb/h. Slumbot 2016 was exploitable for 522±50 (fold/call LBR) and 4020±115 (fold/call/pot/all-in). Act1 2016 was exploitable for 407±47 and 2597±140. With 97.5% confidence, every evaluated bot was exploitable by more than 3180 mbb/h, which is more than 4x the cost of always folding — [ar5iv 1612.07547](https://ar5iv.labs.arxiv.org/html/1612.07547)
- Later systems reported LBR as a headline robustness check. ReBeL beat a restricted LBR (forced to call for the first two rounds) by 881±94 mbb/g, and the ReBeL paper cites DeepStack at 383±112 against LBR — [ar5iv 2007.13544](https://ar5iv.labs.arxiv.org/html/2007.13544). Student of Games won about 434±9 mbb/hand against LBR, and the paper says LBR "fails to find an exploit" — [ar5iv 2112.03178](https://ar5iv.labs.arxiv.org/html/2112.03178)

**Approximate exploitability / ISMCTS-BR, Timbers, Bard, Lockhart, Lanctot, Schmid, Burch, Schrittwieser, Hubert, Bowling; arXiv Apr 2020, IJCAI 2022**
- The method freezes the evaluated agent, which turns the game into a single-agent problem, then learns a BR with search plus RL. The paper presents this as a generalization of LBR — [arXiv 2004.09677](https://arxiv.org/pdf/2004.09677v1); [IJCAI 2022](https://www.ijcai.org/proceedings/2022/0484.pdf)
- ISMCTS-BR runs information-set MCTS with PUCB, guided by a policy/value net (losses: value MSE plus cross-entropy against visit counts). Hidden states are sampled from the exact posterior under the opponent's policy (Bayes' rule over reach probabilities), not uniformly. This requires tractable belief spaces; the authors suggest a sample model or stochastic planner as a relaxation — [ar5iv 2004.09677](https://ar5iv.labs.arxiv.org/html/2004.09677)
- BR training budget: 100k learner steps (vs AlphaZero's 800k), on about 128 Cloud TPUv4 actor chips and 4 TPUv2 learner chips per experiment, averaged over the last 5 network updates (about 5M states) — [ar5iv 2004.09677](https://ar5iv.labs.arxiv.org/html/2004.09677)
- Validation where the exact value is known. In limit hold'em, approximate NashConv vs exact was 2270 vs 2330 mbb/h against an always-call bot and 7960 vs 8800 against a random bot. Against CFR+ at 10 iterations the BR recovered 94.9-99.9% of exact NashConv. It exploited Slumbot for 1259 mbb/h in the fcpa abstraction — [ar5iv 2004.09677](https://ar5iv.labs.arxiv.org/html/2004.09677)
- Head-to-head strength and exploitability can disagree. In Go, AlphaZero-MSE beats AlphaZero-TD25 (TD25 wins only 23.25%), yet ISMCTS-BR beats MSE about 20% of the time and TD25 about 40%. A simulation curriculum raised the estimated exploitability to about 90%. Approximate exploitability is "a lower bound on the true value" — [ar5iv 2004.09677](https://ar5iv.labs.arxiv.org/html/2004.09677)

**Search-free superhuman agents are exploitable (perfect-information analogue)**
- Adversarial policies trained against frozen KataGo win >97% at superhuman settings. FAR.AI's summary gives >99% with no tree search and >77% with superhuman search. The adversaries are weak players that win by inducing blunders, transfer to other Go AIs, persisted against adversarially trained KataGo versions, and could be executed by human experts (Wang, Gleave et al., ICML 2023) — [PMLR v202](https://proceedings.mlr.press/v202/wang23g.html); [arXiv 2211.00241](https://arxiv.org/abs/2211.00241); [FAR.AI](https://far.ai/publication/wang2022adversarial/)

**Exact exploitability on mid-size proxy games, and the 2025-2026 methodology debate**
- Rudolph, Lichtlé, …, Kolter, Zhang, Farina, Vinitsky, Sokota, "Reevaluating Policy Gradient Methods for Imperfect-Information Games" (arXiv Feb 2025, v4 May 2026). They compute exact exploitability with sequence-form treeplexes (about 1000x more compact than trees) in 30-90 s per evaluation on commodity hardware, for 5 games with 6M-27M infostates: 2D5F Liar's Dice, 3x3 (Abrupt) Dark Hex, and (Abrupt) Phantom TTT. The code is released as exp-a-spiel with OpenSpiel bindings — [arXiv 2502.08938](https://arxiv.org/html/2502.08938v4)
- That study ran about 7000 runs over 350 configs and more than 345k CPU-hours. Its conclusion: "FP-, DO-, and CFR-based approaches fail to outperform generic policy gradient methods." MMD, PPO, and PPG were roughly on par; ESCHER was uniformly noncompetitive; PSRO was largely noncompetitive; NFSP and R-NaD typically underperformed. The authors argue that head-to-head comparisons are muddled by "the intransitive dynamics" of imperfect-information games, so exact exploitability is preferred where it is feasible — [arXiv 2502.08938](https://arxiv.org/html/2502.08938v4)
- Related BR and counter-strategy work, seen as titles and abstracts only: "Continual Depth-limited Responses for Computing Counter-strategies in Sequential Games" — [arXiv 2112.12594](https://arxiv.org/pdf/2112.12594); "Adapting Beyond the Depth Limit: Counter Strategies in Large Imperfect Information Games" (2025) — [arXiv 2501.10464](https://arxiv.org/pdf/2501.10464); and "Data-Augmented Game Starts…" (2026), which contributes a long-horizon 2p0s benchmark with fast analytic exploitability — [arXiv 2605.14379](https://arxiv.org/pdf/2605.14379)
- DeepNash reports no exploitability number for full Stratego. It states that minimizing exploitability is intractable there because it needs a best-response estimate — [ar5iv 2206.15378](https://ar5iv.labs.arxiv.org/html/2206.15378). Ataraxos (2025) also reports no approximate best response; it relies on head-to-head play — [arXiv 2511.07312](https://arxiv.org/html/2511.07312v2)

### Inferences
- A practical exploitability suite for Matgo, cheapest first:
  - (a) An "LBR-Matgo" probe: track the opponent's hidden hand via Bayes over the frozen policy's action probabilities. The unseen cards are few, so the belief is a distribution over card subsets, which can be sampled. Then pick the action with the best one-step-plus-rollout EV, where the rollouts use a fixed default policy and the GPU branch API supplies the branches.
  - (b) A frozen-target RL best response: train a fresh PPO agent only against the frozen candidate, with a large budget (e.g., ≥10-20% of the main training budget). Report its paired score margin as a lower bound on exploitability.
  - (c) Exact exploitability on a shrunken Matgo variant (fewer months/cards) using exp-a-spiel-style sequence-form computation, to validate (a) and (b).
- Because Matgo uses score-based reward, exploitability should be measured in points per game, consistent with the paired-score metric. A win-rate BR and a score BR may find different holes.
- At 700k games/s, a BR-training budget of billions of games is affordable. The binding constraint is BR training stability, not simulation speed.

### Gaps
- I did not find a published LBR-style method for non-poker card games (e.g., Mahjong, Dou Dizhu), nor guidance on the BR budget needed relative to training budget. ABR only offers "100k steps, average last 5 updates."
- No published Go-Stop/Hwatu/Matgo RL or evaluation paper turned up in searches.

---

## A2. Head-to-head tournaments, Elo/TrueSkill, α-Rank, Nash averaging, population evaluation

### Takeaway
Head-to-head ratings are cheap but can be misleading under intransitivity. Game-theoretic aggregators (Nash averaging, α-Rank) handle redundancy and cycles. Recent papers still report mostly pairwise head-to-head results with CIs, plus a fixed anchor pool.

### Cited Findings
- Nash averaging (Balduzzi, Tuyls, Pérolat, Graepel, NeurIPS 2018) evaluates agents against the maxent Nash of the empirical meta-game. It is invariant to redundant agents/tasks, so weak or duplicated agents cannot skew the ranking, and this makes inclusive evaluation costless — [arXiv 1806.02643](https://arxiv.org/abs/1806.02643); [NeurIPS](https://papers.nips.cc/paper/2018/hash/cdf1035c34ec380218a8cc9a43d438f9-Abstract.html)
- α-Rank (Omidshafiei, Papadimitriou, Piliouras, Tuyls, Rowland, Lespiau, Czarnecki, Lanctot, Pérolat, Munos; arXiv Mar 2019) ranks agents via evolutionary dynamics and Markov-Conley chains, which capture cycles that static Nash misses. It runs in polynomial time in the number of pure strategy profiles and was validated on AlphaGo, AlphaZero, MuJoCo Soccer, and poker — [arXiv 1903.01373](https://arxiv.org/abs/1903.01373)
- Ataraxos calls Elo against a fixed reference "useful but flawed" and reports effective win rate (draw = 0.5) — [arXiv 2511.07312](https://arxiv.org/html/2511.07312v2)
- Exploitability and head-to-head strength are "complementary": a stronger agent head-to-head can be more exploitable — [ar5iv 2004.09677](https://ar5iv.labs.arxiv.org/html/2004.09677)
- Student of Games used round-robin tournaments (200 games per pair, colors swapped) with Elo anchored to Stockfish and GnuGo — [ar5iv 2112.03178](https://ar5iv.labs.arxiv.org/html/2112.03178)

### Inferences
- For Matgo, keep a frozen league: past checkpoints, rule-based bots, a greedy-score bot, and ABR exploiters. Compute the full paired-score payoff matrix (each entry a paired mean ± CI). Then report (i) mean vs the pool, (ii) the minimum over the pool (worst-case proxy), and (iii) Nash-averaging/maxent-Nash or α-Rank over the matrix. The worst-case row entry against the exploiter set is the closest cheap proxy for exploitability.
- Elo assumes transitivity. With a payoff matrix available, Nash averaging is strictly more informative for checking whether new checkpoints are improving or cycling.

### Gaps
- No source compared TrueSkill directly with these methods for 2p0s card games. I did not fetch the 2023 "Evaluating agents using social choice theory" (Lanctot et al.) paper.

---

## A3. Variance reduction for chance-heavy evaluation (duplicate/paired deals, AIVAT/DIVAT, CRN) and sample sizes

### Takeaway
Duplicate/paired deals are the baseline and are already in use here. AIVAT-style control variates, which are unbiased with any heuristic value function, typically cut required samples by 10x or more in poker. The Matgo setup (exact chance model in the simulator, known policy probabilities for one's own agent, a learned value net, and a GPU branch API to compute expectations over chance outcomes) is well suited to AIVAT.

### Cited Findings
- AIVAT (Burch, Schmid, Moravčík, Bowling; arXiv Dec 2016 / AAAI-17 workshop) is a "provably unbiased" low-variance estimator. It reduces variance from both chance events and actions of players with a known strategy. It needs only a subset of agents' strategies and an arbitrary heuristic value estimate, and it can reduce the number of hands needed "by more than a factor of 10" in no-limit poker — [arXiv 1612.06915](https://arxiv.org/abs/1612.06915)
- DeepStack: 44,852 hands against 33 pros. The raw estimate was 492±220 mbb/g; with AIVAT it was 486±40 mbb/g, about an 85% reduction in std — [arXiv 1701.01724](https://arxiv.org/abs/1701.01724); figures from [DeepStack PDF](https://arxiv.org/pdf/1701.01724) via search excerpt
- ReBeL vs Dong Kim over 7,500 hands: +165±69 mbb/g with AIVAT, while the raw score was a loss of 358±188 — [ar5iv 2007.13544](https://ar5iv.labs.arxiv.org/html/2007.13544)
- Student of Games vs Slumbot2019: 7±3 mbb/hand over 3.1M hands with AIVAT — [ar5iv 2112.03178](https://ar5iv.labs.arxiv.org/html/2112.03178)
- Pluribus (Science 2019): 10,000 hands, AIVAT-adjusted 48 mbb/game, SE 25, p=0.028. This comes from a secondary summary and was not verified against the primary source — [greaterwrong summary](https://www.greaterwrong.com/posts/6qtq6KDvj86DXqfp6/let-s-read-superhuman-ai-for-multiplayer-poker)
- Duplicate deals in other card games:
  - DouZero plays every deck twice with the algorithms swapping Landlord/Peasant: 10k decks per pair in the preliminary round, 100k decks repeated 5 times with different decks for the Elo round. It reports no CIs beyond the std of Elo — [ar5iv 2106.06135](https://ar5iv.labs.arxiv.org/html/2106.06135)
  - Ataraxos reports Dou Dizhu role-averaged score over 10,000 duplicated deals as mean ± SE (e.g., 0.199±0.015 vs PerfectDou) — [arXiv 2511.07312](https://arxiv.org/html/2511.07312v2)
- Mahjong without duplication: Suphx says stable rank needs "at least a few thousands of games." Its offline evaluation used 1M games against 3 fixed SL-weak agents (20 K80 GPUs × 2 days), with interquartile ranges from 1000 subsamples of 800K games — [ar5iv 2003.13590](https://ar5iv.labs.arxiv.org/html/2003.13590)

### Inferences (sample-size arithmetic from the user's numbers; not from a source)
- With per-pair std σ≈12, the 95% CI half-width is 1.96·12/√n: about 0.235 points at n=10k pairs, 0.074 at 100k, and 0.024 at 1M.
- To detect a 0.1-point true difference with 80% power (two-sided α=0.05), n ≈ ((1.96+0.84)·12/0.1)² ≈ 113k pairs. A 0.5-point difference needs about 4.5k pairs.
- The simulator can play 1M pairs in seconds. In practice, batched NN inference for the two policies is the bottleneck, so variance reduction matters mainly for (i) expensive evaluations (search-augmented agents, ABR loops) and (ii) many-entry league matrices.
- A Matgo AIVAT sketch: for each chance event (deck draws and flips), add the correction V(state before chance) − V(state after chance), with the expectation computed exactly by branching over the remaining cards via the clone API. For each action of our agent, whose policy is known, add Σ_a π(a)Q(a) − Q(a_taken). Unbiasedness holds for any value net, and a better value net gives lower variance.
- Paired seat-swapped deals with a fixed deck order act as common random numbers. Once play diverges, the same deck order is consumed differently, so the pairing correlation decays over the game. AIVAT removes the remaining chance variance at each draw.
- Report both mean paired score and win rate. The "Score vs Winrate" paper (Pasqualini et al., ICMLA 2022) shows that score-maximizing and win-maximizing policies differ: an outcome-optimal policy is risk-seeking when behind — [arXiv 2201.13176](https://arxiv.org/abs/2201.13176). Matgo go/stop decisions are exactly such risk choices.

### Gaps
- I found no source that applied AIVAT outside poker, and no published per-game variance reduction factors for Mahjong/Dou Dizhu-like games. DIVAT (Zinkevich et al. 2006) and MIVAT were not fetched.

---

## A4. What landmark systems actually reported (DouZero, Suphx, R-NaD/DeepNash, Student of Games, ReBeL, Ataraxos)

### Takeaway
None of the large-game systems reported full-game exploitability. They reported head-to-head play (duplicated deals or swapped roles) against bot pools and humans, plus exact exploitability only on small proxy games (SoG, ReBeL) or LBR in poker. Recent work (Ataraxos 2025) adds mean ± SE over 10k duplicated deals and searched-vs-unsearched comparisons.

### Cited Findings
- **DouZero (ICML 2021):** metrics were WP (win %) and ADP (avg point difference, bombs double). Paired decks with swapped roles; 10k decks per pair in the preliminary round, 100k×5 for Elo. Botzone #1 of 344 bots (Elo 1625.11, Oct 2020). Trained 30 days on 4×1080Ti + 48 CPUs. WP- and ADP-trained agents play differently: ADP agents avoid bombs, WP agents bomb aggressively — [ar5iv 2106.06135](https://ar5iv.labs.arxiv.org/html/2106.06135)
- **Suphx (arXiv Mar 2020):** Tenhou stable rank 8.74 dan over 5,760 games (top humans 7.46). Offline evaluation used 1M games against fixed SL agents. Stable rank has high variance — [ar5iv 2003.13590](https://ar5iv.labs.arxiv.org/html/2003.13590)
- **DeepNash (arXiv Jun 2022, Science 2022):** Gravon play over about 2 weeks gave 42/50 wins (84%) and ranked 3rd in both the 2022 and all-time rankings. Against bots it won 97.1-100% over 800 games per bot. No full-game exploitability; search was rejected because Stratego has about 10^535 states and about 10^66 deployments per player — [ar5iv 2206.15378](https://ar5iv.labs.arxiv.org/html/2206.15378)
- **Student of Games (arXiv Dec 2021; Science Advances 2023):**
  - Exact exploitability on Leduc and small Scotland Yard (min/avg/max over 50 seeds).
  - Elo round robins in chess and Go.
  - HUNL: 7±3 mbb/hand vs Slumbot with AIVAT; 434±9 vs LBR.
  - Scotland Yard: 55% vs PimBot over 2000 matches per point, roles swapped.
  - [ar5iv 2112.03178](https://ar5iv.labs.arxiv.org/html/2112.03178)
- **ReBeL (NeurIPS 2020):** exact exploitability in Liar's Dice and turn endgame hold'em (TEH) (TEH equivalent to about 125 full-game CFR iterations). Head-to-head: Slumbot 45±5, BabyTartanian8 9±4, LBR 881±94, human 165±69 with AIVAT — [ar5iv 2007.13544](https://ar5iv.labs.arxiv.org/html/2007.13544)
- **Ataraxos (Sokota, Vinitsky, Hu, Kolter, Farina; arXiv Nov 2025):**
  - Stratego: 15-1-4 vs Pim Niemeijer (exact binomial p<2.6e-4, with a caveat that adaptation makes games non-i.i.d.); 38-2 at a 2025 World Championship demo; 97-99/100 against benchmark bots, described as "only a sanity check."
  - Barrage: 50-game series against top-4 humans.
  - Hanabi: 10k games per condition, mean ± SE.
  - Dou Dizhu: search vs its own policy +0.107±0.012; vs PerfectDou +0.199±0.015; vs DouZero +0.350±0.016.
  - No ABR — [arXiv 2511.07312](https://arxiv.org/html/2511.07312v2)

### Inferences
- A defensible Matgo report would include:
  - (1) Paired-deal mean score ± 95% CI and win rate against a fixed anchor pool: rule bots, previous best, and published-style baselines.
  - (2) Worst-case margin against trained exploiters (ABR-style), as a lower bound on exploitability.
  - (3) Exact exploitability on a reduced Matgo variant, to compare training algorithms.
  - (4) Optionally an AIVAT-adjusted estimate.
- The Dou Dizhu figures (gains of about 0.1-0.35 points, SE about 0.004-0.016 over 10k duplicated deals) show the scale of differences that published work treats as meaningful in a score-based card game.

### Gaps
- The Science Advances version of SoG and the Libratus papers were not fetched. Libratus evaluation details (120k hands, duplicate-style mirrored hands) are not verified here.

---

## B1. Search methods for imperfect-information games, 2017-2026 (reference for later deployment)

### Takeaway
There are two families.
1. Public-belief-state (PBS) subgame solving: DeepStack 2017, Libratus/Pluribus, ReBeL 2020, SoG 2021/23. It is sound but needs common-knowledge public states and enumeration of all infostates per public state.
2. Non-PBS methods, the 2023-2026 trend: update-equivalence/MMD search (ICLR 2024), critic-based look-ahead on policy-gradient agents (Kubíček et al. 2023-25), knowledge-limited subgame solving without common knowledge (Obscuro, 2025/ICLR 2026), Ataraxos's belief-network + rollout + one-MMD-step search (Nov 2025), learned abstract models (LAMIR, Oct 2025), and test-time RL with implicit gadgets (Aug 2026).

The second family is designed for exactly the setting of a strong search-free policy plus cheap test-time refinement.

### Cited Findings
**PBS family**
- DeepStack (Moravčík et al., arXiv Jan 2017, Science 2017) combines "recursive reasoning", decomposition, and a learned intuition (counterfactual value nets) via continual re-solving. It beat pros over 44k hands and is "theoretically sound" and harder to exploit than prior approaches — [arXiv 1701.01724](https://arxiv.org/abs/1701.01724)
- ReBeL (Brown, Bakhtin, Lerer, Gong; arXiv Jul 2020, NeurIPS 2020):
  - A PBS is the public observation sequence plus a joint distribution over each player's infostates. Rules and all policies, including the search algorithm, are assumed common knowledge.
  - The value/policy input "grows linearly with the number of infostates in a public state," which is "intractable in games such as Recon Chess."
  - Guarantees cover 2p0s only. Poker needs a ≤9-action abstraction.
  - Data generation used 90 DGX-1s; play takes <2 s/hand.
  - [ar5iv 2007.13544](https://ar5iv.labs.arxiv.org/html/2007.13544)
- Student of Games (Schmid et al., arXiv Dec 2021; Science Advances 2023):
  - Uses growing-tree CFR (GT-CFR) over a public tree, a counterfactual value-and-policy network (CVPN) on PBSs, and sound continual re-solving.
  - Enumerating infostates per public state can be "prohibitively expensive." Poker actions were abstracted from about 20,000 to 4-5. Runtime is quadratic in GT-CFR iterations.
  - Exploitability bound is (5D+2)(Fε + NU√(A/T)). Compute was similar to AlphaZero (3500 TPUv4 actors).
  - [ar5iv 2112.03178](https://ar5iv.labs.arxiv.org/html/2112.03178)
- DeepNash rejected search because Stratego "computationally challenges all existing search techniques" — [ar5iv 2206.15378](https://ar5iv.labs.arxiv.org/html/2206.15378)

**Non-PBS / decision-time planning family (2023-2026)**
- Update-Equivalence Framework (Sokota, Farina, et al.; arXiv Apr 2023, ICLR 2024):
  - Planning imitates one update step of a last-iterate learning algorithm instead of solving a subgame, which avoids reliance on public information.
  - Mirror descent is used for cooperative games and magnetic mirror descent (MMD) for adversarial games.
  - In Hanabi it matches or beats public-belief search with about 100x less search time. It improved results in 3x3 Abrupt Dark Hex and Phantom TTT.
  - [arXiv 2304.13138](https://arxiv.org/html/2304.13138v3); [ICLR 2024 poster](https://ICLR.cc/virtual/2024/poster/18938); [project page](https://www.mit.edu/~gfarina/2024/dtp_iclr24)
- Look-ahead Search on Top of Policy Networks (Kubíček, Burch, Lisý; arXiv Dec 2023, v3 Jan 2025; listed as IJCAI 2024 by a search snippet):
  - Trains a critic that estimates values under policy transformations, used for depth-limited search and to start search at any decision point.
  - Needs "no search during train time." Built on R-NaD.
  - Tested on Leduc, imperfect-information Goofspiel, and Battleships.
  - [arXiv 2312.15220](https://arxiv.org/abs/2312.15220)
- Obscuro (Brian Hu Zhang & Tuomas Sandholm; arXiv Jun 2025, v2 Feb 2026; ICLR 2026 as "General search techniques without common knowledge for imperfect-information games, and application to superhuman Fog of War chess"):
  - The first superhuman Fog-of-War chess AI, "significantly stronger" than the prior SOTA and top humans.
  - Described as the largest turn-based zero-sum game, by amount of imperfect information, where superhuman play has been achieved.
  - [arXiv 2506.01242](https://arxiv.org/abs/2506.01242); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9705eec26e511d7844a591ce10c33a3f-Abstract-Conference.html)
  - Note: Obscuro is FoW chess, not Stratego. The Stratego system is Ataraxos.
- Ataraxos, "Superhuman AI for Stratego Using Self-Play RL and Test-Time Search" / "Scalable Decision Making for Games of Imperfect Information" (Sokota, Vinitsky, Hu, Kolter, Farina; arXiv Nov 2025):
  - Training: self-play RL with a reverse-KL magnet, advantage filtering, and annealed regularization.
  - Belief: a separate autoregressive transformer belief network, trained on final-policy self-play with teacher forcing and dropout for robustness to human opponents.
  - Test-time search: sample hidden states from the belief net, run depth-limited rollouts with the policy for both sides, average value predictions per candidate move, then apply one tabular MMD step at the root with the training regularizers and sample from the result.
  - No PBS or subgame-solving machinery. Configuration was 40 ply and 1000 rollouts, about 1.26 s/move on an H100.
  - Search gain is "ultimately bounded" (it mimics one update). Without the reverse-KL term, search overfits the policy net's idiosyncrasies and falls below the raw policy.
  - Training: 16 H100 × 1 week (163M games, 208B env steps), belief net 4 H100 × 4 days, "<$8,000."
  - [arXiv 2511.07312](https://arxiv.org/html/2511.07312v2)
- LAMIR (Kubíček & Lisý; arXiv Oct 2025, cited as ICLR 2026 in a citing paper) learns an abstracted game model from interaction, keeping subgames small enough for principled look-ahead for pre-trained agents, including in large games — [arXiv 2510.05048](https://arxiv.org/abs/2510.05048)
- Test-time Reinforcement Learning in Imperfect Information Games (Kubíček, Lisý, Sandholm; arXiv 31 Aug 2026):
  - Extends gadget games to RL by representing them implicitly through modified sampling plus a neural policy, which removes subgame-size limits.
  - Proves that regularized policy-gradient methods bound strategy degradation from test-time reasoning, even without gadgets.
  - Extra test-time training "often" improves on the blueprint.
  - [arXiv 2608.30635](https://arxiv.org/abs/2608.30635)
- MAPLE (Li, Guei, Wu, Wu; arXiv May 2026, IEEE CoG 2026) is an AlphaZero-style tree that aggregates evaluations over several sampled worlds, combining PIMC's and IS-MCTS's strengths at controlled cost. It gained +291 Elo in Phantom Go and +136 in Dark Hex over PIMC-AlphaZero — [arXiv 2605.24139](https://arxiv.org/abs/2605.24139)

### Inferences
- The field has moved toward "strong regularized policy plus belief sampling plus a regularized one-step improvement at the root" (Ataraxos, update-equivalence, test-time RL). That matches a "search-free net first, search later" roadmap and does not require PBS machinery.

### Gaps
- Obscuro's algorithm details (knowledge-limited subgame solving variant, human-match statistics) were not retrieved beyond the abstract. Neither were the games and exploitability results of the 2026 test-time RL paper.

---

## B2. Feasibility for ~33-step chance-heavy Matgo; cheap alternatives (PIMC/IS-MCTS, Suphx-style policy adaptation, search only at go/stop)

### Takeaway
Full PBS solving (ReBeL/SoG) is probably feasible in principle for 2-player Matgo, since private info is a hand drawn from a small deck. But it is engineering-heavy, and the GPU's branch throughput suits the cheaper Ataraxos/MMD-style root improvement better. PIMC/IS-MCTS is simple but suffers strategy fusion and non-locality. Suphx's runtime adaptation gave large but expensive gains. Search restricted to high-stakes decisions such as go/stop is the lowest-risk first step.

### Cited Findings
- PIMC samples hidden states, solves each as perfect-information, and averages. Its known failure modes are strategy fusion, non-locality, and budget splitting across trees (Cowling et al.). Long, Sturtevant, Buro, Furtak analyzed which game properties make PIMC strong or weak in practice — [Sturtevant lecture](https://www.cs.du.edu/~sturtevant/w13-games/Lecture10.pdf); [Cowling thesis](https://etheses.whiterose.ac.uk/id/eprint/8117/1/Feb%2016%20-%20FINAL.pdf); [MCTS review arXiv 2103.04931](https://arxiv.org/pdf/2103.04931)
- IS-MCTS builds one tree over information sets to address strategy fusion. αμ (Cazenave et al.) and Extended PIMC also target these issues — [Cowling thesis](https://etheses.whiterose.ac.uk/id/eprint/8117/1/Feb%2016%20-%20FINAL.pdf); [Optimizing αμ](https://www.lamsade.dauphine.fr/~cazenave/papers/OptimizingAlphaMuCoG2021.pdf)
- IS-MCTS is "costly when combined with neural networks," while PIMC suffers strategy fusion (motivation for MAPLE, 2026) — [arXiv 2605.24139](https://arxiv.org/abs/2605.24139)
- Suphx run-time policy adaptation (pMCPA):
  - Per round, fix the hand, simulate 100K trajectories, and fine-tune with policy gradient, restarting from the offline policy each round.
  - The adapted RL-2 beat the non-adapted version in 66% of games. It was tested on only "hundreds" of rounds because it is time-consuming.
  - Gains concentrated in the last 1-2 rounds. Suphx did not use it on Tenhou because of time limits.
  - [ar5iv 2003.13590](https://ar5iv.labs.arxiv.org/html/2003.13590)
- Suphx's oracle guiding (train with perfect-info features, then drop them out) and GRU global reward prediction are training-side tricks — [ar5iv 2003.13590](https://ar5iv.labs.arxiv.org/html/2003.13590)
- Ataraxos search in Dou Dizhu (a chance/hidden-hand card game) added +0.107±0.012 role-averaged score over its own policy on 10k duplicated deals — [arXiv 2511.07312](https://arxiv.org/html/2511.07312v2)
- ISMCTS-BR shows the same IS-MCTS + exact-posterior machinery can be repurposed as an evaluator (exploiter) rather than a player — [ar5iv 2004.09677](https://ar5iv.labs.arxiv.org/html/2004.09677)

### Inferences (judgments for Matgo; not from sources)
- **PBS solving.** In 2-player Matgo, the opponent's hidden hand comes from the cards not seen by the player (opponent hand plus draw pile). Infostates per public state are the combinations of the remaining unseen cards, which can be large early (choose ~10 from ~20+) and shrink later. ReBeL/SoG inputs scale linearly with this count, and they also need a public-tree CFR solver over ~177 actions with chance nodes at every draw. This is feasible but heavy. Restricting PBS solving to late-game endgames, where unseen sets are small, is the practical variant (cf. Libratus/DeepStack endgame solving).
- **Ataraxos-style root MMD step.** A belief net over the opponent's hidden cards (or exact Bayes over policy reach probabilities, as in ISMCTS-BR/LBR, since the card space is small) feeds K sampled worlds. Each candidate action gets a depth-limited rollout with the policy for both sides plus a value bootstrap. A KL-regularized MMD update to the root policy follows. With a 65M branches/s clone API, thousands of rollouts per decision are cheap. The key lesson to carry over is the regularization toward the training magnet, which keeps search from overfitting value-net errors.
- **Go/stop-only search.** Go/stop is a binary, high-variance, score-multiplier decision. It suits exact-ish evaluation: branch both options and roll out the remainder under sampled worlds. It also suits the risk-sensitivity point from the score-vs-winrate literature [arXiv 2201.13176](https://arxiv.org/abs/2201.13176). It is the cheapest place to measure search gain in the paired-deal harness.
- **PIMC/IS-MCTS.** These could be useful baselines and exploiters. Strategy fusion matters in Matgo wherever information-gathering or hiding matters (e.g., which matching card to capture, which reveals information). PIMC also lets the searcher "see" the deck order in each world, which biases toward overly optimistic captures.
- **Evaluating any search agent.** Use the same paired-deal harness and report search vs policy on identical deals, as Ataraxos did for Dou Dizhu. Also re-run the ABR exploiter against the search agent: search can raise head-to-head strength while the paper notes its gain is bounded and can fall below the raw policy if unregularized.

### Gaps
- No primary-source quantitative comparison of PIMC vs IS-MCTS vs MMD-root-search was found for a Hanafuda/Hwatu-family game. No fetched source states an exact Libratus/Pluribus depth-limited search procedure; only DeepStack's and ReBeL's high-level descriptions were verified.
