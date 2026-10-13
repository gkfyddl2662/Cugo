# Real-world AI systems for imperfect-information card and tile games: case studies and transferable training techniques for 2-player Matgo (through Oct 2026)

Project lens: 2-player Shin Matgo; hidden opponent hand and stock; heavy chance; go/stop multiplier decisions; shake/bomb; ~177 masked actions; ~33 decisions per game; score-based zero-sum reward; CUDA batch simulator at ~700k games/s on one RTX 5080; goal is the strongest search-free policy network.

## Q1. Hanafuda / Koi-Koi and Hwatu / Go-Stop / Matgo AI: academic, Korean, Japanese, GitHub, commercial

### Takeaway
I searched in English, Korean and Japanese. I found no peer-reviewed paper, KCI/DBpia/RISS article or thesis on Go-Stop or Matgo AI. The only Go-Stop RL artifact I found is a small 2026 hobby GitHub project. The method behind commercial Matgo bots (Hangame, Netmarble, Pmang) is not public. The closest real work is Koi-Koi, which uses the same 48-card hanafuda deck and the same continue-or-stop mechanic. The state of the art there is a 2023 IEEE TAI paper: a Transformer over 48 card tokens, trained by Deep Monte-Carlo self-play with a "phased round reward". It reached roughly experienced-human level after about 10M games on one P100 GPU.

### Cited Findings
**Go-Stop / Matgo (Korean)**
- English search for "Go-Stop Matgo reinforcement learning" returned no game-specific paper. Wikipedia confirms that 2-player Go-Stop is called Matgo (맞고) — [Wikipedia Go-Stop](https://en.wikipedia.org/wiki/Go-Stop)
- Korean searches (맞고 인공지능 강화학습 논문; 화투 인공지능 고스톱 불완전정보 KCI; 고스톱 게임 인공지능 몬테카를로 학위논문) returned only generic Korean game-RL papers. Examples are Othello RL (2006) and RL-vs-human playstyle similarity (2020). None was about Hwatu or Go-Stop — [KCI 강화학습 기반의 지능형 게임에 관한 연구](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001033213); [DBpia 사람과 강화학습 인공지능의 게임플레이 유사도 측정](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10510514)
- The only Go-Stop-related Korean IP that surfaced is online Go-Stop *service* patents, with no AI method — [KR20060083609A](https://patents.google.com/patent/KR20060083609A/ko); [KR100742129B1](https://patents.google.com/patent/KR100742129B1/ko)
- Searching for Hangame/commercial Matgo AI found no technical disclosure of how commercial Matgo bots work — search returned only generic game-AI articles, e.g. [KOITA webzine](http://webzine.koita.or.kr/202105-specialissue/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%EC%97%90-%EC%97%B4%EA%B4%91%ED%95%98%EB%8A%94-%EC%9C%A0%EC%A0%80%EB%93%A4-%E2%80%A6-%EA%B2%8C%EC%9E%84%EC%97%85%EA%B3%84-%ED%8C%90%EB%8F%84-%EB%B0%94%EA%BE%BC-AI)
- **GitHub `yubyunghun/hwatu-gostop-rl` (2026 copyright footer; commit dates not visible).** Details from the README:
  - It is a 2-player Go-Stop engine in pure Python, with a flat `Discrete(51)` action space (card ids 0–47 plus skip/go/stop) and a legality mask.
  - The observation is an egocentric vector of about 250 dims. The opponent's hand is never encoded.
  - The reward is the signed final settlement differential, scaled; nagari (draw) scores 0. There is optional potential-based shaping on score difference.
  - Training uses sb3-contrib MaskablePPO, behavior cloning of a heuristic, and distillation of a determinized-search teacher. — [GitHub hwatu-gostop-rl](https://github.com/yubyunghun/hwatu-gostop-rl)
- Results from the same repo:
  - PPO from scratch (~650k steps) won only 21.5% vs the heuristic, with paired payoff about −5.3.
  - Clone + PPO fine-tune reached 38.5% (+3.39). Distilled search reached 42.9% (+3.51). Card-play search reached +2.40, and adding go/stop search reached +5.32.
  - The author says the PPO fine-tune's gain "comes mostly from pressing Go nearly every time", which exploits the project's payoff rules.
  - All numbers were measured before a late capture-rule fix. — [GitHub hwatu-gostop-rl](https://github.com/yubyunghun/hwatu-gostop-rl)

**Koi-Koi (Japanese hanafuda; same 48-card deck, same "continue or stop" decision)**
- **Guan et al., "Learning to Play Koi-Koi Hanafuda Card Games with Transformers", IEEE Trans. AI 4(6):1449–1460, 2023 (DOI 10.1109/TAI.2023.3240674). Authors are from iFLYTEK / Beihang and others.** Open code, models and dataset are on GitHub — [GitHub KoiKoi-AI](https://github.com/guansanghai/KoiKoi-AI); [IEEE](https://ieeexplore.ieee.org/iel7/9078688/9184921/10032777.pdf). Details below are from the paper PDF in that repo:
  - **Input encoding ("deck tokens + special tokens").** There are 48 card columns, each a ~300-dim 0/1 + scalar vector with three parts:
    - (1) Card State: position (own hand / field / both acquired piles / unseen) plus per-turn historical actions (discard, draw, pair, collect, koi-koi claim).
    - (2) Card Attribute: suit, category, and which yaku the card can form.
    - (3) Game Progress: public scalars broadcast to every column (points, card counts, collected counts per category, dealer, round, turn).
    - Two extra special tokens [K0]/[K1] work like [CLS] for the koi-koi (continue/stop) decision.
  - **Network.** Two position-wise FF layers produce a card embedding (256 dims). This feeds a Transformer encoder with 2 blocks, 4 heads, FF dim 512, post-LN, then a position-wise linear output of 1 dim per token.
    - Outputs are read only at legal card columns, so masking comes for free.
    - There are three separate models with the same architecture for discard, pick and koi-koi; the authors note sharing is feasible.
  - **Algorithm.** A DQN-style network trained by self-play Monte-Carlo learning, explicitly "deep Monte-Carlo introduced in [DouZero]". The target is the episode reward (λ=1, no bootstrapping) with an MSE loss.
    - Exploration is ε-greedy, decayed 0.15→0.02.
    - Each episode is one round.
    - Phase 1 trains against a supervised-learning agent until it is beaten; phase 2 is self-play.
  - **Reward ("phased round reward").** The authors compared two variants:
    - RL-Point: the round's point gain or loss.
    - RL-WP: the estimated probability of winning the 8-round game given post-round points and dealer. A per-round logistic-regression win-probability predictor is fitted on greedy self-play and refreshed periodically.
  - **Compute.** A 12-core Xeon CPU (simulation) plus one Tesla P100 (training), "over 10 million games in one week". 2,500 games per loop, batch 256, Adam with lr 1e-4.
  - **Supervised baseline.** Trained on only 200 human 8-round games (21,435 samples), with 75.9% overall action accuracy (koi-koi decision 80.2%).
  - **Strength.**
    - RL-Point vs the author (an experienced player): 53% wins, +2.02 avg points over 50 games. RL-WP vs the author: 47%, +1.04.
    - RL-WP vs RL-Point: 51.5% wins but −2.88 points.
    - RL-WP beats the SL agent at 63.6% (+3.44) and beats random at 99.2%.
    - The authors say RL-WP "always gets a higher winning rate" and RL-Point "always obtains more points". So the reward choice changes style: WP plays safe, cheap yaku to protect a lead late in the game.
  - **Self-assessed limitations.** The logistic-regression WP model is too weak. Training speed and sample efficiency are limited by the attention model plus MC learning.
- **Prior Koi-Koi AI cited by Guan et al.:**
  - UCT-based MCTS (Takaoka et al., IEEE/SII 2017), with no baseline.
  - Expectimax-style adversarial search (Cruz et al., ACM SAC 2015): 56.7% vs random, 50.8% vs greedy.
  - Sato & Ikeda (GPW 2017): 3-layer FC, policy gradient vs Q-learning, +0.5 and −0.3 points per round vs a rule-based bot. — [Guan et al. PDF in repo](https://github.com/guansanghai/KoiKoi-AI)
- **Sato, Uehara, Ikeda (JAIST), IPSJ SIG-GI Technical Report 2017-GI-38 No.6, July 2017.** They used policy gradient with a linear model over hand-designed high-level features, which beat random and rule-based players. Preliminary ANN experiments used low-level features to predict final score — [J-GLOBAL](https://jglobal.jst.go.jp/detail?JGLOBAL_ID=201702231115811024); [JAIST repository](https://dspace.jaist.ac.jp/dspace/handle/10119/15102?mode=full)
- **CyberAgent AI Lab blog (Aug 27, 2019).** Chance-sampling CFR (5M iterations) was run on a "mini hanafuda" with 9 cards (3 months × 3 cards) and 2-card hands. It beat random and a rule-based player that never koi-koi's.
  - The author says full Koi-Koi has >10^16 information sets from the initial deal alone, which is too large for CFR/CFR+ (~10^14) without abstraction — [CyberAgent AI Lab](https://cyberagent.ai/blog/research/2522/)
- A Japanese Qiita hobby article reports a Monte-Carlo "Hard CPU" winning about 60.3% vs a "Normal CPU". The article also notes that commercial hanafuda CPUs often rely on cheating or deal manipulation for difficulty — [Qiita suzukiplan](https://qiita.com/suzukiplan/items/bea9ef05822fc5570541)
- A CiNii record, "モンテカルロ法の報酬の変更による花札「こいこい」の戦略構築" (Koi-Koi strategy construction by changing the reward in Monte Carlo methods), was found in search. Its details page did not load — [CiNii](https://cir.nii.ac.jp/crid/1050574047121432192)

### Inferences
- Koi-Koi is the closest published analogue to Matgo: same deck, discard → (choose pair) → flip-from-stock → (choose pair) → continue/stop flow, and a multiplier-like koi-koi bonus.
  - Guan et al.'s card-token encoding and decision split (discard / pick / continue-stop) map almost 1:1 onto Matgo's discard / choose-match / flip-choose / go-stop / shake / bomb decisions.
  - Their decision-flow figure corresponds to Matgo's flow minus shake/bomb/ppeok specifics.
- Guan et al. used ~10M games per week. Our simulator produces 10M games in ~15 s, a ~40,000× throughput advantage. This makes the high-variance-but-unbiased MC/DMC style they used far more viable. The binding constraint becomes network forward/backward cost, not simulation.
- Their RL-Point vs RL-WP result is the cleanest published evidence on raw-score vs win-probability reward for a hanafuda continue/stop game: point reward → more points, WP reward → more wins.
  - Shin Matgo's objective is score-based zero-sum (expected points). Raw score difference is therefore the "correct" objective. A WP-style transform is only needed if the evaluation metric is win rate or there is a session or bankroll context.
- The hwatu-gostop-rl result has two warnings. First, from-scratch PPO at small scale (650k steps) loses badly to a heuristic. Second, a learned policy can degenerate to "always Go" if the payoff rules (e.g., go-bak / reverse-go penalties, which make a player who calls Go and then loses pay extra) are not implemented faithfully. Rule-fidelity tests of the go/stop payoff are therefore critical.
- The CyberAgent estimate (>10^16 infosets) suggests tabular CFR on full Matgo is not feasible, which supports the project's choice of search-free neural self-play.

### Gaps
- I found no Matgo/Go-Stop paper in KCI, DBpia or RISS via web search. Direct database queries (e.g., RISS full-text search for "고스톱", "맞고", "화투") were not possible through the tools available. A Korean thesis might exist without being indexed by web search.
- I found no public information on the algorithms used by Hangame, Netmarble, Pmang or other commercial Matgo bots. The Qiita article suggests commercial hanafuda CPUs may "cheat", but that is unverified for Korean products.
- The full text of Sato & Ikeda GPW 2017 and the CiNii Monte-Carlo-reward Koi-Koi paper was not retrieved.
- Training dates and commit history of hwatu-gostop-rl were not visible.

## Q2. DouZero (2021) and successors (DouZero+, PerfectDou, AlphaDou, 2026 VRPO): why DMC worked and the role of oracle critics

### Takeaway
DouZero (ICML 2021) showed that plain Deep Monte-Carlo can beat all prior DouDizhu AIs within days on one 4-GPU server:
- Q(s,a) is learned by MSE to the full episode return.
- Actions are input features, so there is no output head over the action space.
- Exploration is ε-greedy, and many parallel actors are used.

It worked because it avoids DQN's overestimation and slow reward propagation, generalizes over a huge action space via action features, and trades variance for massive parallel sampling.

PerfectDou (NeurIPS 2022) then beat DouZero more sample-efficiently with PPO+GAE, using an oracle critic that sees all hands ("perfect-training-imperfect-execution") plus oracle-based reward shaping. DouZero+ added an opponent-hand prediction head and deal filtering. In 2026, VRPO (Fan & Farina) argued that GAE adds variance in imperfect-information self-play and proposed a centralized Q-critic with Expected-SARSA(λ) advantages.

### Cited Findings
**DouZero (Zha et al., ICML 2021, arXiv 2106.06135)**
- **Results and compute.** Starting from scratch on one server with four GPUs, DouZero outperformed all existing DouDizhu AIs within days and ranked first on Botzone among 344 agents. On 48 cores and four 1080Ti GPUs it beat supervised agents in 2 days and surpassed DeltaDou in 10 days — [arXiv 2106.06135](https://arxiv.org/abs/2106.06135)
- **Why DMC over DQN.**
  - (1) DQN overestimation bias is hard to control with function approximation and grows with action-space size; DQN "is very unstable and often diverges in DouDizhu".
  - (2) The long horizon and sparse terminal reward slow down bootstrapped Q-learning, while MC convergence is not affected by episode length.
  - (3) DQN's max over variable legal actions is expensive and hard to batch.
  - "the high variance of DMC is greatly outweighed by the scalability it provides." — [DouZero PDF](https://arxiv.org/pdf/2106.06135)
- **Why not policy gradient.** Taking action features as input lets DMC generalize to rarely seen actions. A classifier-style actor "will still suffer from the large action space", and the authors' actor-critic-with-Q-critic variant was "not very effective". DouDizhu has 27,472 possible moves, of which only a subset is legal per state — [DouZero PDF](https://arxiv.org/pdf/2106.06135); [Self-play survey](https://arxiv.org/pdf/2408.01072)
- **Features and network.**
  - Each card combination is a one-hot 4×15 matrix (count × rank).
  - State features are own hand, the union of other players' hands, recent moves, one-hots for other players' card counts, and bombs played so far.
  - An LSTM encodes historical moves. This is concatenated with state and action features and fed to a 6-layer MLP (512 hidden) that outputs Q.
  - Three separate networks are used, one per seat. — [DouZero PDF](https://arxiv.org/pdf/2106.06135)
- **System.** Many actor processes with local copies of the networks sample trajectories and compute returns. Shared buffers feed one learner with MSE loss — [DouZero PDF](https://arxiv.org/pdf/2106.06135)
- **Reward objectives (WP vs ADP).**
  - ADP is the average point difference, where each bomb doubles the score. ADP-trained agents become cautious about bombs.
  - WP-trained agents play bombs aggressively even when losing, because a bomb does not affect WP.
  - Each objective wins on its own metric. — [DouZero PDF](https://arxiv.org/pdf/2106.06135)

**DouZero+ (Zhao et al., arXiv 2204.02558, Apr 2022)**
- It ranked top on Botzone among >400 agents, including DouZero — [arXiv 2204.02558](https://arxiv.org/abs/2204.02558)
- **Opponent modeling.**
  - A "prediction model" (LSTM + 5 shared MLP layers + multi-head classifier) predicts, for each rank, how many cards of that rank the *next* player holds. It is trained with cross-entropy on the true hidden hand, which is available from the simulator.
  - A "legal label" (the max possible count from own hand + played cards) masks impossible answers.
  - Predictions are concatenated into the decision model's state input. — [DouZero+ PDF](https://arxiv.org/pdf/2204.02558)
- **Coach-guided learning.**
  - A coach network (embedding + FC layers over the initial hands) predicts the landlord's P(win). Deals whose P(win) lies outside a threshold band β are skipped, because they are "of little value for learning".
  - β starts at 0 and increases during training. The coach is trained concurrently on self-play outcomes. — [DouZero+ PDF](https://arxiv.org/pdf/2204.02558)

**PerfectDou (Yang et al., NeurIPS 2022, arXiv 2203.16406; NetEase)**
- **Framework.** "Perfect-training-imperfect-execution" (PTIE), described as a variant of centralized-training-decentralized-execution. The authors cite AlphaStar's full-information value net and Suphx's oracle guiding as precedents — [arXiv 2203.16406](https://arxiv.org/abs/2203.16406); [NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/e26f31de8b13ec569bf507e6ae2cd952-Paper-Conference.pdf)
- **What the actor and critic see.** The actor sees only imperfect information: own hand, unplayed and played cards, last 15 moves, card counts. The critic additionally sees the other two players' hands and their minimum play-out steps — [arXiv HTML v7](https://arxiv.org/html/2203.16406v7)
- **Encoding.**
  - Cards use a 12×15 matrix: 4 count rows plus 8 rows of card-type structure (solo, pair, trio, bomb, rocket, chains).
  - Action features add 7 dims: bomb flag, largest-action flag, whether it matches the left or right player's card count, minimum steps remaining after the action, validity, and action id.
  - 27,472 raw actions are abstracted to 621, with a decoder for kickers. — [arXiv HTML v7](https://arxiv.org/html/2203.16406v7)
- **Oracle reward shaping.** Adv_t = N_t(Landlord) − min(N_t(Peasants)), where N is the minimum steps to play out. The landlord's reward is −1.0·ΔAdv·l and each peasant's is 0.5·ΔAdv·l, with l=50 — [arXiv HTML v7](https://arxiv.org/html/2203.16406v7)
- **Network and hyperparameters.**
  - Policy: LSTM (sequence length 15) state encoder, with each legal action encoded and concatenated, MLP [256,256,256,512], and a softmax over legal actions. Value: MLP [256×4].
  - PPO+GAE with γ=1.0, λ=0.95, entropy 0.1, lr 3e-4, batch 1024.
  - Compute: an IMPALA-like system on 880 CPU cores and 8 GPUs. — [arXiv HTML v7](https://arxiv.org/html/2203.16406v7)
- **Results.**
  - Head-to-head vs DouZero: WP 0.543, ADP 0.143.
  - At equal 1e9 steps it is far stronger (WP 0.732). At 2.5e9 steps it beats DouZero trained ~1e10 steps.
  - Vs skilled humans over 1,260 games: WP 0.625.
  - Decision time is about 6 ms vs DouZero's 2 ms.
  - The table rows should be read carefully, because the per-row comparisons differ. — [arXiv HTML v7](https://arxiv.org/html/2203.16406v7)

**Later DouDizhu work (2023–2026)**
- **AlphaDou (Lei & Lei, arXiv 2407.10279, Jul/Sep 2024).** An end-to-end DMC variant covering bidding and card play. One network estimates both win rate and expected score: the expectation prunes actions, and the win rate selects among them. It claims SOTA among public DouDizhu AIs.
  - The same paper cites RARSMSDou: PPO picks an action category, then DMC picks within it, and it beat DouZero. — [arXiv 2407.10279](https://arxiv.org/html/2407.10279v2)
- **DouRN (CyberC 2023)** replaces DouZero's MLP with residual networks — [DouRN PDF](https://cyberc.org/static/Content/Proceedings/CyberC-2023/pdfs/CyberC2023-36hghVGwLcCZ5k6dSGkB4G/086900a096/086900a096.pdf)
- **VRPO / Q-boosting (Fan & Farina, arXiv 2605.19235, May 2026).**
  - Claim: GAE "suffers from additional variance due to the sampling of stochastic future actions", amplified in equilibrium self-play and persisting "even when the critic is exact".
  - Proposal: Q-boosting, a variance-reduced advantage computed from a centralized action-value critic, inside VRPO. VRPO uses a multi-step Expected SARSA(λ) trace while keeping PPO's clipped objective.
  - Tested on Dou Dizhu and HUNL with "consistently strong performance". — [arXiv 2605.19235](https://arxiv.org/abs/2605.19235)

### Inferences
- **DMC fit for Matgo.** DMC fits well: episodes are short (~33 decisions), rewards are terminal, simulation is extremely cheap, and the action set is small (~177 masked). DouZero's main argument — variance is cheap to fix with parallel samples — is even stronger at 700k games/s. With only ~177 actions, an actor-critic softmax head is also fine; the "large action space" argument against PG is much weaker for Matgo than for DouDizhu.
  - A practical hybrid is a shared trunk with (a) a policy head over 177 masked logits and (b) an oracle Q-critic over the same 177 actions. This yields DMC-style action values plus PPO/VRPO-style policy improvement.
- **Oracle critic transfer.** PTIE transfers directly. The critic (value or Q) can see the opponent's hand and the *stock order*, which removes most chance variance from value targets. Only the actor must stay imperfect-info.
  - The stock order is a perfect-info feature that DouDizhu does not have. It will make the oracle critic's estimates far lower-variance — but also far "sharper" than any imperfect-info policy can realize. Suphx's warning about oracle-to-normal transfer (Q3) applies.
- **Opponent-hand auxiliary head (DouZero+).** It is cheap to add: in 2-player Matgo the hidden set is just opponent hand vs stock, a per-card 2-way classification of unseen cards. Legal-label masking is trivial (cards already seen are excluded).
- **Deal filtering.** Coach-style filtering of lopsided deals speeds learning but changes the training distribution. DAGS (2026, Q4) explicitly warns that changing start-state distributions can bias equilibria, so use it only early or with flags.
- **Reward choice.** DouZero's WP-vs-ADP finding mirrors Koi-Koi's RL-WP vs RL-Point and directly predicts Matgo behavior. A WP objective will over-call Go and over-bomb when behind (variance-seeking). A score objective will be more cautious with multipliers. Since Matgo is scored on points, train on (transformed) score.

### Gaps
- PerfectDou's exact critic loss and total training wall-clock were not extracted (only the first ~100k characters of the HTML were read).
- AlphaDou and DanZero-style full experimental numbers were not verified beyond the abstracts/introductions.
- VRPO numbers (exact win rates vs PerfectDou/DouZero) were not available from the abstract.

## Q3. Suphx, LuckyJ, Mortal, NAGA and 2023–2026 Mahjong RL: oracle guiding, global reward prediction, run-time adaptation, score-across-rounds handling

### Takeaway
Suphx (Microsoft, 2020) introduced three ideas:
- **Global reward prediction.** A GRU maps per-round state to the final game reward, and each round is credited with Φ(x_k) − Φ(x_{k−1}).
- **Oracle guiding.** RL first runs with perfect features, which are then dropped out gradually to zero.
- **pMCPA.** Per-round test-time fine-tuning on sampled rollouts.

Suphx reached 10 dan / stable rank 8.74 on Tenhou. Tencent's LuckyJ (2023) later reached stable rank 10.68, but its method is undisclosed. Open-source Mortal uses offline CQL-regularized Q-learning with a GRP-based rank reward and a 40-block ResNet. In 2026, Mahjax showed a JAX GPU-vectorized Riichi simulator reaching 1–2M steps/s on 8 A100s, but its RL results are modest (BC + PPO).

### Cited Findings
**Suphx (Li et al., Microsoft Research Asia, arXiv 2003.13590, 2020)**
- It reached 10 dan on Tenhou and is "rated above 99.99% of all the officially ranked human players". Stable rank was 8.74 vs 7.46 for top humans. A table lists Suphx at 5,760 games, NAGA (8 dan, 6.64) and Bakuuchi (9 dan, 6.59) — [arXiv 2003.13590](https://arxiv.org/pdf/2003.13590)
- **Pipeline.**
  - Five models (discard, Riichi, Chow, Pong, Kong) are first trained by supervised learning on top-human Tenhou logs. Discard accuracy was 76.7%.
  - They are then trained with policy gradient using importance sampling in a distributed actor setup (CPU simulators + GPU inference engines + a parameter server). — [arXiv 2003.13590](https://arxiv.org/pdf/2003.13590)
- **Features.**
  - Observations are encoded as 34×1 channels (34 tile types).
  - Integer features (scores, tiles left) are bucketed into channels.
  - 100+ "look-ahead" features are computed by depth-first search, giving the probability and score of reaching specific winning hands. Inputs are 34×838 (discard) and 34×958 (others).
  - Deep residual CNNs are used. — [arXiv 2003.13590](https://arxiv.org/pdf/2003.13590)
- **Entropy.** "RL training is sensitive to the entropy of the policy". The entropy coefficient α is adapted dynamically toward a target entropy H_target — [arXiv 2003.13590](https://arxiv.org/pdf/2003.13590)
- **Global reward prediction.**
  - Raw round scores are a poor signal. Late in a game the leader deliberately plays conservatively, so "a negative round score may not necessarily mean a poor policy".
  - Φ is a 2-layer GRU followed by 2 FC layers. It is trained by MSE on top-human logs to predict the final game reward from the sequence of per-round features (round score, accumulated score, dealer, repeat-dealer counter, Riichi bets).
  - The RL reward for round k is Φ(x_k) − Φ(x_{k−1}). — [arXiv 2003.13590](https://arxiv.org/pdf/2003.13590)
- **Oracle guiding.**
  - The oracle sees opponents' private tiles and the wall. "Simple knowledge distillation does not work well", because the normal agent cannot mimic a far stronger oracle.
  - Instead, the oracle is RL-trained with all features. A Bernoulli dropout mask on the perfect features is then decayed with γ_t from 1→0, so the same network becomes a normal agent.
  - After γ_t=0, training continues with the learning rate cut to one tenth and with importance-weight rejection above a threshold; "without these tricks, the continual training is not stable". — [arXiv 2003.13590](https://arxiv.org/pdf/2003.13590)
- **pMCPA (run-time adaptation).** At the start of each round:
  - Opponents' hands and the wall are sampled consistent with own hand, and K rollouts are run with the offline policy.
  - The policy is fine-tuned by gradient steps on those rollouts and used for that round only.
  - It is motivated by top humans playing aggressively with good starting hands and conservatively with poor ones. — [arXiv 2003.13590](https://arxiv.org/pdf/2003.13590)
- **Compute.** Each RL agent took 44 GPUs (4 Titan XP parameter server + 40 K80 self-play) for 2 days, about 1.5M games. The deployed Suphx was trained on about 2.5M games. Offline evaluation over 1M games took 20 K80s for 2 days — [arXiv 2003.13590](https://arxiv.org/pdf/2003.13590)

**Tencent LuckyJ (2023)**
- On July 11, 2023, Tencent announced that LuckyJ reached 10 dan on Tenhou with stable rank 10.68, after 1,321 games, vs 5,373 for Suphx and 26,598 for NAGA. It also beat six professional players in Guobiao (Chinese Official) Mahjong invitationals — [TechNode](https://technode.com/2023/07/12/tencents-mahjong-ai-sets-new-gaming-record-on-international-mahjong-platform/)
- Related Tencent AI Lab method: **ACH (Actor-Critic Hedge), Fu et al., ICLR 2022.** The actor-critic objective minimizes a weighted cumulative counterfactual regret, approximated by a network, with Hedge-generated self-play policies. It is derived from neural weighted CFR with Nash-convergence guarantees, and beat a human champion at 1-on-1 Mahjong — [ICLR 2022 poster](https://iclr.cc/virtual/2022/poster/6627); [slides](https://iclr.cc/media/iclr-2022/Slides/6627.pdf)

**Mortal (open source, Equim-chan, Rust + PyTorch)**
- The docs say it is "powered by deep reinforcement learning". Its emulator reaches up to 40K hanchans/hour (RTX 4090 + Ryzen 9 7950X, batch 2000). The documentation sections on RL/reward were empty when fetched — [Mortal docs](https://mortal.ekyu.moe/)
- From the example config:
  - `[cql] min_q_weight = 5`, i.e., conservative Q-learning regularization.
  - `[env] gamma = 1` and `pts = [6.0, 4.0, 2.0, 0.0]`, i.e., rank-based placement points.
  - A `[grp]` module (hidden 64, 2 layers) gives a GRP reward model.
  - `[aux] next_rank_weight = 0.2`, an auxiliary rank-prediction head.
  - `[resnet] conv_channels = 192, num_blocks = 40`.
  - Separate offline (dataset) and `online` training modes.
  - No oracle keys. — [Mortal config.example.toml](https://raw.githubusercontent.com/Equim-chan/Mortal/main/mortal/config.example.toml)
- A DeepWiki summary (auto-generated, unverified) says GRP predicts game outcomes from the current state for reward calculation (`libriichi/src/dataset/grp.rs`) — [DeepWiki Mortal](https://deepwiki.com/Equim-chan/Mortal)

**Other Mahjong (2022–2026)**
- **Meowjong (3-player Sanma, arXiv 2202.12847, 2022).** CNNs with SL followed by policy-gradient self-play — [arXiv 2202.12847](https://arxiv.org/abs/2202.12847)
- **Tjong (CAAI Trans. Intell. Tech., Aug 2024).**
  - A 15M-parameter Transformer with hierarchical decisions (action-type stage → tile stage).
  - "Fan backward" reward redistribution: winning-hand (fan) points are assigned back to the actions that built them, addressing sparse reward.
  - Trained by SL on ~0.5M samples for 7 days on 2 GPUs. — [DOAJ/CAAI](https://doaj.org/article/59647e6aed75405f880c94040d2b053a); [NJIT](https://digitalcommons.njit.edu/fac_pubs/267)
- **Mahjax (Nishimori, …, Sugiyama; arXiv 2605.20577, May 20, 2026).**
  - A fully vectorized Riichi environment in JAX (Pgx API, functional immutable state). It replaces branching with matrix ops and caches yaku evaluation in precomputed bitmasks.
  - Throughput: up to 2M steps/s (no-red) and 1M steps/s (red) on 8×A100. A single GPU saturates at batch ~2^10. It is >10× faster than the CPU Libriichi.
  - RL demo: BC on 500k samples from a rule-based agent, then PPO with KL to BC, using a Transformer encoder with separate policy/value heads, 1,024 envs, and 100M steps in 5.8 h on one GH200. Average rank only improved below 2.5 vs BC opponents.
  - Its motivation is that prior work "heavily relied on supervised learning from human play logs"; learning from scratch is future work. — [arXiv 2605.20577](https://arxiv.org/html/2605.20577v1)
- **Sparrow Mahjong (arXiv 2508.07522, Aug 2025).** CMA-ES evolution of LSTM agents beats random/rule-based agents — [arXiv 2508.07522](https://arxiv.org/pdf/2508.07522)

### Inferences
- **Score across rounds and go/stop.** The Suphx and Mortal GRP idea — learn a model from intermediate score state to final utility, and reward each segment by the change in predicted utility — is exactly what Guan et al. did with logistic-regression WP for Koi-Koi rounds.
  - In Shin Matgo, each game is scored independently and the target is expected points, so no cross-round GRP is needed. The raw (or monotonically transformed) final score difference is already the correct utility.
  - A GRP-style model matters only for match play (multi-game sessions, bankroll, point caps) or if win rate becomes the metric.
- **Within-game potential.** Suphx's Φ-difference reward is potential-based shaping. Within a Matgo game, an oracle value V(s_full) can play the same role: shaped reward r_t = V(s_{t+1}) − V(s_t) with γ=1 leaves the optimal policy unchanged, and with the oracle critic it greatly reduces variance from stock flips.
- **Oracle guiding vs oracle critic.** For a search-free Matgo policy, the PerfectDou style (oracle *critic* only) is simpler and avoids Suphx's unstable oracle-to-normal transition. Suphx-style feature dropout is a secondary option if the imperfect-info actor learns too slowly.
- **pMCPA.** pMCPA is a form of test-time search (rollouts + fine-tuning). That conflicts with the "search-free" goal, but it is a natural post-hoc boost if ever allowed. The simulator speed would make per-deal adaptation cheap.
- **Mahjax confirms the project's architecture choice.** A GPU-resident, branch-minimized, cached-scoring simulator is the 2026 frontier for tile-game RL. The project's 700k games/s × ~33 steps ≈ 23M decision steps/s on one consumer GPU is >10× Mahjax's 8-A100 figure, though for a much simpler game. Mahjax also shows that from-scratch RL in a complex tile game remains hard without BC.
- **Mortal's offline CQL design** reflects plentiful human logs. With no Matgo human logs, online self-play (DouZero/PerfectDou style) is the relevant template.

### Gaps
- LuckyJ's algorithm, network and training details are not public in what I found. Linking it to ACH is plausible but unconfirmed.
- Mortal's detailed reward design (exact GRP architecture, how placement points map to per-kyoku reward) is undocumented in the fetched pages. Values are inferred from the config file only.
- NAGA (Dwango) methods were not researched beyond Suphx's comparison table.

## Q4. AlphaHoldem and search-free poker; Hanabi (brief contrast); other card-game RL 2023–2026 (Big Two, Gin/Indian Rummy, Uno, Guandan, etc.)

### Takeaway
AlphaHoldem (AAAI 2022) is the clearest proof that a search-free end-to-end PPO policy can beat strong poker bots on modest compute: one 8-GPU / 64-CPU server for 3 days. It beat Slumbot by 111.56 mbb/h and DeepStack by 16.91 mbb/h, at 2.9 ms per decision. Its key tricks were value-target clipping by chips at stake, an extra policy-ratio clip for negative advantages, and K-best historical-opponent self-play. Guandan's DanZero+ (2023) showed DMC-pruned top-k candidates plus PPO beat pure DMC. Recent 2026 small studies (Big 2, Indian Rummy) consistently find PPO plus moderate entropy plus attention over card tokens works well. Hanabi methods (SAD, OBL) target cooperative conventions and mostly do not transfer.

### Cited Findings
**AlphaHoldem (Zhao et al., CASIA, AAAI 2022)**
- It is an end-to-end self-play RL agent for HUNL with a pseudo-Siamese architecture. Card and action tensors are encoded separately into two ConvNet streams and fused.
  - Trained in three days on one server with 8 GPUs and 64 CPU cores. It beat Slumbot by 111.56 mbb/h and DeepStack by 16.91 mbb/h over 100k hands, using 2.9 ms per decision on a single GPU, >1,000× faster than DeepStack. — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/20394); [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20394/20153)
- **Trinal-Clip PPO.**
  - Standard PPO "is difficult to converge" in HUNL for two reasons: the policy loss has large variance when the ratio ≫ 1 with Â<0, and the value loss is huge because of opponent uncertainty and bluffing.
  - Fix: an extra clip δ1 on the ratio for negative advantages, and clipping the value *target* R to [−δ2, δ3], where δ2 and δ3 are chips put in by self and opponent, computed dynamically per sample.
  - This "significantly reduces the variance of the value function", and clipping the value target "further improves the training efficiency and stability significantly, especially for imperfect-information games". — [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20394/20153)
- **K-best self-play.** It keeps a pool of historical versions and trains against the K best of them. It is contrasted with naive, best-win and delta-uniform self-play, PBT and NFSP, and claimed to learn "strong and diverse" styles at low cost — [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20394/20153)
- **AlphaExploitem (arXiv 2605.09150, 2026)** builds on AlphaHoldem's architecture, trained via PPO with K-best league self-play, to exploit suboptimal play — [arXiv 2605.09150](https://arxiv.org/pdf/2605.09150)

**Guandan**
- **DanZero (arXiv 2210.17087, Oct 2022).** DMC with distributed self-play, 30 days on 160 CPUs + 1 GPU, human-level — [arXiv 2210.17087](https://arxiv.org/abs/2210.17087)
- **DanZero+ (arXiv 2312.02561, Dec 2023).**
  - The pretrained DMC model scores legal actions and keeps the top-k. A PPO actor-critic then picks among the k candidates (best k=2). It beats DMC at 55.13%, while k=3 gave 42.30% and k=5 gave 38.45%.
  - State is 513-dim, actions are 54-dim card counts.
  - Reward: +3/+2/+1 by partner finishing position, plus a global ±1 per episode.
  - Trained on one RTX 3070 server, with PPO reaching good performance in under a day vs 30 days for DMC. No oracle features.
  - Beat a graduate-student team in 71 of 100 rounds. — [arXiv HTML 2312.02561](https://arxiv.org/html/2312.02561)
- **GuanZero (arXiv 2402.13582, Feb 2024)** focuses on network encoding to regulate behavior — [arXiv 2402.13582](https://arxiv.org/pdf/2402.13582). **OpenGuanDan (arXiv 2602.00676, early 2026)** is a large-scale benchmark — [arXiv 2602.00676](https://arxiv.org/pdf/2602.00676)

**Big 2 (Patwa, arXiv 2605.28863, May/Jun 2026)**
- Under an identical budget, PPO beat Monte-Carlo Q, SARSA and Q-learning.
- Architecture: a shared card embedding, self-attention over held cards, pooling, a state embedding, and legal actions (80-dim features) scored by dot product with the state.
- Reward: the winner gets the sum of losers' remaining cards; each loser gets minus its own count.
- Entropy: unregularized PPO became over-deterministic (>90% top action). β=0.05 was best and 0.10 was worse.
- Current-policy self-play beat checkpoint-pool self-play and fixed-opponent curricula under a finite budget.
- Win rates vs "Smart" heuristic: PPO+entropy 43.5% vs PPO 37.1% vs MC-Q 32.5% (4-player). Single seed, 7–13 h on a laptop CPU. — [arXiv 2605.28863](https://arxiv.org/html/2605.28863v2)

**Indian Rummy — IRumAI (arXiv 2606.21975, Jun 2026)**
- PPO with a one-time BC warm start, a dual-branch CNN, meld-aware encoding, and a deadwood-shaped reward.
- Linear probes suggest the network implicitly models the opponent's hidden hand.
- 0.33 ms per action. 53.9% vs the strongest search-based heuristic, which was unseen in training. — [arXiv 2606.21975](https://arxiv.org/abs/2606.21975)

**Other recent RL/game-AI items**
- **Data-Augmented Game Starts (DAGS; Lanier, Monette, Baldi, Fox; arXiv 2605.14379, May 2026).**
  - Starts regularized policy-gradient self-play from intermediate states drawn from offline data, giving lower exploitability under fixed compute in hard-exploration games (Kuhn, Goofspiel variants).
  - Augmented start distributions "can produce biased equilibria", mitigated by multi-task observation flags. — [arXiv 2605.14379](https://arxiv.org/abs/2605.14379)
- **Regularized policy optimization for two-player games (Ota et al., arXiv 2602.10894, ICML 2026).** Reverse-KL + entropy-regularized PG with convergence guarantees. More efficient than existing methods on 5 board games; no card games tested — [arXiv 2602.10894](https://arxiv.org/abs/2602.10894)
- **RLCard** is the standard card-RL toolkit (DouDizhu, Mahjong, UNO, Gin Rummy, etc.) — [GitHub rlcard](https://github.com/datamllab/rlcard)
- **Thousand Schnapsen (2025)**: a REINFORCE-based approach for a Skat-family game — [ResearchGate](https://www.researchgate.net/publication/389077163_Policy-Based_Reinforcement_Learning_Approach_in_Imperfect_Information_Card_Game)

**Hanabi (cooperative contrast, brief)**
- **SAD (Hu & Foerster, ICLR 2020, arXiv 1912.02288).** Teammates see the greedy action alongside the exploratory one, addressing exploration noise that obscures signaling. It set SOTA self-play for 2–5 players — [arXiv 1912.02288](https://arxiv.org/pdf/1912.02288)
- **OBL (Hu et al., ICML 2021, arXiv 2103.04000).** Learns grounded, convention-free policies for zero-shot coordination, scoring 20.92 in self-play — [arXiv 2103.04000](https://arxiv.org/pdf/2103.04000)

### Inferences
- **AlphaHoldem is the nearest analogue to Matgo's reward problem.** Matgo also has heavy-tailed, multiplier-driven payoffs (go counts, shake/bomb doubling, pi-bak/gwang-bak — penalties for ending with almost no junk or bright cards — and go-bak). Clipping the value target to a state-dependent bound reduces critic variance.
  - The Matgo analogue is to clip or transform returns relative to the current potential stake.
  - Simpler options are a symmetric log/signed-sqrt transform, or "two-hot" categorical value bins.
  - The extra ratio clip on negative advantages is a cheap PPO stability fix.
- **DanZero+'s "DMC prune → PPO select" result is notable.** PPO on top of a Q-model beat pure DMC. For Matgo the action set is small enough that pruning is unnecessary, but the result supports Q-informed policy improvement over pure ε-greedy DMC.
- **Matgo is 2-player zero-sum like HUNL.** Exploitability matters more than in 3–4-player games. AlphaHoldem's K-best pool, and possibly regularized PG (Ota et al. 2026; MMD-style), are relevant to avoid cyclic self-play.
  - Big 2's finding that current-policy self-play is best was at tiny compute in a 4-player game, so it may not hold at Matgo scale.
- **Entropy.** Suphx (adaptive target entropy), Big 2 (β=0.05 best) and PerfectDou (0.1) all regularize entropy explicitly. Go/stop is a binary high-stakes decision that can collapse early, so entropy floors or targets per decision type are advisable.
- **Hanabi techniques address communication conventions** in cooperative play and do not transfer to competitive 2-player Matgo, except as general recurrent/belief-modeling ideas.

### Gaps
- AlphaHoldem's exact network sizes and K value were not extracted.
- No 2025–2026 PPO self-play paper on Skat, Uno or Bridge at scale was found in my searches.
- Win-rate details of VRPO and OpenGuanDan baselines were not retrieved.

## Q5. Reward design for score-based games, variance-reduction tricks, and feature encodings for card games

### Takeaway
The systems surveyed consistently show that the training objective sets the play style:
- Point/ADP-trained agents maximize expected score and are cautious with multipliers.
- Win-probability-trained agents gamble when behind and play safe when ahead.
- Multi-round games need a learned utility model (GRP or WP regression) to credit each round.

The main variance-reduction tools are:
- privileged/oracle critics (PerfectDou, Suphx);
- potential-based or oracle-shaped rewards (PerfectDou min-steps advantage, Suphx Φ-differences);
- clipped value targets (AlphaHoldem);
- centralized Q-critics with expected backups (VRPO 2026);
- massive parallel MC sampling (DouZero);
- deal filtering (DouZero+ coach).

Encodings have moved from card-count matrices plus LSTM history (DouZero, PerfectDou) to per-card tokens plus attention (Koi-Koi 2023, Tjong 2024, Big 2 2026, Mahjax 2026), with legal actions scored as features or read from per-card outputs.

### Cited Findings
- **Raw score vs win/loss.**
  - DouZero: ADP-trained agents avoid risky bombs; WP-trained agents bomb aggressively even when losing — [DouZero PDF](https://arxiv.org/pdf/2106.06135)
  - Koi-Koi: RL-WP gets higher win rate, RL-Point gets more points. RL-WP ends rounds fast with cheap yaku to protect a late lead — [Guan et al. 2023](https://github.com/guansanghai/KoiKoi-AI)
  - AlphaDou trains one network to estimate both win rate and expected score, pruning with expectation and choosing by win rate — [arXiv 2407.10279](https://arxiv.org/html/2407.10279v2)
- **Multi-round credit assignment.**
  - Suphx GRP: GRU, round reward = Φ(x_k)−Φ(x_{k−1}) — [Suphx](https://arxiv.org/pdf/2003.13590)
  - Koi-Koi: per-round logistic-regression WP, refreshed from self-play — [Guan et al. 2023](https://github.com/guansanghai/KoiKoi-AI)
  - Mortal: GRP module plus placement points [6,4,2,0] — [Mortal config](https://raw.githubusercontent.com/Equim-chan/Mortal/main/mortal/config.example.toml)
- **Oracle-shaped dense reward.** PerfectDou rewards changes in the min-steps-to-finish advantage, computed with perfect information — [PerfectDou](https://arxiv.org/html/2203.16406v7)
- **Reward redistribution.** Tjong's "fan backward" assigns hand-value reward back to the contributing actions — [Tjong / DOAJ](https://doaj.org/article/59647e6aed75405f880c94040d2b053a)
- **Penalty-type rewards.** IRumAI shapes by deadwood — [arXiv 2606.21975](https://arxiv.org/abs/2606.21975). Big 2 rewards by remaining-card counts — [arXiv 2605.28863](https://arxiv.org/html/2605.28863v2). DanZero+ uses finishing-position tiers plus a global ±1 — [arXiv 2312.02561](https://arxiv.org/html/2312.02561)
- **Variance reduction.**
  - Value-target clipping by stake (AlphaHoldem) — [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20394/20153)
  - Q-boosting / Expected-SARSA(λ) advantages from a centralized Q critic (VRPO, 2026) — [arXiv 2605.19235](https://arxiv.org/abs/2605.19235)
  - Importance-weight rejection plus LR decay after the oracle-to-normal transition (Suphx) — [Suphx](https://arxiv.org/pdf/2003.13590)
  - Massive parallelism to outweigh MC variance (DouZero) — [DouZero PDF](https://arxiv.org/pdf/2106.06135)
  - Balanced-deal filtering (DouZero+ coach) — [DouZero+ PDF](https://arxiv.org/pdf/2204.02558)
- **Encodings.**
  - Count×rank card matrices (DouZero 4×15, PerfectDou 12×15 with combo-structure rows) — [DouZero](https://arxiv.org/pdf/2106.06135); [PerfectDou](https://arxiv.org/html/2203.16406v7)
  - 34-tile channel stacks with look-ahead features (Suphx) — [Suphx](https://arxiv.org/pdf/2003.13590)
  - 48 card tokens × ~300 features plus [K0]/[K1] special tokens, with outputs read at legal card tokens (Koi-Koi) — [Guan et al.](https://github.com/guansanghai/KoiKoi-AI)
  - Card-embedding self-attention with dot-product action scoring (Big 2) — [arXiv 2605.28863](https://arxiv.org/html/2605.28863v2)
  - Pseudo-Siamese card vs action tensors (AlphaHoldem) — [AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20394/20153)
  - Tokenized hand/discards/melds into a Transformer (Mahjax) — [arXiv 2605.20577](https://arxiv.org/html/2605.20577v1)

### Inferences (Matgo-specific recommendations synthesized from the above)
1. **Utility.** Train on final signed score difference, including all multipliers (go bonuses, shake/bomb doubling, pi-bak/gwang-bak/meong-bak, go-bak). Apply a monotone variance-reducing transform for the critic, e.g. symmetric log or signed sqrt, or categorical two-hot bins, and invert it for advantages. Report both points-per-game and win rate, as DouZero and Koi-Koi did, because they diverge.
2. **Oracle critic (PTIE).**
   - The critic or Q-head gets the opponent hand plus the full stock order. The policy gets only public info plus own hand.
   - Optionally add potential-based shaping r_t = V_oracle(s_{t+1}) − V_oracle(s_t), which preserves the optimal policy with γ=1. It cancels most flip and shuffle luck from the policy gradient.
   - Consider VRPO-style expected advantages using the oracle Q over all 177 legal actions. These are cheap with small action sets and remove action-sampling noise.
3. **Auxiliary heads.** Add a DouZero+-style opponent-hand prediction head (per unseen card: in opponent hand vs in stock), with legal-label masking. Add a score/outcome-distribution head (cf. Mortal's next-rank auxiliary).
4. **Separate treatment of go/stop and shake/bomb.**
   - These are rare, high-leverage binary decisions. Use a dedicated head or special token ([K0]/[K1] style) and per-decision-type entropy targets (Suphx adaptive α).
   - Check for "always-Go" collapse, as observed in hwatu-gostop-rl.
5. **Self-play diversity.** In 2-player zero-sum play, keep a historical K-best opponent pool (AlphaHoldem) or a regularized PG update (Ota et al. 2026) to avoid cycling. Evaluate with seat- and deal-paired duplicates.
6. **Encoding.** The project's 496-dim flat vector is comparable to Koi-Koi's ~300 features per card. A per-card token Transformer over 48 (+ bonus) cards is the best-documented hanafuda encoding. Given the throughput budget, compare it against a wide MLP or ResNet on the flat vector, since network cost will dominate at 700k games/s.

### Gaps
- None of the systems published a controlled comparison of value-transform choices (log, sqrt, two-hot) for heavy-tailed card-game payoffs. That recommendation is an inference from AlphaHoldem's clipping result, not a measured card-game finding.
- No source quantified how much an oracle critic helps in a 2-player game with a hidden stock order like Matgo's. The PerfectDou and Suphx evidence comes from 3–4-player games.
