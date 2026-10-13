# CFR-Family Training Methods for Large Imperfect-Information 2p0s Games (through Oct 2026), judged for 2-player Shin Matgo

Notes compiled 2026-10-12. Sourcing: items marked **[fetched]** were confirmed against the linked page during this research session. Items marked **[canonical, not re-fetched]** are standard results from the original paper at the linked URL, stated from prior knowledge and not re-read in this session. The report writer should treat them as reliable but give the [fetched] items priority if anything conflicts.

Project constraints used for the fit judgments: about 33 decision steps per game (about 16 per player), up to 177 masked actions, heavy chance (shuffle, flips), go/stop branching, score-based zero-sum reward with multipliers. Hardware is one RTX 5080 (16 GB) running a GPU batch simulator at about 700k games/s, with 176-byte POD states and about 65M state clones/branches per second. The goal is a search-free policy network.

---

## Q1. Neural CFR methods (Deep CFR, SD-CFR, DREAM, ESCHER, and 2023-2026 successors): what they are and how they compare

### Takeaway
The neural CFR line runs from Deep CFR (2019, external sampling, needs a perfect simulator) to SD-CFR (2019, no average-policy net), then to DREAM (2020, outcome sampling plus a learned baseline, model-free), ESCHER (ICLR 2023, no importance sampling, uses a history value function), and VR-DeepDCFR+/VR-DeepPDCFR+ (AAAI 2026, DREAM-style baseline plus bootstrapped cumulative advantages with DCFR+/PDCFR+ discounting). VR-DeepPDCFR+ is the strongest published neural CFR. One caution matters for planning: the largest independent benchmark (ICLR 2026, more than 7000 runs) found that ESCHER was "uniformly noncompetitive" and that tuned generic policy-gradient methods (PPO/MMD/PPG) matched or beat all game-theoretic deep RL methods on exploitability. Neural CFR is therefore not automatically the strongest option at scale.

### Cited Findings

**Deep CFR (Brown, Lerer, Gross, Sandholm; ICML 2019)** [canonical, not re-fetched]
- Approximates tabular CFR with a value (advantage) network per player. The network is retrained each CFR iteration on a reservoir-sampled buffer of sampled instantaneous regrets from all prior iterations. A separate average-strategy network is trained on a reservoir buffer of (infoset, strategy) samples. Traversals use external sampling, and the loss is weighted by iteration t to mimic Linear CFR. It was evaluated on Flop Hold'em Poker (FHP) and heads-up limit hold'em, where it beat NFSP and was competitive with abstraction-based approaches. — [arXiv 1811.00164](https://arxiv.org/abs/1811.00164)
- Convergence argument: if the value-net approximation error is bounded, the average policy converges to an ε-Nash in two-player zero-sum games. The guarantee is for the average strategy, not the last iterate. — [arXiv 1811.00164](https://arxiv.org/abs/1811.00164)

**Single Deep CFR (SD-CFR; Steinberger, 2019)** [fetched]
- Removes Deep CFR's average-strategy network. The average strategy is instead represented by storing every iteration's value network and sampling or mixing among them. — [arXiv 1901.07621](https://arxiv.org/abs/1901.07621)
- The author claims lower overall approximation error than Deep CFR and reports that SD-CFR beats Deep CFR on exploitability and in one-on-one play (poker). — [arXiv 1901.07621](https://arxiv.org/abs/1901.07621)

**DREAM (Steinberger, Lerer, Brown; 2020)** [fetched abstract]
- A model-free regret-based deep RL algorithm. Unlike other regret-based deep methods, it does not require a perfect simulator to perform well. It claims convergence to Nash in two-player zero-sum games (and to an extensive-form coarse correlated equilibrium otherwise), state of the art among model-free methods, and competitiveness with methods that use a perfect simulator. — [arXiv 2006.10410](https://arxiv.org/abs/2006.10410)
- Mechanism (via the Deep PDCFR+ paper's description): outcome-sampled advantages are corrected with a learned history baseline Q, following VR-MCCFR. — [arXiv 2511.08174 (html)](https://arxiv.org/html/2511.08174)

**ESCHER (McAleer, Farina, Lanctot, Sandholm; ICLR 2023)** [fetched]
- Motivation: DREAM's regret target has high variance because of the importance-sampling term inherited from MCCFR. ESCHER is unbiased and model-free with no importance sampling. It trains a history value function and computes regret estimates from it, sampling the traverser under a fixed policy. — [arXiv 2206.04122](https://arxiv.org/abs/2206.04122v2); [Farina page](https://www.mit.edu/~gfarina/2023/escher_iclr23)
- Guarantee: converges to an approximate Nash equilibrium with high probability. The workshop version stated this for the tabular case. — [arXiv 2206.04122](https://arxiv.org/abs/2206.04122v2)
- Reported results: regret-estimate variance is orders of magnitude lower than DREAM's. It beats DREAM and NFSP, with the margin widening as game size grows. In dark chess it beats DREAM and NFSP head-to-head more than 90% of the time. — [arXiv 2206.04122](https://arxiv.org/abs/2206.04122v2); [ICLR 2023 poster](https://iclr.cc/virtual/2023/poster/10948)

**VR-DeepDCFR+ / VR-DeepPDCFR+ (Xu, Li, Fu, Fu, Xing, Cheng; arXiv Nov 2025, AAAI 2026)** [fetched]
- Problem stated: neural CFR methods mostly approximate vanilla CFR and struggle to incorporate advanced variants. — [arXiv 2511.08174](https://arxiv.org/abs/2511.08174)
- Networks per player: a cumulative-advantage net R(I,a) and an average-policy net Π(I,a). The PDCFR+ variant adds an instantaneous-advantage net r(I,a) that predicts the next iteration's advantages. A history value network Q serves as the variance-reduction baseline. — [arXiv html](https://arxiv.org/html/2511.08174)
- Bootstrapping: each iteration, R is trained toward the previous net's output after discounting by (t−1)^α/((t−1)^α+1) and clipping to the positive part, plus the new sampled advantages. The advantage buffer is therefore cleared every iteration instead of being reservoir-accumulated. — [arXiv html](https://arxiv.org/html/2511.08174)
- Sampling: outcome sampling with exploration ε = 0.6 plus the DREAM-style baseline. The average-policy and history-value buffers hold 1M samples each (reservoir for the average policy). The average strategy is weighted by t^γ with γ = 2. Tuned α is 2 (DCFR+) and 2.3 (PDCFR+). — [arXiv html](https://arxiv.org/html/2511.08174)
- Hyperparameters: 3 hidden layers of 64 units (128 for FHP), learning rate 1e-3, batch 2048. Per iteration: 750 SGD steps for the advantage net, 5,000 for the average-policy net, 10,000 for the value net, and 10,000 traversals. Totals are 10M episodes (convergence experiments) and 1e8 episodes with a 1e7 buffer (FHP). Hardware is a 64-core AMD EPYC CPU server with no GPU mentioned. Runtime is stated to be roughly equal to DREAM, with no wall-clock figures. — [arXiv html](https://arxiv.org/html/2511.08174)
- Results: on 8 OpenSpiel games (Kuhn, Leduc, Liar's Dice 5/6, GoofspielImp 5/6, Battleship 2/3), both methods converge faster than OS-DeepCFR and DREAM in most games (exploitability curves; no table). In FHP against 5 rule-based agents (20k matches), average chips/hand are: OS-DeepCFR −7.8±1.4, DREAM −2.0±3.1, VR-DeepDCFR+ 11.6±1.2, VR-DeepPDCFR+ 11.3±0.9. ESCHER and Deep CFR (external sampling) were not among the experimental baselines. — [arXiv html](https://arxiv.org/html/2511.08174)
- Ablations: removing cumulative-advantage fitting, replacing the DCFR/PDCFR updates with Linear-CFR style, or removing the baseline each hurt performance. — [arXiv html](https://arxiv.org/html/2511.08174)
- Code: [github.com/rpSebastian/DeepPDCFR](https://github.com/rpSebastian/DeepPDCFR) (link from the AAAI version, [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38780/42742)).

**Other neural CFR work listed as related (2020-2024)** [fetched, as listed in the Deep PDCFR+ related work]
- ARMAC (Gruslys et al. 2020), Double Neural CFR (Li et al. 2020), model-free neural CFR with bootstrap learning (Liu et al. 2022), Meng et al. 2023, and "Learning not to regret" (Sychrovský et al. 2024, meta-learned regret minimizers). — [arXiv 2511.08174 html](https://arxiv.org/html/2511.08174)
- D2CFR (dueling-network Deep CFR, 2021). — [arXiv 2105.12328](https://arxiv.org/abs/2105.12328)
- "Robust Deep MCCFR" (El Jaafari, Aug 2025) proposed target networks, uniform exploration mixing and variance-aware losses. The paper has been **withdrawn** by its author for errors, and its stated percentage improvements do not match its own numbers. Do not rely on it. — [arXiv 2509.00923](https://arxiv.org/abs/2509.00923)

**Independent cross-family comparison: "Reevaluating Policy Gradient Methods for IIGs" (Rudolph, Lichtlé, Mohammadpour, ..., Farina et al.; arXiv Feb 2025, ICLR 2026)** [fetched]
- Compared NFSP, PSRO, ESCHER, R-NaD, PPO, PPG and MMD on 3x3 Dark Hex, Abrupt Dark Hex, Phantom TTT and Abrupt Phantom TTT. These games have 19-29B states and 6-27M infostates, and exact exploitability was computed via sequence form. — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2)
- Budget: about 5,600 runs (v1), more than 7000 runs over five games (ICLR version), more than 278k CPU hours, 50 tuning configurations per algorithm-game pair, 10M steps per run, and a 3×512 MLP for all methods. — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006833)
- Results: ESCHER was "uniformly noncompetitive". PSRO was largely noncompetitive. NFSP and R-NaD sometimes matched but usually underperformed. PPO, PPG and MMD were strongest and roughly equal, and they matched or beat the others head-to-head. The authors caution that untuned hyperparameters or the 10M-step cap could explain weak results, and that only one representative per family was tested. DREAM and Deep CFR were not evaluated. Best entropy coefficients were 0.05-0.2, far above common library defaults. — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2)

### Inferences
- Within neural CFR, the published ordering is roughly VR-DeepPDCFR+ ≈ VR-DeepDCFR+ > DREAM > OS-DeepCFR (Xu et al. 2026), with ESCHER > DREAM per McAleer et al. 2023. No paper compares VR-DeepPDCFR+ against ESCHER or against tuned PPO/MMD directly, so the overall ordering across families is unresolved.
- The ICLR 2026 negative result was measured on deterministic phantom/dark board games with long horizons. Matgo is chance-heavy and short, which is closer to poker, where the CFR-family methods were developed. The result is a strong signal to always run a tuned PPO/MMD baseline with exact or approximate exploitability measurement. It does not prove that CFR will lose on Matgo.
- SD-CFR's trick (no average-policy net) yields a "policy" that is a mixture over stored networks. For a single search-free deployed network you would still need to distill the mixture into one network, so in practice an average-policy network (Deep CFR / DREAM / Deep PDCFR+ style) is still needed.

### Gaps
- No exploitability tables were found for Deep PDCFR+ (only plots). No wall-clock numbers exist for any neural CFR paper read here.
- No published head-to-head of Deep CFR vs ESCHER vs VR-DeepPDCFR+ vs PPO/MMD on the same chance-heavy card game was found.
- I found no 2026 neural CFR algorithm newer than VR-DeepPDCFR+ in arXiv search results. 2026 CFR papers found were about tabular/GPU parallelization (Q4).

---

## Q2. Sampling schemes: external vs outcome sampling, variance reduction, and which suits a ~33-step, high-chance game

### Takeaway
External sampling (ES) has low variance but its cost grows exponentially with the number of the traverser's own decisions (it enumerates every traverser action). Outcome sampling (OS) costs one trajectory but carries high-variance importance weights. Baselines (VR-MCCFR/DREAM) or learned history values (ESCHER) are the established fixes. For Matgo, with about 16 traverser decisions, full ES is infeasible. The practical options are OS plus a learned baseline, ESCHER-style value-function regret estimates, or a hybrid that branches all actions at traverser nodes on one sampled trajectory and estimates each child by rollouts or a value net. The GPU clone API makes the hybrid cheap.

### Cited Findings
- MCCFR (Lanctot, Waugh, Zinkevich, Bowling; NeurIPS 2009) introduced external sampling (sample chance and opponent actions, enumerate the traverser's actions) and outcome sampling (sample a single terminal history with importance weighting). Both give unbiased counterfactual regret estimates and converge in two-player zero-sum games. [canonical, not re-fetched] — [NeurIPS 2009](https://papers.nips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)
- VR-MCCFR (Schmid, Burch, Lanctot, Moravcik, Kadlec, Bowling; 2018/AAAI 2019): rewrites sampled values with state-action baselines, as in policy-gradient RL. It stays unbiased, reaches zero variance with a perfect baseline, reduces empirical variance by about 3 orders of magnitude, and speeds things up by about 10×. The lower variance makes CFR+ usable with sampling for the first time, giving about 100× speedup. [fetched] — [arXiv 1809.03057](https://arxiv.org/abs/1809.03057)
- DREAM brings the VR-MCCFR baseline into the neural, outcome-sampling setting, making it model-free (no perfect simulator needed). [fetched] — [arXiv 2006.10410](https://arxiv.org/abs/2006.10410); [Deep PDCFR+ html](https://arxiv.org/html/2511.08174)
- ESCHER removes importance sampling entirely by using a history value function and reports regret variance orders of magnitude below DREAM's. [fetched] — [arXiv 2206.04122](https://arxiv.org/abs/2206.04122v2)
- Critique from the ICLR 2026 benchmark: importance-sampling CFR methods have high-variance feedback, and ESCHER's fix requires a uniform behavioral policy, which makes trajectory coverage difficult. [fetched] — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2)
- VR-DeepPDCFR+ uses outcome sampling with ε = 0.6 exploration plus a baseline and reports that removing the baseline hurts. [fetched] — [arXiv html](https://arxiv.org/html/2511.08174)
- Deep CFR uses external sampling and relies on a perfect simulator (state reset and branching). [canonical, not re-fetched] — [arXiv 1811.00164](https://arxiv.org/abs/1811.00164)
- Pluribus (Brown & Sandholm, Science 2019) computed its blueprint with external-sampling MCCFR plus Linear-CFR weighting and negative-regret pruning on a 64-core CPU server (about 8 days). [canonical, not re-fetched] — [Science 2019](https://www.science.org/doi/10.1126/science.aay2400)

### Inferences
- **ES cost in Matgo:** with roughly b legal actions at each of about 16 traverser decisions, one ES traversal visits on the order of b^16 nodes before chance and opponent sampling. Even with b of about 3-5 that is 4×10^7 to 10^11 nodes per traversal, so full ES is infeasible despite 65M branches/s. ES worked for Deep CFR in FHP because that game has only a few betting decisions per player.
- **Recommended hybrids for the GPU simulator:**
  - (a) *OS plus baseline* (DREAM / VR-DeepPDCFR+): one trajectory per traversal. This is cheapest and most GPU-friendly (it is literally batched self-play). Variance is controlled by the history-value baseline, and the 496-dim features plus hidden-state features can feed Q(h,a), because the trainer sees the full state.
  - (b) *ESCHER-style*: sample the trajectory under a fixed or mixed policy, and at every traverser node compute regrets for all legal actions from Q(h,a). There is no branching and no importance weighting. The weakness is the quality of the history value function.
  - (c) *"One-level ES" / branch-and-rollout*: on a sampled trajectory, clone the state at each traverser node, branch every legal action, and estimate each child's value with K rollouts under the current policy, or with Q bootstrapping. Cost per trajectory is roughly 16 × b × K × (remaining steps). With b = 8, K = 4 and about 16 remaining steps, that is about 8k simulator steps, so the 65M branches/s and 700k games/s budget allows tens of thousands of such traversals per second. This is unbiased with respect to the current policy's values, but it is a different estimator from textbook ES (it does not recurse), and no paper found validates it as a CFR method.
- Heavy chance (stock flips) adds variance that ES does not remove. ES samples chance anyway, so this hurts all schemes. Baselines conditioned on the full history (including hidden cards) remove the most variance because the trainer can see the true deal.

### Gaps
- No paper found benchmarks ES vs OS vs ESCHER directly on a game with about 30 sequential decisions and heavy chance.
- No published GPU-batched (simulator-on-GPU) neural MCCFR throughput figures were found (see Q4).

---

## Q3. Modern regret-update rules (CFR+, LCFR, DCFR, PCFR+, PDCFR+, DDCFR, HS-PCFR+, learned/meta CFR) and which transfer to the sampled/neural setting

### Takeaway
In tabular solving the fastest practical rules (2024-2026) are PCFR+, DCFR, their combination PDCFR+, and schedule-based HS-DCFR/HS-PCFR+ (training-free, about 10-15 lines of code). DDCFR learns discount weights by RL. Linear-CFR-style weighting has transferred to sampled/neural CFR since Deep CFR and Pluribus. CFR+ only became sampling-friendly with variance reduction. DCFR+/PDCFR+ discounting was shown to transfer to model-free neural CFR only in 2026, via bootstrapped cumulative-advantage nets. Guarantees are for the average iterate. Last-iterate behavior of RM+/PRM+ is not guaranteed in theory.

### Cited Findings
- CFR+ (Tammelin 2014; used to solve HULHE): regret-matching+ (cumulative regrets floored at 0), alternating updates and linear averaging. It is much faster than CFR in practice, and its current iterate is often good in practice. [canonical, not re-fetched] — [arXiv 1407.5042](https://arxiv.org/abs/1407.5042)
- Linear CFR / DCFR (Brown & Sandholm, AAAI 2019): discount positive regrets by t^α/(t^α+1), negative regrets by t^β/(t^β+1), and average contributions by (t/(t+1))^γ. The recommended setting is α = 1.5, β = 0, γ = 2. DCFR outperformed CFR+ on many poker benchmarks, and LCFR is the variant they paired with MCCFR. [canonical, not re-fetched] — [arXiv 1809.04040](https://arxiv.org/abs/1809.04040)
- Predictive CFR+ (Farina, Kroer, Sandholm; AAAI 2021): an optimistic or predictive regret matcher (uses the last instantaneous regret as a prediction), state of the art in many non-poker games. [canonical, not re-fetched] — [arXiv 2007.14358](https://arxiv.org/abs/2007.14358)
- Last-iterate theory: RM+ and predictive RM+ can fail to converge in the last iterate, while smoothed or extragradient variants do converge (Cai, Farina, Grand-Clément, Kroer, Lee, Luo, Zheng 2023). [canonical, not re-fetched] — [arXiv 2311.00676](https://arxiv.org/abs/2311.00676)
- DDCFR (Xu, Li, Fu, Fu, Xing, Cheng; ICLR 2024 spotlight): models the CFR iteration process as an MDP and learns a policy that sets discount weights dynamically from runtime features. It generalizes across games and converges faster. [fetched] — [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a89331e2ecbbd9d9618bc94717a91155-Abstract-Conference.html); code [rpSebastian/DDCFR](https://github.com/rpSebastian/DDCFR)
- Hyperparameter Schedules (Zhang, McAleer, Sandholm; arXiv Apr 2024, v2 Feb 2026, AAAI 2026): HS-DCFR schedules α, β, γ (under 15 lines of code). HS-PCFR+ schedules only γ (under 10 lines). Both are training-free and provably converge if the hyperparameters stay within ranges. Early iterations are aggressively down-weighted. Schedules derived from 3 small games generalize to 17 games including large poker, often by orders of magnitude at a given runtime. The paper critiques DDCFR's extra cost (feature computation, policy training, inference). [fetched] — [arXiv 2404.09097](https://arxiv.org/html/2404.09097v2); [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38784)
- PDCFR+ (Xu et al., IJCAI 2024) combines PCFR+ and DCFR. The neural versions are VR-DeepPDCFR+ and VR-DeepDCFR+. [fetched via 2511.08174] — [arXiv 2511.08174](https://arxiv.org/abs/2511.08174)
- Why advanced rules are hard to make neural: DCFR+/PDCFR+ bootstrap from the previous iteration's cumulative regrets, which a replay buffer holding all past iterations cannot reproduce. Counterfactual values are weighted by opponent reach, which spans orders of magnitude. The fix is to fit cumulative *advantages* by bootstrapping from the previous net and to apply discount and clip to the network output. [fetched] — [arXiv html](https://arxiv.org/html/2511.08174)
- "Learning not to regret" (Sychrovský et al. 2024) meta-learns regret minimizers. It is listed as related neural work. [fetched as citation] — [arXiv 2511.08174 html](https://arxiv.org/html/2511.08174)
- GPU-CFR (2026) supports vanilla CFR, CFR+, DCFR and PCFR+ with simultaneous or alternating updates and uniform or linear averaging. This indicates all of these rules are cheap in the tabular GPU setting. [fetched] — [arXiv 2609.11923](https://arxiv.org/html/2609.11923)

### Inferences
- What transfers to Matgo neural training: (1) **Linear/polynomial iteration weighting** (Deep CFR loss weights t, or t^γ for the average policy) is cheap and established. (2) **DCFR+/PDCFR+ via bootstrapped cumulative-advantage nets** is the 2026 state of the art, with code available. (3) **HS-style schedules** can be layered on Deep PDCFR+'s α/γ, but this is untested in the neural setting. (4) **DDCFR's learned discounting** adds an extra RL loop and is unlikely to pay off at one-GPU scale.
- Because guarantees are for the average strategy, the deployed search-free network should be the average-policy net (or a distillation of it), not the current regret-matching policy. That requires a reservoir or weighted buffer of (infoset, policy, weight) samples.

### Gaps
- No evidence was found on whether HS schedules or DDCFR improve neural/sampled CFR.
- No neural method with last-iterate guarantees in the CFR family was found. Regularized methods (MMD, R-NaD) cover that niche but are outside CFR.

---

## Q4. GPU-accelerated / massively batched CFR implementations and throughput

### Takeaway
All published GPU CFR work through 2026 is **tabular, full-width CFR on explicit game trees**. Examples are Kim 2024 (matrix form, up to about 400× over OpenSpiel Python), Kim & Sandholm 2026 (up to 10^4× over OpenSpiel CPU), GPU-CFR 2026 (static dataflow plus CUDA Graphs, median 44× over Kim on an A100), and Pasur 2025. None of these applies to Matgo's full game, which cannot be enumerated. No published GPU-resident simulator plus neural MCCFR pipeline comparable to this project's setup was found. Neural CFR papers still report CPU-only training.

### Cited Findings
- Kim, "GPU-Accelerated CFR" (arXiv Aug 2024): recasts CFR tree traversal as dense and sparse matrix/vector operations. It reports up to about 401× speedup over OpenSpiel Python and about 204× over OpenSpiel C++. Float32 roughly doubles speed and halves memory with negligible accuracy loss. Code: [uoftcprg/gpugt](https://github.com/uoftcprg/gpugt). [fetched via search summary] — [arXiv 2408.14778](https://www.alphaxiv.org/abs/2408.14778)
- Kim & Sandholm, "Parallelizing CFR" (arXiv May 2026): a generalized linear-algebra parallelization framework supporting CFR+, DCFR and predictive CFR variants. The GPU implementation is up to 4 orders of magnitude faster than OpenSpiel CPU CFR. [fetched] — [arXiv 2605.14277](https://arxiv.org/abs/2605.14277)
- GPU-CFR (Li & Huang, Tsinghua; arXiv Sep 2026), details below. [fetched] — [arXiv 2609.11923](https://arxiv.org/html/2609.11923)
  - Approach: compiles a two-player zero-sum perfect-recall game once into flat edge, infoset and depth arrays, folds chance statically, executes one depth level at a time, and captures the whole iteration as one CUDA Graph launch.
  - Speed: on one A100 it is 29.8-80.4× faster than Kim's GPU CFR+ (median 44.1×) and 14-258× faster than LiteEFG on the four largest games. HUNL turn subgame (83k infosets, 434k states) runs at 0.397 ms/iteration.
  - CPU result: the compiled representation running on 8 CPU threads is already 2.2-51.1× faster than the A100 baseline.
  - Limits: tabular only, with neural or sampled variants out of scope. It uses about 570 B per tree node, and the largest test game was Battleship with 276k infosets. Linear averaging needs float64 accumulators. GPU scatter nondeterminism was observed. It reaches only 5.5% of peak bandwidth, so it is launch/latency bound.
- Baghal, "Solving Pasur Using GPU-Accelerated CFR" (arXiv Aug 2025, accepted COLM 2025) uses a CUDA framework for a fishing card game (Pasur) with emphasis on memory management. This is the closest analog to Matgo (a fishing-type card game) found. [fetched via search summary] — [arXiv 2508.06559](https://arxiv.org/abs/2508.06559)
- "Real-Time Parallel CFR" (arXiv May 2026): a heterogeneous CPU-GPU pipeline that offloads leaf evaluation to batched NN inference, with output identical to serial CFR. [fetched via search summary] — [arXiv 2605.19928](https://arxiv.org/html/2605.19928v1)
- Older CUDA tabular implementation: [janrvdolf/gpucfr](https://github.com/janrvdolf/gpucfr). A 2024 KTH master's thesis (Bergmark) covers GPU CFR for real-time decision making. — [DiVA](https://www.diva-portal.org/smash/get/diva2:1933074/FULLTEXT02.pdf)
- The VR-DeepPDCFR+ experiments ran on a 64-core CPU server with no GPU mentioned. [fetched] — [arXiv html](https://arxiv.org/html/2511.08174)
- The ICLR 2026 benchmark used more than 278k **CPU** hours, and its exploitability evaluation used sequence-form on-the-fly payoff generation, under 2 minutes per evaluation for games with about 20B states. [fetched] — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2)
- OpenSpiel provides reference implementations of tabular CFR/CFR+/MCCFR (external and outcome sampling) and Deep CFR (TF/PyTorch/JAX). [canonical, not re-fetched] — [OpenSpiel GitHub](https://github.com/google-deepmind/open_spiel)

### Inferences
- Tabular GPU CFR is useful for Matgo only on **small abstractions or endgame subgames**, for example the last few tricks with known remaining cards, or a reduced deck. These can serve as ground truth for measuring the neural policy's exploitability in subgames. GPU-CFR's compiler pattern (flat arrays, CUDA Graph replay) maps naturally onto a project that already uses a CUDA simulator.
- For the full game, the bottleneck in a neural MCCFR loop on this hardware will be NN training (retraining or bootstrapping advantage nets plus average-policy distillation) and inference during traversals, not simulation. 700k games/s exceeds the traversals-per-iteration budgets used in the literature (10k traversals per iteration in Deep PDCFR+) by orders of magnitude. Designs should spend the simulator surplus on lower-variance estimates (more rollouts per branch, larger traversal batches) rather than on more iterations.

### Gaps
- No throughput figures were found for GPU-batched *sampled* or *neural* CFR (traversals/s with NN inference in the loop).
- No JAX neural CFR implementation with published throughput was found in this session.

---

## Q5. Practical pitfalls: target scaling with large score rewards, buffers, average-policy training, wall-clock

### Takeaway
The main pitfalls are documented in the newest papers. Reach-weighted counterfactual values span orders of magnitude, so fit (cumulative) advantages rather than raw counterfactual regrets. Retraining from scratch with reservoir buffers is costly, and bootstrapping is cheaper. Average-policy networks need weighted reservoirs and many SGD steps. Importance-weight variance is severe without baselines. Precision matters for linear averaging. For Matgo's heavy-tailed, multiplier-based scores, apply a fixed linear rescaling of payoffs (which preserves the equilibrium) and use baselines, rather than nonlinear reward shaping.

### Cited Findings
- Reach-weighted counterfactual values vary by orders of magnitude across infosets and are hard for networks to fit. Fitting cumulative advantages fixes this, and the ablation gap widens in more complex games. [fetched] — [arXiv html](https://arxiv.org/html/2511.08174)
- Buffer and training budgets in Deep PDCFR+: advantage buffer cleared per iteration, average-policy and value buffers of 1M (FHP total buffer 1e7), and 750/5,000/10,000 SGD steps per iteration (advantage/policy/value). Bootstrapping needs fewer total training steps than DREAM, which retrains from scratch. [fetched] — [arXiv html](https://arxiv.org/html/2511.08174)
- Deep CFR retrains the advantage net from scratch each iteration on reservoir buffers (tens of millions of samples in the poker experiments) and trains a separate average-strategy net, which SD-CFR shows adds approximation error. [canonical for Deep CFR, not re-fetched — [arXiv 1811.00164](https://arxiv.org/abs/1811.00164); fetched for SD-CFR — [arXiv 1901.07621](https://arxiv.org/abs/1901.07621)]
- DREAM's regret targets have extremely high variance because of the outcome-sampling importance term. This motivated ESCHER. [fetched] — [arXiv 2206.04122](https://arxiv.org/abs/2206.04122v2)
- ESCHER's uniform behavioral sampling policy makes trajectory coverage of relevant lines difficult. [fetched] — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2)
- Linear averaging needs float64 accumulators (float32 stalls), and GPU scatter-add is nondeterministic across runs. This was shown in tabular GPU CFR. [fetched] — [arXiv 2609.11923](https://arxiv.org/html/2609.11923)
- Hyperparameter sensitivity is large. In the ICLR 2026 benchmark the best entropy coefficients (for PG methods) were 0.05-0.2 versus library defaults of 0-0.01, and the authors flag untuned hyperparameters as a possible cause of any method's weak results. [fetched] — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2)
- The withdrawn Robust Deep MCCFR preprint listed nonstationary targets, action-support collapse, variance explosion and warm-start bias as neural MCCFR failure modes. The diagnosis is plausible, but the paper is withdrawn and its numbers are unreliable. [fetched] — [arXiv 2509.00923](https://arxiv.org/abs/2509.00923)
- Wall-clock: Pluribus' blueprint took about 8 days on a 64-core CPU server [canonical, not re-fetched — [Science 2019](https://www.science.org/doi/10.1126/science.aay2400)]. Deep PDCFR+ reports no wall-clock. The ICLR 2026 benchmark spent 278k CPU hours over about 5.6k runs (about 50 CPU-hours per 10M-step run on average) [fetched — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2)].

### Inferences
- **Score scaling for Matgo:** multiplying all payoffs by a positive constant leaves the Nash equilibrium and regret-matching policies unchanged, because regret matching is scale-invariant. Normalize by a fixed constant, such as a high percentile of |score|, so advantage-net targets are O(1). Do **not** use per-batch normalization, clipping or log-transforms on the CFR targets. Those change the game and bias the regrets, unless you deliberately accept a different objective, such as win-probability instead of expected score. Heavy multiplier tails (go stacks, shake/bomb/bak multipliers) are a variance problem best handled by baselines and larger traversal batches, not by clipping.
- The go/stop decision is a low-branching, very-high-stakes node. Its regrets will dominate magnitude, which argues for per-action advantage outputs with masking (177-way head) and possibly loss weighting so that small card-choice regrets are not swamped.
- For the deployed network, train or distill the average-policy net with iteration weights (t or t^γ) and a large weighted reservoir. On 16 GB VRAM, 1e7 samples × 496 float16 features ≈ 10 GB. Keep the reservoir in pinned host RAM, or store the compact 176-byte state and recompute features on the GPU (1e7 × 176 B ≈ 1.8 GB).

### Gaps
- No paper found quantifies the effect of reward magnitude or heavy-tailed payoffs on neural CFR stability.
- No reported reservoir-size sensitivity study was found for 2023-2026 methods.

---

## Q6. Overall fit for 2-player Shin Matgo on one RTX 5080 (synthesis)

### Takeaway
The best-supported CFR-family plan is a model-free, outcome-sampling neural CFR with a full-information history-value baseline and bootstrapped DCFR+/PDCFR+ cumulative advantages (VR-DeepPDCFR+, AAAI 2026, open code). Run it batched on the GPU simulator, possibly with one-level branch-and-rollout regret estimates using the clone API. Validate against a tuned PPO/MMD self-play baseline, because the 2026 independent benchmark found generic policy gradient competitive or superior to CFR-based deep RL. Full external-sampling Deep CFR is a poor fit because of the roughly 16-deep traverser branching.

### Cited Findings
- VR-DeepPDCFR+ / VR-DeepDCFR+ is the best published model-free neural CFR, beating DREAM and OS-DeepCFR on 8 games and in FHP play, with code available. — [arXiv html](https://arxiv.org/html/2511.08174); [GitHub](https://github.com/rpSebastian/DeepPDCFR)
- ESCHER removes importance sampling and beats DREAM and NFSP in its own paper — [arXiv 2206.04122](https://arxiv.org/abs/2206.04122v2). It was noncompetitive with tuned PPO/MMD/PPG in the ICLR 2026 independent benchmark — [arXiv 2502.08938v2](https://arxiv.org/html/2502.08938v2).
- Tabular GPU CFR (GPU-CFR, Kim & Sandholm 2026) is fast but limited to explicitly enumerated games, with the largest test game at 276k infosets. — [arXiv 2609.11923](https://arxiv.org/html/2609.11923); [arXiv 2605.14277](https://arxiv.org/abs/2605.14277)
- The fishing card game Pasur has been tackled with GPU-accelerated CFR (2025). — [arXiv 2508.06559](https://arxiv.org/abs/2508.06559)

### Inferences
- **Method-by-method fit summary (inference from the findings above):**
  - **Deep CFR (2019):** guarantees an average-iterate ε-Nash given bounded approximation error. Variance is low with ES, but each traversal is exponential in Matgo depth, and it retrains from scratch with large reservoirs. GPU mapping is good for branching but the cost explodes. Fit: poor unless ES is depth-limited.
  - **SD-CFR (2019):** lower approximation error, but it needs all past nets at inference, so distillation is required for a single network. Fit: moderate, as a training trick only.
  - **DREAM (2020):** OS plus baseline, model-free, Nash guarantee in two-player zero-sum. Variance is high but mitigated. One trajectory per traversal matches batched self-play. Fit: good baseline.
  - **ESCHER (2023):** no importance sampling and low variance, but it depends on Q(h,a) quality and coverage under a uniform sampling policy, and was weak in the 2026 benchmark. Fit: moderate. Q(h,a) could exploit full-state features.
  - **VR-DeepDCFR+/PDCFR+ (2025/26):** fastest neural CFR convergence published, bootstrapping keeps training cheap, and the code is open. Fit: best CFR-family candidate.
  - **Tabular GPU CFR (2024-26):** exact, but only on subgames or abstractions. Fit: use it for evaluation (subgame exploitability) and endgame ground truth.
- Short episodes (about 33 steps) are favorable for OS variance, because importance weights compound over about 16 own decisions rather than hundreds. Heavy chance is neutral across schemes. Score rewards need fixed linear rescaling.
- Exploitability cannot be computed exactly for full Matgo. Plan approximate best-response (an RL best response against the frozen average policy, using the 700k games/s simulator) plus tabular-CFR subgame exploitability as the evaluation protocol.

### Gaps
- No published application of any CFR-family method to Go-Stop/Matgo/Hwatu was found.
- No evidence was found on how neural CFR compares with PPO/MMD in high-chance, short-horizon card games specifically. The ICLR 2026 comparison used deterministic phantom/dark board games.
