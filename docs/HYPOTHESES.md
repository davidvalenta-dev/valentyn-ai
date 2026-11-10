# Research hypotheses and first experiments

These are proposed research directions, not established discoveries. The individual ingredients have substantial prior art. “New” below identifies a proposed distinction to investigate, not a verified priority claim. A proper novelty claim requires a deeper literature review and experiments. Six hypotheses are listed so that the program does not depend on one architectural bet.

Update: the program now includes **H7**, below, plus actual from-scratch proof-generating models. The first six hypotheses and their initial negative results are retained as historical research proposals. The new sequential state policy is not the graph-memory architecture proposed in H6.

## Priority ranking

Scores are judgement calls, 1 low to 5 high; low cost is desirable. Novelty is the estimated distinctiveness of the proposed experiment, not a claim of invention.

| Hypothesis | Cost | Novelty | Feasibility | Potential impact | Execution status |
|---|---:|---:|---:|---:|---|
| H1: structure-sensitive strategy selection | 1 | 1 | 5 | 2 | Implemented and measured |
| H2: regression-constrained strategy evolution | 1 | 2 | 5 | 3 | Implemented and measured |
| H3: adaptive proof-search portfolios | 1 | 2 | 5 | 3 | Equal-budget fixed-portfolio probe implemented |
| H4: boundary-guided curricula | 2 | 3 | 4 | 4 | Curriculum-order probe implemented; boundary adaptation proposed |
| H5: certified abstraction transfer | 2 | 4 | 3 | 5 | Family-transfer diagnostic implemented; abstraction learner proposed |
| H6: typed recurrent search memory | 3 | 3 | 3 | 5 | Architecture experiment specified; not implemented |

The first five have executable low-cost probes. For H3–H5, those probes test prerequisites or simple versions; they do not implement the complete proposed algorithms. No unimplemented component is counted as a demonstrated capability.

## H1. Does structure-sensitive selection beat a good constant strategy?

**Motivation.** Different inference orders encounter different search trees. Selecting the order from structural features may amortize trial-and-error. **Distinctive experimental question.** Measure the benefit after explicitly including every constant strategy, structural holdouts, and fitting cost. Contextual algorithm selection itself is established.

**Algorithm and formulation.** Learn `w_s` from verified attempt reward, select `argmax_s w_s · x(g)`. Use zero-initialized weights and identical symbolic rules. Formula features and update equations are documented in ARCHITECTURE.md.

**Executable experiment.** `valentyn train --seed 0`, then `valentyn benchmark`. **Baselines.** All three fixed strategies; subsequent runs should include uniform random selection and an oracle strategy per task to expose remaining headroom. **Success criterion.** On a newly frozen suite over five seeds, at least 10% fewer nodes than the best fixed policy without a solve-rate regression; include learning cost amortized over future tasks. **Falsification.** No gain over the best constant policy, or gains disappear on new families. **Compute.** CPU-only; 36 training searches plus Lean checks and 56 test attempts for the bundled suite. **Scaling.** Replace hand-crafted features with a small typed-graph policy only after the simple learner establishes headroom.

**Hidden assumption.** A single initial strategy choice is sufficient. Actual results already challenge the contextual part: the recorded learner chooses `split` throughout the test set.

## H2. Can constrained evolution improve reasoning without observed regressions?

**Motivation.** Aggregate gains can conceal lost capabilities. **Proposed distinction.** Evolve policies against a protected solved-task set and structurally distinct task families, keeping the verifier immutable. Evolution and nonregression tests are existing techniques; their interaction and generalization are the research question.

**Algorithm.** Propose `θ' = θ + εe_i`; promote only if `score(θ') > score(θ)` and `Solved(θ) ⊆ Solved(θ')` on validation. Score lexicographically prioritizes accepted proofs over node cost. **Executable experiment.** `valentyn evolve --generations 12 --seed 0`, followed by `valentyn benchmark --policy .valentyn/evolved-policy.json`. **Baselines.** Unchanged learned policy, fixed strategies, and budget-matched random search over policies. The last baseline is a required larger follow-up. **Success criterion.** At least 5% lower unseen-task cost with no lost validation solves and no statistically supported unseen solve-rate decline. **Falsification.** No promotions, or validation gains reverse on new families. **Compute.** `(generations + 1) × 6` local searches and proof checks, plus subprocess overhead. **Scaling.** Extend the mutable representation to typed search-controller programs, only after adding an OS sandbox and accounting for evaluation costs.

**Hidden assumption.** Small numeric mutations explore meaningful alternatives. The first 12 trials produced zero promotions; that is a negative result, not recursive acceleration.

## H3. Do failure-sensitive portfolios beat more depth in one strategy?

**Motivation.** Hard subgoals can absorb all search effort. Complementary strategies might be more valuable than repeatedly deepening the same search. **Proposed distinction.** Allocate budgets using observed failure-state structure, not only initial formula features or independent restarts.

**Algorithm.** Initially split total budget `B` across strategies. The full proposal learns a controller `b_s(g, trace)` with `Σ b_s ≤ B`, rewarding marginal verified solves per additional node. **Executable first probe.** `python scripts/ablations.py portfolio --nodes 1000 --report .valentyn/portfolio.json` compares a sequential fixed portfolio with a focused strategy under equal aggregate node and allocated search-time budgets. This does not yet learn the allocation controller. **Baselines.** Fixed focused strategy; follow-up compares best fixed strategy, equal allocation, and learned allocation. **Success criterion.** At least 10% more verified solves at equal end-to-end compute on difficult family-held-out tasks. **Falsification.** Restart overhead eliminates gains, or the same best fixed strategy dominates. **Compute.** CPU only; at most four search attempts per validation task in the probe. **Scaling.** Hypergraph search with shared subgoal caches and learned marginal-value estimates.

**Hidden assumption.** Strategies are complementary. Merely reordering the same rules may provide too little diversity; an oracle portfolio analysis should precede training.

## H4. Can boundary-guided curricula improve transferable reasoning?

**Motivation.** Always solved goals provide little learning signal; impossible or mismatched goals waste effort. **Proposed distinction.** Select conjectures near an empirically measured search-budget boundary while preserving coverage of structurally held-out families and counterexamples.

**Algorithm.** Maintain per-family solve estimates and choose generated tasks with high uncertainty and estimated transfer value, penalizing repeats. Candidate objective: `U(g) = p(g)(1-p(g)) × Transfer(g) / expected_cost(g)`. Transfer must be estimated on a separate development family, not test results. **Executable first probe.** `python scripts/ablations.py curriculum --seed 0 --report .valentyn/curriculum.json` compares random, easy-to-hard, and hard-to-easy observation schedules at exactly 360 updates per arm. It measures schedule sensitivity, not autonomous curriculum discovery. **Baselines.** Uniform random tasks and fixed size curriculum. **Success criterion.** At least 10% lower data/compute to reach a preregistered validation target over five seeds, followed by unchanged-family and new-family evaluation. **Falsification.** Benefit comes solely from repeated easy examples or disappears under equal update/compute budgets. **Compute.** A few dozen measured local searches, then small CPU learner updates; no foundation model training. **Scaling.** Generate typed arithmetic/graph tasks with certified feasibility and reserve external research problems for final evaluation.

**Hidden assumption.** Search difficulty predicts conceptual value. It might instead measure an implementation defect or a poor representation.

## H5. Can certified abstraction memory produce compositional transfer?

**Motivation.** Reproving the same pattern wastes compute, but a giant unfiltered lemma library can increase branching. **Proposed distinction.** Retain a proof-certified abstraction only if its measured future compute savings exceed proof construction, retrieval, and library-maintenance costs across held-out composition depths.

**Algorithm.** Extract repeated typed subproofs as candidate macros, kernel-check their generalization, and score `Value(m) = E[future saved nodes] - λ(description_length + lookup_cost + verification_cost)`. Freeze a library before final testing. **Executable diagnostic.** `python scripts/ablations.py transfer --seed 0 --report .valentyn/transfer.json` holds disjunction training tasks out and compares transfer at equal learner updates. This detects family dependence in the current learner; it does not extract or reuse macros. **Full experiment to implement next.** Compare no memory, exact theorem retrieval, size-limited random macros, and certified cost-selected abstractions on generated proofs with held-out composition depth. **Success criterion.** At least 20% lower total lifecycle cost after including library construction, no solve-rate loss, and benefit at larger unseen composition depths. **Falsification.** Gains are only exact-match caching or are erased by lookup cost. **Compute.** CPU and a few thousand small Lean checks initially; later a small learned retriever. **Scaling.** Mathlib-compatible lemma synthesis with explicit side conditions and domain-specific theorem families.

**Hidden assumption.** Repeated surface patterns correspond to reusable mathematical concepts. The extracted generalization may be invalid, too weak, or irrelevant outside its origin family.

## H6. Does typed recurrent memory improve long-horizon search?

**Motivation.** Flat token sequences can obscure shared dependencies and waste memory on repeated proof states. **Proposed distinction.** A recurrent graph memory stores typed subgoals, verified dependencies, and failed applications, with separately gated persistent and task-local updates.

**Architecture.** Encode a proof-state graph `G_t`, update `h_(t+1) = GRU(h_t, GraphEncode(G_t), feedback_t)`, and predict rule priors plus expected remaining cost. Compare a tiny recurrent model, a small attention model, and the current linear learner at matched parameters/data. All logical actions remain verified.

**Experiment specification.** Collect traces with the existing prover; freeze train/family-validation/test partitions; train all small policies from random initialization on identical traces; evaluate unseen proof depth and retained earlier domains. The neural training/graph encoder is **not implemented in this release**. The executable trace-producing starting point is `valentyn train` and its archived search observations; finer per-node traces must be added before this experiment is runnable in full. **Success criterion.** At least 15% higher solve rate at fixed complete-run compute, with no retained-domain decline. **Falsification.** Gains vanish when parameter count, trace count, context size, and memory lookup time are matched. **Compute.** Start below 1M parameters with batches fitting in a conservative 2 GB process allowance; actual memory must be measured. **Scaling.** Quantified and multi-domain proof graphs only after the small model demonstrates a real benefit.

**Hidden assumption.** More expressive memory is the bottleneck. Better search rules or more diverse tasks may matter more than neural architecture.

## H7. Does learned local proof construction transfer more economically than whole-term generation?

**Formulation.** At fixed teacher-proof data and generation caps, compare `Solve(π_state, typed_environment, B)` with whole-term generation and with learned/untrained/random policies in the same environment. Optimize the full curve of accepted proofs versus inference plus verification cost, not just action accuracy. The model predicts local actions from a goal and typed hypotheses; a fixed interpreter constructs terms, and Lean remains independent.

**Mechanism.** Local states expose what is currently known, reducing the implicit proof-state reconstruction burden of long token generation. Composing learned inference choices may transfer to longer or unfamiliar proof patterns. **Prior work.** Factorization, neural policies, and execution guidance already appear in AlphaProof, HTPS, and NanoProof. Our prospective contribution is a carefully controlled small-memory transfer/cost study and, later, cost-selected reusable abstractions; it is not a claim to have invented factorization.

**Executable experiment.** Generate data with `valentyn model-data`; train whole-term Transformer/GRU models with `model-train`; train local actions with `model-train --kind state`; evaluate with `model-evaluate`. `scripts/frozen_transfer.py` freezes checkpoint hashes before generating a fresh suite. `--random-policy-seed` supplies an environment-matched random control. All completed neural proofs pass Lean.

**Baselines and ablations.** Include the best symbolic prover, proof retrieval, untrained sequence models, untrained state policy, five random-policy seeds, greedy versus wider search, and matched data/parameter/compute variants. Legality masking and projections must be held constant when estimating the value of learning. Remove or equalize each representation advantage in a follow-up experiment.

**Held-out data.** Exclude theorem-family and canonical structure overlap from fitting. Evaluate a newly generated suite after model freezing, then independently sourced arithmetic/finite-combinatorics tasks. Synthetic logic alone cannot establish broad mathematical generalization.

**Success criterion.** A preregistered five-training-seed study shows at least a 15-point solve-rate gain over environment-matched untrained/random controls on fresh families, with a better complete-run compute frontier; benefit survives representation and data-volume ablations. **Failure criterion.** Gains disappear under the same typed environment, are explained by the random policy or copied proof patterns, or vanish on a second mathematical domain.

**Requirements.** The present models fit in small local processes and train with Metal; actual parameter, token, time, and memory counters are recorded. Initial tests do not require a 0.5B+ pretrained model. Larger Mathlib experiments need profiled CPU elaboration and an external training budget only after a positive small-domain result.

**Publishable contribution.** A new abstraction/credit-assignment mechanism with replicated cross-family or cross-domain gains, verified artifacts, honest cost accounting, and reproducible low-memory implementation. The current synthetic pilot alone is not claimed as a publishable algorithmic breakthrough.

## Follow-up experiment matrix for all seven hypotheses

| Hypothesis | Required confounder control / ablation | Held-out data | Evidence needed for a paper | Priority |
|---|---|---|---|---|
| H1 strategy selection | Best constant/oracle, parameter count, fitting cost, feature ablation | New formula families and sizes | Gains beyond a constant selector across seeds | Low after initial null result |
| H2 constrained evolution | Random mutation search at equal total cost; remove regression gate | Frozen new families | Transferable improvements, not selection-set tuning | Medium |
| H3 portfolios | Equal node, latency, and total proposal budgets; independent versus shared caches | Difficult generators with oracle headroom | Improved compute frontier beyond best fixed/restart baseline | Medium |
| H4 curricula | Equal updates/data/compute; uniform and size schedules; remove uncertainty/transfer terms | Independently generated families | Better learning efficiency and external transfer | Medium |
| H5 abstraction memory | Exact-match cache, random macros, matched lookup cost; remove certified generalization | Held-out composition depths and a second domain | Lifecycle cost savings from reusable concepts | High scientific value; higher difficulty |
| H6 recurrent graph memory | Match parameters/context/data; remove persistent memory; retention battery | Longer proofs and prior domains | Transfer gains without context-size or forgetting confounders | High difficulty; defer expensive training |
| H7 local neural construction | Same legality/projection environment for learned/untrained/random; match data and search | Fresh frozen composition suite, then arithmetic | Replicated transfer/cost gains plus a substantive new mechanism | Highest immediate affordability |
