# Project Valentyn: research update, November 10, 2026

This review extends [the initial landscape](LITERATURE.md). Publication dates below refer to the linked version or first preprint, not necessarily journal issue dates. “Reported” means the authors' result; none of these external systems was reproduced here. Availability descriptions distinguish released artifacts from a verified end-to-end reproduction. Hardware totals that were not inspected are marked explicitly rather than estimated from headlines.

## Recent work and its implications

### 1. NanoProof — Kripner and Straka, October 8, 2026; preprint

[Paper](https://arxiv.org/abs/2610.11605), [code](https://github.com/Kripner/nanoproof).
Factorized AND/OR search uses a shared Transformer to predict tactics and value bins, with staged supervision and RL. Authors report 50.8% MiniF2F-Test pass@16. The 1B model's reported run used four A100s for 16.3 hours of supervised training and four MI250 accelerators for 45 hours of RL; final evaluation also consumed substantial accelerator time. Data extraction, training, weights, and Lean interaction are released. Incremental verification and coupled metavariables deserve particular care. For Valentyn, reproduce typed local-state supervision and a tiny policy first; full pretraining/RL is a later external-compute experiment. The paper's cost comparisons require matching what training and evaluation costs are included.

### 2. EasyProver — Ganguly et al., October 28, 2026; preprint

[Paper](https://arxiv.org/abs/2610.38174).
Recursive agents isolate reasoning contexts and use asynchronous tactic exploration and an elastic elaboration service, backed by an existing frontier model. Reported results include 669/672 PutnamBench problems and 1,054/2,000 sampled Aegis conditions under a 30-minute per-statement budget. Aegis generates Rust-to-Lean verification tasks with implementation details. Full inference cost and public artifact completeness were not audited in this pass. The result does not isolate agent architecture from model strength. Valentyn should borrow independent contexts only after a minimal baseline shows a bottleneck; hundreds of parallel workers are inappropriate on the initial machine.

### 3. Formal Conjectures — Firsching et al., May 13, 2026; preprint

[Paper](https://arxiv.org/abs/2605.13171).
Provides an evolving Lean benchmark with 2,615 statements in the reported snapshot, including 1,029 open conjectures and 836 solved problems. Collaborative formalization and proof/disproof feedback audit statement fidelity. This is evaluation infrastructure, not a new trained architecture; system compute depends on the selected prover and frozen subset. Formal statements are openly maintained, but an evolving corpus needs versioned snapshots. “Open” should not be treated as permanently uncontaminated after public proofs appear. Valentyn should eventually consume frozen, expert-reviewed subsets, retaining original problem sources and specification-review records.

### 4. FrontierMath Erdős — Adamczewski and Bloom, September 6, 2026; preprint

[Paper](https://arxiv.org/abs/2609.25050).
Evaluates systems on 68 then-open Erdős problems with the same autonomous budget, $300 per problem, requiring Lean proofs or refutations. The reported leading score is approximately 3%; other evaluated systems score zero. No architecture is trained by the benchmark. The principal contribution is comparable research-task evaluation, rather than selected success stories. Exact problem releases and current solved status must be audited before reuse. Valentyn cannot compare its synthetic logic percentages with this benchmark: the distributions, assistance, and costs are radically different. Use its evaluation discipline when research-level capability is available.

### 5. Aletheia — Feng et al., February 10 / March 6, 2026; preprint

[Paper](https://arxiv.org/abs/2602.10177).
An advanced Gemini-based research agent repeatedly develops, checks, and revises natural-language mathematics while using literature tools. Authors describe research contributions and four autonomous open-question solutions in a larger evaluation. Prompts and model outputs are released; the underlying model is not a from-scratch local reproduction. Full compute totals were not audited here. Natural-language verification differs from kernel checking, and autonomy needs explicit accounting. Applicable lesson: retain research provenance and human-interaction records, while separating model confidence, mathematical review, and formal acceptance.

### 6. mini-Lean-agent — Ortega et al., October 12, 2026; workshop paper/preprint

[Paper](https://arxiv.org/abs/2610.17331).
A minimal shell-facing coding agent delegates final acceptance to an independent evaluator. Authors report 7/12 Putnam 2025 problems with one base model under a $2/problem cap, and 12/12 with a stronger model at $1.28 average cost. The benchmark is small, and different model results should not be conflated with harness improvements. Full release artifacts were not audited here. Valentyn should include minimal baselines before adding orchestration, and never give a model control over the final evaluator. Local from-scratch models require a different, smaller initial task distribution.

### 7. SelfSearch — Yang, Kong, and Jo, September 29–30, 2026; preprint

[Paper](https://arxiv.org/abs/2609.37968).
Agents reuse records of earlier self-modification episodes to propose changes without downstream rewards during proposal search. Authors report gains across six settings, including up to 11.2 percentage points on Terminal-Bench and lower execution cost on shared solved tasks. One search-cost example is $4.03; that is not complete model-training or downstream-evaluation cost. Artifact completeness was not audited. Reward-free proposal generation still needs independent outcome evaluation. For Valentyn, archived failed experiments can inform proposals, but test results must stay outside automatic selection.

### 8. AlgoEvo — Qiu et al., September 14 / 27, 2026; preprint

[Paper](https://arxiv.org/abs/2609.15820).
Combines runtime-guided code editing, a reusable design-skill collection, and hierarchical within/across-task experience. Authors report strong results across six tasks, in some settings using 7% of comparison evaluation budgets. It builds on an LLM rather than training an intelligence from nothing. Absolute training/inference costs and release completeness were not audited here. Evaluator design and transferable skill selection remain potential confounders. Valentyn should test memory and proposal-policy ablations independently before integrating both, and include the cost of creating the reusable experience bank.

### 9. Station — Chung, Du, and Wesley, August 24 / September 14, 2026; preprint

[Paper](https://arxiv.org/abs/2608.23691).
Multiple model families conduct open-ended mathematical work through a shared research environment and accumulated papers. Authors report new results on five of twelve construction problems, with exact constructions or Lean proofs; code and dialogues are released. This is evidence about a combined research process, not proof that open-ended multi-agent organization is necessary. Full total cost was not audited. Reusable mathematical artifacts are relevant to Valentyn; reproducing the collective workflow locally is neither required nor currently justified. Novelty review and precise attribution must accompany verified constructions.

### 10. AI-driven formal proof search — Tsoukalas et al., May 21 / June 8, 2026; research paper

[Paper](https://arxiv.org/abs/2605.22763), [proof artifacts](https://github.com/google-deepmind/alphaproof-nexus-results).
Authors report 9/353 open Erdős problems and 44/492 OEIS conjectures, with a few hundred dollars per problem. A simpler generation/checking agent reproduces the Erdős successes at higher cost on difficult cases. Full proprietary inference/training is not reproduced by released proofs. The project brief cites an October Science publication; the journal metadata endpoint was unavailable during this update, so the accessible preprint is the inspected version. The central implication is to test mechanisms against simple baselines and review whether formal statements faithfully capture the research questions.

## Foundational work that changes implementation choices

### 11. Goedel-Prover-V2 — Lin et al., August 5, 2025; ICLR 2026

[Paper](https://arxiv.org/abs/2508.03613), [release](https://github.com/Goedel-LM/Goedel-Prover-V2).
Scaffolded synthetic tasks, verifier-based revision, and checkpoint averaging support supervised/expert-iteration and RL training. Released models include 8B and 32B variants; the paper reports 84.6% and up to 90.4% MiniF2F pass@32 for particular configurations. These are not comparable to greedy single-attempt results. Training-scale cost was not independently audited; full model training is beyond the conservative local prototype. Use staged tasks and correction ablations, preserving proof diversity and measuring actual verification success rather than token loss alone.

### 12. Darwin Gödel Machine — Zhang, Hu, Lu, Lange, and Clune, May 29, 2025 / March 12, 2026; ICLR 2026

[Paper](https://arxiv.org/abs/2505.22954), [code](https://github.com/jennyzzt/dgm).
An archive of executable agents supports evolutionary modification and empirical selection. Authors report SWE-bench improvement from 20% to 50% in their setup. Search uses existing foundation-model capabilities; benchmark gains do not establish indefinite recursive acceleration. Public code enables experiments, but complete cost and reproduction conditions still need auditing. Valentyn currently mutates data/configurations, not unrestricted executable code. Archive diversity, immutable evaluation, and cost-matched random search are needed before stronger self-improvement claims.

### 13. AlphaEvolve — Novikov et al., June 16, 2025; technical report

[Paper](https://arxiv.org/abs/2506.13131), [result verification](https://github.com/google-deepmind/alphaevolve_results).
LLM-generated program changes are selected by evolutionary search and objective evaluators. Reported applications span mathematical constructions and production algorithms. Released constructions are not the whole production agent or model-training pipeline; complete compute was not audited here. Applicability depends on evaluators that measure the intended property. Valentyn can first evolve typed search configurations under fixed budgets, then test whether learned reusable constructions reduce future discovery costs. Mathematical importance and broad generality cannot be inferred from one numerical objective.

### 14. DeepSeek-Prover-V2 — Ren et al., April 30 / July 18, 2025; preprint

[Paper](https://arxiv.org/abs/2504.21801), [release](https://github.com/deepseek-ai/DeepSeek-Prover-V2).
Recursive subgoal decomposition supplies initial proof supervision before RL. Authors report 88.9% MiniF2F test performance for the 671B variant and 49/658 PutnamBench solves under their settings. Released weights/code permit inference research; reproducing teacher generation and training is a separate cost. Subgoal quality can dominate performance, and sampling budgets matter. Valentyn's local experiment tests the smaller question of factorized proof-state learning without importing a pretrained teacher. It does not reproduce the paper's broad mathematical ability.

### 15. LeanDojo — Yang et al., June 27 / October 27, 2023; NeurIPS 2023

[Paper](https://arxiv.org/abs/2306.15626), [tooling](https://github.com/lean-dojo/LeanDojo).
Provides programmatic Lean interaction and premise annotations, enabling retrieval-augmented ReProver and novel-premise evaluation splits. Authors describe roughly a GPU week of training and a 98,734-theorem benchmark. Tooling, datasets, and models are open. Retrieving only premises accessible at the theorem's position is crucial; library-version changes can contaminate evaluation. Valentyn should later extract typed transitions and dependencies from a pinned Mathlib snapshot, after its small synthetic experiment demonstrates a reason to pay extraction and training costs.

### 16. HTPS — Lample et al., May 23, 2022; NeurIPS 2022

[Paper](https://arxiv.org/abs/2205.11491), [Evariste](https://github.com/facebookresearch/Evariste).
Hypergraph proof search and online training handle all-required subgoals rather than treating proofs as ordinary game trajectories. Reported Metamath performance grows from 65.4% to 82.6% with online learning; task settings differ from Lean. Full training cost was not audited in this update. Some historical code releases do not provide every artifact needed for exact reproduction. The relevant design insight is factorization and local-state learning; a small policy plus explicit typed actions offers a cheaper initial test than full MCTS infrastructure.

### 17. FunSearch — Romera-Paredes et al., December 2023 online / Nature 2024

[Publication](https://www.nature.com/articles/s41586-023-06924-6), [official project](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/).
Evolutionary program proposals from a pretrained model are scored by executable evaluators; published applications include cap-set constructions and bin packing. Result/code availability does not recreate the proprietary model, and a full compute audit was not completed here. A fast numerical evaluator can accelerate discovery but cannot certify an arbitrary universal theorem. For Valentyn, finite construction search should retain exact witnesses and a separate checker; promote reusable algorithms only after testing unseen sizes and counting evaluation cost.

### 18. Mamba-2 — Dao and Gu, May 31, 2024; ICML 2024

[Paper](https://arxiv.org/abs/2405.21060), [code](https://github.com/state-spaces/mamba).
Structured state-space duality connects recurrent computations and attention; reported kernels improve layer speed while retaining competitive language modeling. Efficient sequence processing is not evidence of stronger mathematical discovery. Official optimized training paths require a portability/hardware audit before an M2 claim. Valentyn uses a portable GRU as an inexpensive recurrent control now. A true Mamba-versus-attention experiment should match data, parameters, training work, and full proof-solving cost, rather than comparing only throughput or context length.

### 19. Titans — Behrouz, Zhong, and Mirrokni, December 31, 2024; preprint

[Paper](https://arxiv.org/abs/2501.00663).
Combines attention with neural memory updated at test time, reporting benefits on several sequence tasks. Its long-context demonstrations do not establish proof planning or stable lifelong mathematical learning. Official end-to-end artifact availability and total compute were not audited. Valentyn should compare a small recurrent state representation with explicit verified retrieval before implementing adaptive long-term neural memory. Memory should be charged for construction, retrieval, and regression risk; a larger effective context is a possible confounder.

### 20. DreamCoder — Ellis et al., June 15, 2020; research paper

[Paper](https://arxiv.org/abs/2006.08381).
Wake/sleep learning alternates program synthesis, symbolic-library growth, and learned search guidance. The authors demonstrate compositional abstractions across several program domains. Full published compute and current reproduction compatibility were not audited here. Library growth can increase branching unless useful abstractions amortize discovery and lookup costs. Valentyn should first test exact certified theorem reuse, then bounded macro extraction, with held-out composition depths. Merely storing accepted proofs in SQLite is not equivalent to learning an abstraction library.

### 21. AlphaProof — Hubert et al., Nature 2025

[Publication record](https://research.google/pubs/olympiad-level-formal-mathematical-reasoning-with-reinforcement-learning/).
AlphaZero-style learning over formal proofs and problem-specific test-time RL demonstrate grounded mathematical reasoning. Reported competition performance used multi-day computation and millions of related training problems; it is not a local quick-start recipe. The reviewed publication does not give a ready complete training reproduction. Valentyn tests local policy learning and proof decomposition first. A learned value function is a subsequent ablation, not something to add before the policy-only baseline establishes remaining search headroom.

## Announcements and tooling, kept separate from the paper count

[AlphaProof Code](https://google-deepmind.github.io/code-verification/), November 10, 2026: the project announcement reports approximately 6.5 million lines of Lean proof over roughly 26,000 lines of trusted specifications/semantics for a libsignal build. The project page/full report intermittently failed retrieval; the official indexed announcement was inspected, not a complete proof audit. Verification depends on the specification and exact build scope. Do not translate this into a claim that every Signal application, side channel, or security property is verified.

[MLX-LM](https://github.com/ml-explore/mlx-lm) provides Apple Silicon inference/fine-tuning and quantized-model workflows. It is useful for optional pretrained instruments, not necessary for the independent small-model track. This release uses already-installed PyTorch with actual Metal training. It does not report an unperformed MLX benchmark. Four-bit inference is worth testing for a later 0.5B–1.5B proposal model; quantization is unnecessary for the sub-megabyte policies now being studied.

## Three gaps worth attacking

1. **Compositional transfer versus proof-pattern imitation.** A whole-proof model can learn short patterns but fail when familiar steps must be composed further. The executable local experiment compares full-term generation with a neural policy over typed local goals, including an untrained policy and legality-matched random controls. The novelty question is cost-effective transfer and abstraction, not inventing factorization itself.
2. **Complete-run cost versus selected success rates.** More beam search can improve solving while degrading verified capability per unit of compute. Valentyn records both inference and verification time and blocks automatic decoding promotion when the observed inference cost per solve rises. That gate is exploratory, not a proof of economic superiority.
3. **Verified artifacts versus reusable scientific knowledge.** A checked proof may be known, mis-specified, or useless to future search. Future work must couple typed retrieval/abstraction with exact provenance, novelty review, and independent statement auditing. SQLite storage alone does not close this gap.

The strongest affordable first hypothesis is the first gap. Factorized local decisions reduce the length of the model's prediction problem; a typed construction environment exposes current hypotheses instead of forcing the network to reconstruct them from a long token history. The critical controls are identical inference rules for learned/untrained/random policies and an unchanged external Lean gate.
