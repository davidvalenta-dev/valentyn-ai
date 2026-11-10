# Architecture and trust boundaries

## Implemented system

The smallest coherent system combines reasoning, verification, memory, and evaluation. Meta-reasoning and evolution share a policy representation rather than adding independent agents. No language-model chatbot is required.

| Requested capability | Current implementation | Boundary / next extension |
|---|---|---|
| A. Mathematical reasoning | Constructive propositional natural deduction, forward chaining, case splitting | No quantifiers, real analysis, algebra, or arbitrary Lean tactics yet |
| B. Verification | Pinned Lean 4.24.0, restricted term grammar, compiler check, explicit axiom audit | Trusted compiler/toolchain and standard library; not a Python correctness proof |
| C. Discovery | Seeded grammar generates conjectures and attempts proofs or finds counterexamples | No literature novelty checking; generated results are elementary |
| D. Learning | Seven-feature linear reward prediction for three strategy choices, trained from verified attempts | Small supervised contextual learner; not foundation-model training or full RL |
| E. Meta-reasoning | Predict which strategy should work economically from formula structure | No autonomous scientific judgement |
| F. Evolution | Numeric policy mutation, validation scoring, per-problem nonregression gate, lineage | Does not modify source code or invent new inference rules |
| G. Memory | SQLite attempts, accepted proof sources/hashes, counterexamples, exhausted problems, experiment records | No automatic abstraction extraction or theorem retrieval in search yet |
| H. Evaluation | Separate fitting/selection/test splits; canonical structural dedup; all fixed-strategy baselines; deterministic node counts | Tiny public suite; not a benchmark of research mathematics |

## Proof search

A state is `(goal, context, remaining_depth)`, with a path set for cycle detection. Each context entry carries a proof term. Cheap conjunction projections and modus ponens saturate the context. A recursive search then tries introduction, implication application, and disjunction elimination in different orders. Iterative deepening shares one node/time budget across depths. At most one proof is returned; Lean decides whether it is accepted.

Classical truth-table search finds counterexamples for formulas with at most ten atoms. Passing all assignments is never converted into a proof. The proof engine is constructive, so excluded middle and double-negation elimination are expected examples of valid classical statements it does not establish. This is an intentional rule-system boundary, not evidence that those statements are false.

Formula input is restricted to 128 syntax nodes, 4096 characters, and bounded nesting. Search is limited to 1,000,000 nodes, depth 64, and 300 seconds. Proof terms have a 100,000-character limit. Axiom declarations, tactics, comments, native execution, and arbitrary identifiers cannot cross the proof-term lexical boundary. Lean still rejects ill-typed terms made from allowed tokens.

## Learning and improvement

For features `x(g)` and strategy `s`, the predictor is `r_hat(g,s) = w_s · x(g)`. Updates use `w_s ← clip(w_s + η(r - r_hat)x, -10, 10)`. Weights start at zero. Features are bias, size, implication count, conjunction count, disjunction count, falsity count, and atom count. Reward is zero without accepted verification; otherwise it is `1 - 0.25 min(nodes / budget, 1)`.

This reward strongly favors solving a task within an attempt. It is not a calibrated value function and does not estimate mathematical importance. Fitting replays measured observations; training epochs do not represent additional independent proof discoveries.

Evolution mutates one weight, validates the entire policy schema, and scores validation results lexicographically by `(verified_count, -nodes)`. A mutation must strictly improve that score and preserve the incumbent's solved-task set. Promotion guarantees apply only to the observed validation run. Repeated selection can overfit this small validation set.

The formal `promote` theorem models a scalar incumbent update; it does not prove the exact Python lexicographic promotion function. `derivation_sound` proves the inference rules preserve truth given a true context. It does not prove that every Python search branch implements those rules correctly. Per-result compilation is the operational safeguard.

## Worker isolation

Candidate policies are JSON, bounded to finite weights of magnitude at most ten. No candidate code is imported or evaluated. The trusted evaluation worker runs under `python -I`, in a fresh temporary working directory with a minimal environment. CPU, open-file and file-size limits apply; the parent enforces a wall timeout. There is no OS-level network prohibition or general memory sandbox. Process isolation must not be advertised as secure execution of arbitrary source.

The parent retains the evaluation problems and independently verifies returned proofs. Candidate workers cannot change the split or promote themselves. Policy files, raw logs, and code hashes support review. A malicious local process with write access to the checkout or database remains outside the current threat model.

## Extension decisions

First expand evaluation diversity and theorem reuse. Add arithmetic and typed term synthesis only with separate parsers, formal semantics, and immutable evaluators. Quantifiers and Mathlib should follow a memory/disk assessment. An optional pretrained model may propose structured candidates later; it must not control acceptance, corpus selection, or report generation. Training small policy/value networks from random initialization is useful for architecture experiments; recreating language pretraining is not a prerequisite for testing a reasoning hypothesis.
