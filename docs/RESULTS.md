# Recorded results — November 10, 2026

Executed locally on macOS 14.4, arm64, Python 3.11.9, Lean 4.24.0. These are development experiments on elementary propositions. No result here establishes mathematical novelty, research-level theorem proving, or general intelligence.

## Verification

- **26 tests passed**, including real Lean verification, rejection of a type-incorrect candidate, input-injection checks, unavailable/wrong-toolchain failures, budget exhaustion, archive gating, deterministic search, policy validation, and isolated-worker execution. [Check log](../reports/checks.txt).
- **Nine named project theorems** built and passed an axiom-dependency audit with zero axioms, including inference-rule soundness and promotion monotonicity. The arithmetic step proof was rewritten to avoid foundational dependencies introduced through a standard-library arithmetic lemma.
- Saved reports contained **140 accepted-proof occurrences, comprising 33 distinct proof artifacts**. Every distinct artifact was independently reconstructed, hash-compared, and rechecked with Lean: **zero failures**. [Recheck report](../reports/proof-recheck.json).

The inference-rule soundness theorem applies to the formal derivation system. It is not a proof that the Python implementation or the learner is correct. Per-result Lean checking covers the actual returned proof terms.

## Training and held-out evaluation

Training measured three strategies on 12 training formulas: 36 observations, including 33 verified proof attempts and three counterexample outcomes. The learner replayed those observations for 100 epochs with seed zero. [Training report](../reports/train.json), [policy weights](../reports/trained-policy.json).

All evaluation arms received a 10,000-node cap, depth 24, and five-second search-time cap per task. Across 14 test formulas:

| Arm | Verified | Counterexamples | Exhausted | Rejected | Total search nodes |
|---|---:|---:|---:|---:|---:|
| Fixed focused | 10 | 2 | 2 | 0 | 1,353 |
| Fixed forward | 10 | 2 | 2 | 0 | 1,594 |
| Fixed split | 10 | 2 | 2 | 0 | 1,152 |
| Learned policy | 10 | 2 | 2 | 0 | 1,152 |

[Raw paired results and proof sources](../reports/baseline-vs-trained.json).

The learned policy used 14.9% fewer nodes than focused, but **exactly matched the best constant policy**, choosing `split` on every test formula. This is not evidence that contextual learning adds value. All strategies exhausted on excluded middle and double-negation elimination, which are not derivable in the constructive rule system used here. They found actual falsifying assignments for the two false goals.

These test formulas were excluded from automatic fitting and selection, but were visible during development. A valid proof was initially rejected because Lean's unused-binder linter was treated as an error. The generated module now disables only that linter while retaining compiler errors, other warnings-as-errors, and the axiom audit. The final run has no rejections. This development history limits any claim of independently held-out generalization.

## Policy evolution

Twelve seeded mutations, each evaluated on six validation formulas in an isolated worker, produced **zero promotions**. The initial and final validation score was four solved tasks and 427 search nodes. The evolved policy is identical to the trained policy. [Lineage](../reports/evolution.json), [evolved weights](../reports/evolved-policy.json).

Interpretation: weight mutation around this policy did not yield a useful improvement. It does not demonstrate compounding recursive self-improvement. A richer policy representation, more diverse strategies, and a larger validation suite are candidate next experiments, not explanations established by this run.

## Conjecture generation

Forty seeded grammar-generated conjectures produced **12 verified proofs**. Eleven were new to the local archive after the recorded training/evolution/benchmark sequence. [Generated formulas, outcomes, and proofs](../reports/discovery.json).

These are elementary propositions, not claimed discoveries in the mathematical literature. The local novelty count depends on prior archive contents and exact rendered formula strings; it does not establish novelty under logical equivalence.

## Controlled development probes

| Probe | Actual result | Interpretation |
|---|---|---|
| Equal-budget sequential portfolio | Both arms solved four validation tasks; portfolio used 897 nodes versus focused's 493 | More expensive without added solves on this suite |
| Curriculum order | Random, easy-to-hard, and hard-to-easy each solved four tasks using 427 nodes after 360 updates | No observed schedule effect |
| Withheld disjunction training family | Both policies solved four tasks; full training used 427 nodes, withholding disjunction tasks used 493 | Possible family sensitivity in cost; tiny suite and one seed do not establish reliable transfer |

[Portfolio report](../reports/portfolio.json), [curriculum report](../reports/curriculum.json), [transfer report](../reports/transfer.json).

All raw experiment reports share implementation digest `36b90e560143f2672d915f45242cc7249d5729217f7eb44f9d7123b2a664081d`. This hashes Python source, bundled corpus, Lean proof modules/toolchain/build settings, package configuration, and experiment scripts. Timing and attempt IDs may differ on reproduction. The test and policy digests are retained in the reports.

The most useful next scientific test is a larger, freshly frozen family/size-held-out suite and an oracle strategy comparison. If even the per-task oracle barely improves over a constant strategy, training a more elaborate selector is unlikely to be the right investment.
