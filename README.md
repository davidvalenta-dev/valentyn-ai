# Valentyn

**A research program in verifiable reasoning, mathematical discovery, and measured self-improvement.**

Valentyn starts with an executable symbolic reasoning system and a learning loop built from scratch. It runs locally on Apple Silicon without a GPU, model downloads, API keys, or Python runtime dependencies. The long-term research objective is increasingly general scientific discovery, including major open mathematical problems.

Version 0.1 is a bounded propositional prover with Lean verification, a small learned strategy selector, controlled policy evolution, generated conjectures, and an experiment archive. It does not demonstrate superintelligence, unrestricted recursive improvement, or new research mathematics. Those are research objectives with separate evidence requirements.

## Run on your Mac

Install Python 3.11 or newer, Git, and curl. From a terminal:

```sh
git clone https://github.com/davidvalenta-dev/Valentyn.git
cd Valentyn
sh scripts/setup.sh
. .venv/bin/activate
valentyn prove '(p & q) -> (q & p)' --output .valentyn/proofs/swap.lean
```

Setup installs the pinned Lean toolchain using Elan and builds the proof library. The initial Lean download needs internet and disk space. Subsequent reasoning experiments are offline. Mathlib is deliberately not required for this first small-memory version. Python sources live under `src/valentyn`, separately from `Valentyn/` Lean modules, including on case-insensitive macOS filesystems.

```sh
# Verify environment, tests, and all project theorem dependencies.
valentyn doctor
sh scripts/check.sh
python scripts/recheck_reports.py

# Collect verified feedback and train a strategy selector from zero weights.
valentyn train --seed 0 --report .valentyn/train-report.json

# Propose and test mutations on validation problems, retaining only improvements.
valentyn evolve --generations 12 --seed 0

# Compare the learned policy against ALL three fixed strategies on held-out tasks.
valentyn benchmark --policy .valentyn/evolved-policy.json

# Generate conjectures; classify false goals and verify successful proofs.
valentyn discover --count 40 --seed 17
valentyn history
```

Use `valentyn --help` and `valentyn prove --help` for options. `--project` and `--state` are global options and go before the command. `--nodes`, `--depth`, and `--timeout` limit each search. The `prove` exit codes are 0 verified, 1 error/rejection, 2 counterexample, and 3 search exhausted. Exhaustion is not disproof. `prove` accepts `p` through `z` (optionally numbered), `True`, `False`, `&`, `|`, `->`, `!`, and parentheses. It does not accept unrestricted Lean source or natural-language problems.

## What executes

1. A parser constructs a bounded formula tree.
2. Natural-deduction search introduces implications, constructs conjunctions/disjunctions, eliminates hypotheses, performs forward chaining, and splits cases. Three strategies change rule ordering.
3. A contextual linear learner selects a strategy using formula features. Only Lean-accepted results receive positive reward; cheaper proofs receive slightly higher reward.
4. Every candidate becomes a restricted proof term in an isolated temporary Lean file. The pinned compiler checks the theorem and reports its axiom dependencies. Acceptance requires successful compilation and **zero axiom dependencies**.
5. SQLite stores attempts, failures, counterexamples, proof sources, hashes, and experiment histories.
6. Evolution mutates numeric strategy-selection weights, evaluates in a resource-limited worker, and promotes only strict validation improvements without losing a previously solved validation problem. The test split is excluded from promotion.

Generated conjectures are only described as **new to the local archive**. The implementation does not search the mathematical literature or establish global novelty. The archive stores results but does not yet invent reusable mathematical abstractions.

## Actual evidence and research plan

- [Recorded experiments](docs/RESULTS.md), with raw machine-readable reports in `reports/`.
- [Architecture, trust boundaries, and implementation status](docs/ARCHITECTURE.md).
- [Primary-source literature review](docs/LITERATURE.md).
- [Six falsifiable research hypotheses and executable probes](docs/HYPOTHESES.md).
- [Six-phase research roadmap](docs/ROADMAP.md).
- [Reproduction and evaluation protocol](docs/EVALUATION.md).

The current inference rules have a Lean soundness proof in `Valentyn/Invariants.lean`. That theorem covers the formal rule system. Per-result Lean checks test the actual emitted proof terms. Neither constitutes a formal verification of the Python program, its operating environment, a learning algorithm's generalization, or intelligence.

## Development experiments

```sh
python scripts/ablations.py portfolio --nodes 1000 --report .valentyn/portfolio.json
python scripts/ablations.py curriculum --seed 0 --report .valentyn/curriculum.json
python scripts/ablations.py transfer --seed 0 --report .valentyn/transfer.json
```

These are small development probes, not evidence of broad intelligence. All benchmark tasks are public and human-authored; structural deduplication prevents atom-renaming copies across splits, but does not establish contamination-free research evaluation.

Self-improvement currently changes **validated policy data**, never arbitrary executable source. Workers use a fresh Python interpreter, resource limits, timeouts, and a minimal environment. This is process isolation, not a security sandbox for untrusted code. Arbitrary source evolution is deferred until a real OS/container sandbox and an immutable evaluator are available.

Apache-2.0. See [LICENSE](LICENSE).
