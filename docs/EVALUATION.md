# Reproduction and evaluation protocol

## Environment

Python 3.11+, Lean `leanprover/lean4:v4.24.0`, standard library only at Python runtime. A virtual environment is required for isolated workers to import the installed package. Setup uses setuptools as a build dependency. There is no Mathlib checkout, GPU runtime, pretrained model, or API service.

```sh
sh scripts/setup.sh
. .venv/bin/activate
sh scripts/check.sh
valentyn --state .valentyn/reproduction train --seed 0 --report .valentyn/reproduction/train.json
valentyn --state .valentyn/reproduction evolve --seed 0 --generations 12 --report .valentyn/reproduction/evolution.json
valentyn --state .valentyn/reproduction benchmark --report .valentyn/reproduction/test.json
valentyn --state .valentyn/reproduction discover --seed 17 --count 40 --report .valentyn/reproduction/discovery.json
```

The benchmark defaults to the trained policy. To evaluate an evolved policy, supply `--policy .valentyn/reproduction/evolved-policy.json`. Zero promotions means those policies are identical. SQLite history and local novelty counts depend on prior state, so discovery archive counts must be compared using a fresh state directory and the same preceding command sequence.

## Split discipline

The bundled corpus has 12 training, six validation, and 14 test formulas. Training measures all three strategies on training only. Evolution selects using validation only. A canonical syntax representation renames atoms by first occurrence and rejects duplicate structures anywhere in the corpus. It does not identify logically equivalent formulas or prevent human familiarity with these elementary results.

The test suite is held out from weight fitting and automatic selection, not from implementation development. A compiler-linter issue was found while developing the benchmark and fixed before the recorded final run. Therefore these results are explicitly development evidence. A credible research result needs a newly frozen, independently sourced or generated test suite after development decisions are complete.

Always compare against focused, forward, and split fixed strategies. A learned policy matching a good constant policy is not evidence of useful contextual learning. Report solving rates, per-problem regressions, aggregate node counts, rejected proofs, and exhausted searches. Counterexamples and incomplete search outcomes must remain separate. Report training and selection costs outside the equal-budget evaluation comparison.

All arms receive identical per-attempt node/depth/search-time caps. A Lean check has a separate timeout and heartbeat cap. Elapsed wall time varies with OS load; deterministic node counts are the primary local cost proxy. They count recursive calls and saturation work, not all CPU instructions or compiler work. Claims about real efficiency require end-to-end wall time and memory measurements as well.

## Evidence artifacts

Reports contain formulas, returned terms, source hashes, complete generated Lean source, axiom-check output, policy/corpus digests, and Python/platform/toolchain provenance. SQLite stores accepted proofs separately from all attempts. The repository's `reports/` contains recorded runs, not hand-written success examples.

Run `python scripts/recheck_reports.py --export-dir .valentyn/verified-proofs` to reconstruct and recheck all distinct saved proof artifacts, and export standalone `.lean` files. Hash mismatches or failed proof checks stop the audit.

`scripts/audit_lean.py` imports the compiled project, prints dependencies for every named project theorem, and requires zero axioms. Grep is only an additional check. The small proof modules intentionally avoid even ordinary foundational axioms; this does not imply that using Lean's standard classical axioms in future mathematics would make a theorem invalid. Those dependencies would need an explicit policy and transparent reporting.

## Research-grade follow-up

Use at least five independent training seeds, size/family-held-out generated suites, paired confidence intervals, total compute accounting, and a preregistered stopping rule. Freeze the evaluator and test corpus before a search over architectures. For novelty, retain literature searches and mathematical review in addition to a machine-checked proof. A valid proof of the wrong formal statement is not a solution to the original problem.
