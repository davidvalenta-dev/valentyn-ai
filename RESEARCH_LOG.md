# Research log — 2026-11-10

Implemented optional PyTorch models trained from scratch; verified 1,180 teacher references with Lean; trained whole-term Transformer and GRU baselines. Their family-transfer failures motivated local typed proof-action learning. Trained a 72,005-parameter GRU policy without pretrained weights.

Recorded initial tests, validation-only beam promotion, frozen transfer, untrained/retrieval/random controls, including five random greedy seeds. The random beam control nearly matched learned search, limiting the central improvement claim.

Inspected pinned frontier source and license files rather than inferring implementation from model names. Implemented small MHA, GQA, absorbed MLA and delta/GQA hybrid cores; ran causal/cache/gradient tests and sequential matched-update pilots. No giant checkpoints were downloaded. Exact source hashes and fetch script are under `research/` and `scripts/`.

Added strict data-only configs, hardware reporting, conservative training resource checks and model CLI aliases. Resource checks are estimates and counter limits, not an OS memory sandbox. Experiments ran on a 16 GB M2; the 8 GB target remains unvalidated on physical 8 GB hardware.

Final local validation: 49 tests passed with PyTorch; the dependency-free environment passed the same suite with 15 optional model tests skipped. Lean audited nine project theorems with zero axiom dependencies. An independent archived-output pass recompiled 518 distinct proof terms representing 1,387 accepted report occurrences, with zero failures; see `reports/neural/proof-recheck.json`. These counts exclude teacher references without the report-attempt schema, which were checked during dataset generation.
