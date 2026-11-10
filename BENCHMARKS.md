# Benchmark protocol

`benchmarks/neural/dataset.json` contains 1,000 training, 80 validation and 100 test examples with canonical-shape deduplication across splits. Training references were independently Lean checked. `fresh-transfer.json` preserves fitting splits and replaces test with 128 fresh problems generated after checkpoint hashes were frozen. Four synthetic families probe longer chains and compositional cases. These families remain related to the teacher's domain; teacher solvability is a selection property.

Use all failures in denominators. Compare learned, untrained, retrieval, uniform-random and symbolic policies under explicit node/beam limits. Report verification separately from inference latency. Never promote using test results. The recorded beam-promotion trial used validation only and promoted neither candidate.

```sh
valentyn evaluate --held-out --checkpoint models/valentyn-state
valentyn model-evaluate --help
valentyn compare --help
python scripts/recheck_reports.py
```

Run full fresh-suite evaluation sparingly: hundreds of independent Lean invocations take time. Weights and datasets are committed; runtime archives live under ignored `.valentyn/`. No public research-mathematics benchmark has been passed.
