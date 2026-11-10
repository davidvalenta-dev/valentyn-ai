# Recorded experiments

Models were trained from random initialization on 1,000 Lean-checked synthetic training examples, with 80 validation and 100 initial test examples. Two template families were excluded from fitting. The small dataset is not a benchmark of research mathematics.

| Initial test, 100 problems | Lean accepted |
|---|---:|
| Whole-proof Transformer, greedy |49|
| Whole-proof GRU, greedy |66|
| Typed state policy, greedy |75|
| Typed state policy, beam 4 |97|

On a fresh frozen 128-problem transfer suite, the learned state policy solved 52 greedily and 91 with beam 4. The untrained policy solved 34 greedily. Five random greedy controls solved 28, 30, 35, 31, 32; the random beam-4 control solved 90. Transformer solved 2, GRU and retrieval solved 0. These controls show substantial contribution from the typed environment. A one-problem beam-search difference does not establish a robust learned advantage. Neural inference was slower than random action scoring.

Four attention implementations were trained sequentially on a smaller matched-update pilot. MHA/GQA/MLA/hybrid solved 5/3/3/5 of 32 problems. Their retained two-layer caches at length 256 were 262144/131072/65536/69632 bytes. Parameter counts differ; these are not equal-FLOP comparisons. Hybrid training was slower. One seed and short training do not support frontier-performance claims.

Raw training logs, attempts, rejected terms, timings, and controls: [reports/neural](reports/neural). Architecture details and exact source revisions: [code review](FRONTIER_CODE_REVIEW.md). Historical symbolic results: [initial experiments](docs/RESULTS.md). No demonstrated new mathematical theorem or superintelligence.
