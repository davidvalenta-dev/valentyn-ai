# Recorded experiments

Models were trained from random initialization on 1,000 Lean-checked synthetic training examples, with 80 validation and 100 initial test examples. Two template families were excluded from fitting. The small dataset is not a benchmark of research mathematics.

| Initial test, 100 problems | Lean accepted |
|---|---:|
| Whole-proof Transformer, greedy |49|
| Whole-proof GRU, greedy |66|
| Typed state policy, greedy |75|
| Typed state policy, beam 4 |97|

On a fresh frozen 128-problem transfer suite, the learned state policy solved 52 greedily and 91 with beam 4. The untrained policy solved 34 greedily. Five random greedy controls solved 28, 30, 35, 31, 32; the random beam-4 control solved 90. Transformer solved 2, GRU and retrieval solved 0. These controls show substantial contribution from the typed environment. A one-problem beam-search difference does not establish a robust learned advantage. Neural inference was slower than random action scoring.

The follow-up action model adds intermediate forward derivations and is trained on bounded alternative-action costs. On a new frozen 192-problem suite excluding both earlier suites, its three seeds solved 174/188/166 greedily and 188/190/188 with beam 4. The selected seed-11 model reached 190/192 with beam 4. Matching controls reached 149/192 for the old symbolic prover at 200 nodes, 148/192 for the previous learned policy, 137/192 for a structural heuristic, and 143/192 for the best closed-first random seed. A 5000-node symbolic run solved 192/192, so this is a strong bounded-search improvement rather than a replacement for exhaustive symbolic search.

Four attention implementations were trained sequentially on a smaller matched-update pilot. MHA/GQA/MLA/hybrid solved 5/3/3/5 of 32 problems. Their retained two-layer caches at length 256 were 262144/131072/65536/69632 bytes. Parameter counts differ; these are not equal-FLOP comparisons. Hybrid training was slower. One seed and short training do not support frontier-performance claims.

Raw training logs, attempts, rejected terms, timings, and controls: [reports/neural](reports/neural). Architecture details and exact source revisions: [code review](FRONTIER_CODE_REVIEW.md). Historical symbolic results: [initial experiments](docs/RESULTS.md). No demonstrated new mathematical theorem or superintelligence.
