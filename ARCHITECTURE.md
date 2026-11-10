# Valentyn architecture

Valentyn contains trainable proof generators initialized from random weights, a typed proof-construction environment, and an independent Lean 4.24 verifier. The current domain is constructive propositional logic.

The 72,005-parameter GRU policy reads a local goal/context and scores legal inference actions. Bounded best-first search composes actions into a proof term. It does not call the symbolic teacher during inference. The environment supplies inference rules and conjunction projections, so random-policy controls are essential. Whole-proof Transformer and GRU models provide alternative learned baselines.

Training references come from the symbolic teacher and are checked by Lean. Model selection uses validation data; the frozen transfer suite was generated after checkpoint hashes were recorded. Accepted results require compilation and zero axiom dependencies. This verifies individual terms, not Python correctness or intelligence.

See [implementation and objectives](docs/VALENTYN_1.md), [source audit](FRONTIER_CODE_REVIEW.md), and [trust boundaries](docs/ARCHITECTURE.md). Optional PyTorch models coexist with the dependency-free symbolic system. No pretrained model weights or paid API is required.
