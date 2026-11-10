# Bundled research checkpoints

All weights were trained from random initialization on the committed synthetic corpus. No pretrained weights are included. Each model directory contains architecture metadata, SHA-256 weight integrity metadata, weights-only state dictionaries, and training logs. `initial/` holds the corresponding untrained control where available. Load only via the bounded project loader (`weights_only=True`).

`valentyn-state` is the default 72,005-parameter typed action policy. `valentyn-transformer` and `valentyn-gru` generate full proof terms. `core-attention/` contains short CPU pilot runs, not recommended production models. See root EXPERIMENTS.md for controls and limitations. These checkpoints do not solve research mathematics.
