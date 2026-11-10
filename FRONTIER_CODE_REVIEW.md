# Frontier implementation audit — November 10, 2026

Actual public source was downloaded to the ignored `.valentyn/frontier-sources/` directory and inspected. The [machine-readable manifest](research/frontier_manifest.json) records exact revisions, file checksums, repository push timestamps, successful reads, and failures. Repository update time is not publication or model-release time. `python scripts/fetch_frontier_sources.py` independently verifies all **28** pinned source/license files without downloading weights. No external implementation was vendored into Valentyn.

## Exact revisions and licenses

| Repository | Inspected commit | License / scope |
|---|---|---|
| openai/gpt-oss | `277020131d524f06bed5f66b694a92ed7f6e5224` | Apache-2.0; inspected LICENSE |
| deepseek-ai/DeepSeek-V3 | `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` | MIT in LICENSE-CODE; model-weight terms are a separate question |
| QwenLM/Qwen3 | `7a2f61ffc7a20d47efcd2bf97f6f2bf52729042e` | README inspected; root license endpoint returned 404; do not infer a blanket code/weight license |
| QwenLM/Qwen3.8 | `2ea10dc725823bf7c3e21ce8557cbe15245132ae` | Apache-2.0 root LICENSE inspected |
| huggingface/transformers | `07cc8e8c1b19160cfba6893ceb3099dc8f09d90c` | Apache-2.0 implementation; not a blanket license for referenced weights |
| ml-explore/mlx-lm | `56230fa02b8ff8330daf26dc95c28480423ed5fe` | MIT |
| zai-org/GLM-5 | `83c03d317378f86f0a012f7c4cd48b6243c7e69b` | Apache-2.0 root LICENSE inspected |
| MoonshotAI/Kimi-K2 | `1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4` | **Modified MIT**, including a large-commercial-service attribution condition; not ordinary MIT |
| karpathy/nanochat | `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` | MIT |

Licenses were inspected to guide reuse, not as legal advice. Valentyn's new modules are original reference implementations of mathematical mechanisms, with source attribution here. No pretrained frontier weights were loaded. Public inference code does not expose every training or proprietary post-training detail.

## Code-level findings

### gpt-oss

[Pinned model.py](https://github.com/openai/gpt-oss/blob/277020131d524f06bed5f66b694a92ed7f6e5224/gpt_oss/torch/model.py).
`RMSNorm.forward` accumulates normalization in float32. `RotaryEmbedding` includes context scaling. `AttentionBlock.forward` reshapes queries into KV-head groups; even layers use a sliding window. `sdpa` appends learned sink logits, normalizes including them, then discards sink probabilities: these are not ordinary prefix-token values. `MLPBlock.forward` selects top-k router logits, softmaxes selected logits, gathers expert matrices, and mixes their outputs. Its activation has clamping and an offset, not textbook SwiGLU. `from_checkpoint` upcasts MXFP4 through the loader before reference sharding. `TokenGenerator.generate` in **this file** recomputes the prefix; no incremental KV cache is implemented on that path. The README describes 21B/117B total parameters; neither is a local target.

### DeepSeek-V3

[Pinned inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py).
`MLA.forward` has naive and absorbed branches. The latter caches normalized latent KV plus rotary keys, contracts queries with key-expansion weights, and applies value expansion after attention. `Gate.forward` supports score transforms, group pruning, and optional correction bias for selection; mixture weights come from the original scores. `MoE.forward` combines routed experts and always-computed shared experts. `Transformer.forward` handles position offsets and returns final-position logits. FP8/custom-kernel and tensor-parallel paths assume an accelerator stack that is not an M2 implementation. Multi-token prediction and reasoning RL are not implemented by this inference file. A direct `deepseek-ai/DeepSeek-V4` repository lookup returned 404; no V4 code-level claims are made.

### Qwen3 and Qwen3.5/3.8

[Qwen3 source](https://github.com/huggingface/transformers/blob/07cc8e8c1b19160cfba6893ceb3099dc8f09d90c/src/transformers/models/qwen3/modeling_qwen3.py), [Qwen3.5 source](https://github.com/huggingface/transformers/blob/07cc8e8c1b19160cfba6893ceb3099dc8f09d90c/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
`Qwen3Attention.forward` applies per-head Q/K normalization, RoPE, cache update, and a selectable attention backend; its MLP uses gated activation. Qwen3.5 adds a sigmoid output gate to full attention. `Qwen3_5GatedDeltaNet.forward` uses depthwise causal convolution, decay/write gates, and recurrent or chunked delta updates. `torch_recurrent_gated_delta_rule` explicitly decays state, predicts stored values, corrects the residual, and reads with the query. `Qwen3_5DecoderLayer` chooses linear/full layers from configuration. The Qwen3.8 README says it builds on the Qwen3.5 foundation; this is not evidence that every 3.8 post-training detail or configuration is public or identical.

### MLX Qwen3.5

[Pinned MLX implementation](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/models/qwen3_5.py), [cache implementations](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/models/cache.py).
`GatedDeltaNet.__call__` retains convolution history and recurrent state separately; `DecoderLayer` chooses full attention at a configurable interval. This is an actual Apple-oriented implementation, unlike assuming a CUDA kernel transfers directly. Quantization, cache classes, and model sharding require separate measurement. Valentyn uses already-installed PyTorch MPS for its tiny models; no MLX speed benchmark was run, and no superiority over MLX is claimed.

### GLM-5 family

[Pinned family README](https://github.com/zai-org/GLM-5/blob/83c03d317378f86f0a012f7c4cd48b6243c7e69b/README.md), [actual framework implementation](https://github.com/huggingface/transformers/blob/07cc8e8c1b19160cfba6893ceb3099dc8f09d90c/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py).
The README separates 5.2 architecture changes from 5.3 post-training gains. `GlmMoeDsaIndexer.forward` computes causal token scores and top-k indices. `GlmMoeDsaAttention.forward` either computes or reuses those indices according to `indexer_types`, and caches compressed KV before expansion. The eager/SDPA path builds a dense additive mask; sparse semantics alone do not guarantee sparse FLOPs. MTP/speculative-decoding claims in the README are not demonstrated by this inspected forward path. The family reports 744B total/40B active parameters; sparse active compute does not make the resident weights small. Index sharing is unattractive for a two-layer, short-context pilot without an indexer-cost bottleneck.

### Kimi

[K2 README](https://github.com/MoonshotAI/Kimi-K2/blob/1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4/README.md), [Kimi Linear code](https://github.com/huggingface/transformers/blob/07cc8e8c1b19160cfba6893ceb3099dc8f09d90c/src/transformers/models/kimi_linear/modeling_kimi_linear.py), [K2.5 wrapper](https://github.com/huggingface/transformers/blob/07cc8e8c1b19160cfba6893ceb3099dc8f09d90c/src/transformers/models/kimi_k25/modeling_kimi_k25.py).
The requested K2 `inference/model.py` was unavailable. Its README reports a 1T/32B-active MLA MoE, eight routed experts, one shared expert, and MuonClip training; these are release claims, not inspected K2 training code. In the framework, `KimiLinearForgetGate` uses per-channel decay and `KimiLinearDeltaAttention` combines convolution and gated recurrence. `Kimi_K25Model` delegates text construction through `AutoModel.from_config`; the wrapper alone does not specify its entire language backbone. No undisclosed Kimi architecture is inferred.

### nanochat

[gpt.py](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py), [optimizer](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/optim.py), [training entry](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py).
`CausalSelfAttention` supports GQA, Q/K normalization, optional value embeddings, and an explicit cache path. `MLP` uses squared ReLU, not SwiGLU. `GPT.forward` includes residual scalars, token mixing, value residuals, and soft-capped logits. `init_weights` separates shape/meta construction from allocation. `MuonAdamW` uses AdamW for embeddings/scalars and orthogonalized matrix updates with variance corrections; it is more than plain SGD. The training entry accepts CPU/MPS, but some reported memory/throughput counters are CUDA-specific. Its clear interfaces inform Valentyn's checkpoints and training loop; its many optimizations are not automatically combined here.

## Mathematical mechanisms and scaling

Let width be `d`, query heads `H`, KV heads `G`, head width `a=d/H`, sequence length `S`, batch `B`, latent rank `r`, rotary width `p`, value width `v`, layers `L`, and bytes/scalar `b`.

**Attention/GQA:** `Q=XWq`, `K=XWk`, `V=XWv`; head `h` uses KV group `floor(hG/H)`. `A_h=softmax(Q_h K_groupᵀ/√a + causal_mask)`, `Y=concat(A_h V_group)Wo`. Projection parameters are approximately `2d² + 2dGa`; attention arithmetic remains quadratic in `S`. Retained KV is `2LBSGa b`. MHA is the case `G=H`.

**RoPE:** rotate coordinate pairs by a position-dependent angle. The query/key inner product then depends on relative rotations. Cache offsets must equal previous token count; matching full-prefix and incremental outputs is a required test.

**Attention sink:** a learned logit contributes `exp(s_h)` to the attention denominator without an ordinary value contribution. Windowing removes distant logits, changing what dependencies can be represented. Neither is assumed beneficial for short formal proofs without an ablation.

**Latent KV:** `c_t=Norm(Wdown x_t)`, `k_(t,h)=Uk_h c_t`, `v_(t,h)=Uv_h c_t`. The content score can be computed as `(Uk_hᵀ q_(t,h))ᵀ c_s`; attend to latent vectors first, then apply `Uv_h`. A separate rotary key supports position. Retained cache becomes `LBS(r+p)b`. Low rank reduces cache, but contractions and normalization can increase latency or constrain representation.

**Gated delta state:** `S̄_t=α_t S_(t-1)`, `δ_t=β_t(v_t-k_tᵀ S̄_t)`, `S_t=S̄_t+k_t δ_tᵀ`, `y_t=q_tᵀ S_t`. Per-head recurrent state is `a×v`; its size is independent of context. Decay can be parameterized using an exponential of a positive softplus rate. Hybrid full-attention layers still retain sequence-dependent KV. Chunked/fused training matters: a Python recurrence is not an efficiency claim.

**Experts:** `y=Σ_(e∈TopK(router(x))) w_e f_e(x) + f_shared(x)` where a shared branch may be absent. Router selection and weighting differ between implementations. Total expert parameters scale with `E`, active expert work with `k`; retaining all expert weights still costs memory. Tiny data also risks routing collapse. MoE is deferred until a dense baseline shows a capacity bottleneck.

**Normalization/FFN/optimization:** RMS normalization scales by `1/√(mean(x²)+ε)`. Standard SwiGLU is `Wdown[SiLU(Wgate x) ⊙ Wup x]`; implementations may modify it. Float32 Adam parameter/gradient/two-moment storage is roughly `16P` bytes before activations. Muon-like methods orthogonalize momentum matrices with iterative polynomial updates; optimizer choice must be compared at matched training budgets rather than assumed better.

## Measured local comparison

[Raw report](reports/neural/attention-comparison.json). Same seed 7, 256 teacher proofs, 32 validation and 32 test problems, width 64, two layers, 12 epochs, batch 16, CPU float32. Parameters and FLOPs are not exactly matched; no architecture winner is claimed. Seven timed profiling runs follow warmup, batch one.

| Variant | Parameters | Training seconds | Best validation token loss | Lean solves / 32 | Actual KV/state bytes at S=256 | S=256 prefill median |
|---|---:|---:|---:|---:|---:|---:|
| MHA | 106,688 | 4.01 | .176 | 5 | 262,144 | 2.02 ms |
| GQA | 98,496 | 3.95 | .217 | 3 | 131,072 | 2.06 ms |
| Latent | 99,056 | 4.21 | .192 | 3 | 65,536 | 2.29 ms |
| Delta/full hybrid | 103,116 | 12.97 | .164 | 5 | 69,632 | 8.29 ms |

Peak training process RSS was approximately 365–461 MB. These are measured tiny-model numbers, not estimates for billion-parameter models. Whole models are far below the 8 GB target, but physical 8 GB hardware was not tested. Tested context tops out at 256 in this pilot; a maximum useful context was not established. Incremental cached-token medians and other sequence lengths are in the raw report. Dynamic concatenation remains a cache-copy overhead; no production-kernel efficiency claim is made.

## Selection for Valentyn Core — Version 1

1. **Typed recurrent local policy:** strongest initial transfer signal; retain identical symbolic environments for learned/untrained/random controls. It is still beaten or closely matched by strong nonlearned search in important settings.
2. **Dense GQA Core:** simple small-memory token baseline with measured cache reduction; proof quality did not improve in this pilot.
3. **Compact latent Core:** useful cache-compression candidate; measured latency and solve counts do not justify default promotion.

Keep MHA as a required control. Keep the hybrid as an experimental backend; the readable recurrence currently costs more to train/prefill. Do not combine MoE, IndexShare, long windows, alternative optimizers, and quantization without individual tests. This audit supports controlled experiments, not a claim of superiority to frontier systems.
