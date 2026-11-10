"""Original small reference attention variants, informed by the source-code audit.

These implement mathematical mechanisms, not pretrained frontier checkpoints.
All paths support float32 CPU/MPS; no CUDA or third-party fused kernels required.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

CORE_KINDS = ("mha", "gqa", "mla", "hybrid")


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                * self.weight).to(x.dtype)


def rotary(x, offset=0):
    dimension = x.shape[-1]
    if dimension % 2:
        raise ValueError("Rotary dimension must be even")
    positions = torch.arange(offset, offset + x.shape[1], device=x.device).float()
    frequency = 10000.0 ** (-torch.arange(0, dimension, 2, device=x.device).float() / dimension)
    angle = (positions[:, None] * frequency[None, :])[None, :, None, :]
    cosine, sine = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1).flatten(-2)


def causal_probabilities(scores, offset):
    query = torch.arange(offset, offset + scores.shape[-2], device=scores.device)
    key = torch.arange(scores.shape[-1], device=scores.device)
    return scores.masked_fill(key[None, :] > query[:, None], float("-inf")).softmax(-1)


class GroupedAttention(nn.Module):
    def __init__(self, width, heads, kv_heads):
        super().__init__()
        if heads % kv_heads or width % heads:
            raise ValueError("Head dimensions must divide evenly")
        self.heads, self.kv_heads, self.dimension = heads, kv_heads, width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, kv_heads * self.dimension, bias=False)
        self.value = nn.Linear(width, kv_heads * self.dimension, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x, cache=None):
        batch, length, _ = x.shape
        offset = 0 if cache is None else cache[0].shape[1]
        q = rotary(self.query(x).view(batch, length, self.heads, self.dimension), offset)
        k = rotary(self.key(x).view(batch, length, self.kv_heads, self.dimension), offset)
        v = self.value(x).view(batch, length, self.kv_heads, self.dimension)
        if cache is not None:
            k, v = torch.cat((cache[0], k), 1), torch.cat((cache[1], v), 1)
        saved = (k, v)  # Cache retains only KV heads, not their query-head repetitions.
        groups = self.heads // self.kv_heads
        keys, values = k.repeat_interleave(groups, 2), v.repeat_interleave(groups, 2)
        scores = torch.einsum("bthd,bshd->bhts", q, keys) / math.sqrt(self.dimension)
        probabilities = causal_probabilities(scores, offset)
        output = torch.einsum("bhts,bshd->bthd", probabilities, values).reshape(batch, length, -1)
        return self.output(output), saved


class LatentAttention(nn.Module):
    """Compressed latent KV with absorbed content projections and decoupled RoPE."""
    def __init__(self, width, heads, rank):
        super().__init__()
        self.heads, self.dimension, self.rank = heads, width // heads, rank
        self.rope_dimension = min(8, self.dimension // 2 // 2 * 2)
        self.content_dimension = self.dimension - self.rope_dimension
        self.query = nn.Linear(width, width, bias=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.latent_norm = RMSNorm(rank)
        self.key_rope = nn.Linear(width, self.rope_dimension, bias=False)
        self.key_up = nn.Parameter(torch.empty(heads, self.content_dimension, rank))
        self.value_up = nn.Parameter(torch.empty(heads, self.dimension, rank))
        nn.init.xavier_uniform_(self.key_up)
        nn.init.xavier_uniform_(self.value_up)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x, cache=None):
        batch, length, _ = x.shape
        offset = 0 if cache is None else cache[0].shape[1]
        q = self.query(x).view(batch, length, self.heads, self.dimension)
        content, position = q.split((self.content_dimension, self.rope_dimension), dim=-1)
        position = rotary(position, offset)
        latent = self.latent_norm(self.down(x))
        key_position = rotary(self.key_rope(x).unsqueeze(2), offset).squeeze(2)
        if cache is not None:
            latent = torch.cat((cache[0], latent), dim=1)
            key_position = torch.cat((cache[1], key_position), dim=1)
        absorbed_query = torch.einsum("bthd,hdr->bhtr", content, self.key_up)
        scores = (torch.einsum("bhtr,bsr->bhts", absorbed_query, latent) +
                  torch.einsum("bthd,bsd->bhts", position, key_position)) / math.sqrt(self.dimension)
        probabilities = causal_probabilities(scores, offset)
        attended_latent = torch.einsum("bhts,bsr->bhtr", probabilities, latent)
        output = torch.einsum("bhtr,hdr->bthd", attended_latent, self.value_up)
        return self.output(output.reshape(batch, length, -1)), (latent, key_position)


class GatedDeltaAttention(nn.Module):
    """Readable recurrent delta-rule reference, not Qwen's fused implementation.

    Omits the causal convolution and output gate to isolate the state update.
    The Python recurrence is expected to be slow for long training sequences.
    """
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.dimension = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.gates = nn.Linear(width, 2 * heads)
        self.log_rate = nn.Parameter(torch.zeros(heads))
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x, cache=None):
        batch, length, width = x.shape
        q, k, v = (part.view(batch, length, self.heads, self.dimension)
                   for part in self.qkv(x).chunk(3, dim=-1))
        q, k = F.normalize(q, dim=-1) / math.sqrt(self.dimension), F.normalize(k, dim=-1)
        a, b = self.gates(x).chunk(2, dim=-1)
        decay = torch.exp(-self.log_rate.exp() * F.softplus(a))
        beta = b.sigmoid()
        state = (x.new_zeros(batch, self.heads, self.dimension, self.dimension)
                 if cache is None else cache)
        outputs = []
        for index in range(length):
            state = decay[:, index, :, None, None] * state
            predicted = torch.einsum("bhd,bhdv->bhv", k[:, index], state)
            correction = beta[:, index, :, None] * (v[:, index] - predicted)
            state = state + k[:, index, :, :, None] * correction[:, :, None, :]
            outputs.append(torch.einsum("bhd,bhdv->bhv", q[:, index], state))
        output = torch.stack(outputs, dim=1).reshape(batch, length, width)
        return self.output(output), state


class CoreBlock(nn.Module):
    def __init__(self, width, heads, kv_heads, rank, kind):
        super().__init__()
        self.norm_attention, self.norm_ff = RMSNorm(width), RMSNorm(width)
        if kind == "mla":
            self.attention = LatentAttention(width, heads, rank)
        elif kind == "delta":
            self.attention = GatedDeltaAttention(width, heads)
        else:
            self.attention = GroupedAttention(width, heads, heads if kind == "mha" else kv_heads)
        # 3*d*(8*d/3) roughly matches a 4*d two-matrix GELU FFN budget.
        intermediate = 8 * width // 3
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, x, cache=None):
        attended, cache = self.attention(self.norm_attention(x), cache)
        x = x + attended
        normalized = self.norm_ff(x)
        x = x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        return x, cache


class CoreStack(nn.Module):
    def __init__(self, width, layers, heads, kv_heads, rank, kind):
        super().__init__()
        self.layers = nn.ModuleList(CoreBlock(width, heads, kv_heads, rank,
                    "delta" if kind == "hybrid" and index % 2 == 0 else
                    "gqa" if kind == "hybrid" else kind) for index in range(layers))

    def forward(self, x):
        output, _ = self.cached(x)
        return output

    def cached(self, x, caches=None):
        saved = []
        for index, layer in enumerate(self.layers):
            x, state = layer(x, None if caches is None else caches[index])
            saved.append(state)
        return x, saved


def cache_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(cache_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(cache_bytes(item) for item in value.values())
    return 0
