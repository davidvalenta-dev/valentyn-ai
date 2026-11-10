"""Valentyn's from-scratch neural proof generator. Symbolic search is not used here."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import resource
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from .logic import parse
from .proof_data import ascii_formula

VOCAB = ("<pad>", "<bos>", "<proof>", "<eos>", "p", "q", "r", "s", "t", "u",
         "v", "w", "x", "y", "z", "True", "False", "->", "&", "|", "(", ")",
         "fun", "=>", "True.intro", "False.elim", "And.intro", "And.left", "And.right",
         "Or.inl", "Or.inr", "Or.elim", *(f"h{i}" for i in range(32)))
IDS = {token: i for i, token in enumerate(VOCAB)}
PROOF_IDS = [IDS[token] for token in ("<eos>", "(", ")", "fun", "=>", "True.intro",
             "False.elim", "And.intro", "And.left", "And.right", "Or.inl", "Or.inr", "Or.elim",
             *(f"h{i}" for i in range(32))) ]


def tokens(text: str) -> list[int]:
    parts = re.findall(r"->|=>|[A-Za-z][A-Za-z0-9_.]*|\S", text)
    try:
        return [IDS[part] for part in parts]
    except KeyError as error:
        raise ValueError(f"Unsupported model token: {error.args[0]}") from error


def prompt(formula: str) -> list[int]:
    return [IDS["<bos>"], *tokens(ascii_formula(parse(formula))), IDS["<proof>"]]


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "transformer"
    width: int = 96
    layers: int = 2
    heads: int = 4
    context: int = 256
    kv_heads: int = 2
    latent_rank: int = 24

    def validate(self) -> None:
        if self.kind not in ("transformer", "gru", "mha", "gqa", "mla", "hybrid"):
            raise ValueError("Unknown model architecture")
        if self.width not in (32, 64, 96, 128, 160, 192, 256) or not 1 <= self.layers <= 4:
            raise ValueError("Unsupported model dimensions")
        if self.heads not in (1, 2, 4, 8) or self.width % self.heads:
            raise ValueError("Invalid attention heads")
        if self.context not in (128, 256, 384, 512):
            raise ValueError("Context must be 128, 256, 384, or 512")
        if self.kv_heads < 1 or self.heads % self.kv_heads or not 4 <= self.latent_rank <= self.width:
            raise ValueError("Invalid grouped or latent attention dimensions")


class ProofModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(len(VOCAB), config.width, padding_idx=0)
        if config.kind == "transformer":
            self.position = nn.Embedding(config.context, config.width)
            layer = nn.TransformerEncoderLayer(config.width, config.heads,
                        dim_feedforward=4 * config.width, dropout=0.0,
                        activation="gelu", batch_first=True, norm_first=True)
            self.backbone = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        elif config.kind == "gru":
            self.backbone = nn.GRU(config.width, config.width, config.layers, batch_first=True)
        else:
            from .attention import CoreStack
            self.backbone = CoreStack(config.width, config.layers, config.heads,
                                      config.kv_heads, config.latent_rank, config.kind)
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, len(VOCAB))
        # Independent random initialization, including each Transformer layer.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length > self.config.context:
            raise ValueError("Model context exceeded")
        hidden = self.embedding(ids)
        if self.config.kind == "transformer":
            hidden = hidden + self.position(torch.arange(length, device=ids.device))
            causal = torch.ones(length, length, device=ids.device, dtype=torch.bool).triu(1)
            hidden = self.backbone(hidden, mask=causal, is_causal=True)
        elif self.config.kind == "gru":
            hidden, _ = self.backbone(hidden)
        else:
            hidden = self.backbone(hidden)
        return self.output(self.norm(hidden))

    def forward_cached(self, ids, cache=None):
        from .attention import CORE_KINDS
        if self.config.kind not in CORE_KINDS:
            raise ValueError("Incremental cache is supported by Core attention variants only")
        previous = 0 if cache is None else cache["length"]
        if previous + ids.shape[1] > self.config.context:
            raise ValueError("Cached context exceeded")
        hidden, states = self.backbone.cached(self.embedding(ids), None if cache is None else cache["states"])
        return self.output(self.norm(hidden)), {"length": previous + ids.shape[1], "states": states}

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def select_device(request: str) -> torch.device:
    from .hardware import hardware_snapshot
    hardware_snapshot()  # Detect capacity and disk before model allocation/loading.
    if request not in ("cpu", "mps", "auto"):
        raise ValueError("Device must be cpu, mps, or auto")
    if request == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple Metal backend unavailable")
    return torch.device("mps" if request == "mps" or
                        request == "auto" and torch.backends.mps.is_available() else "cpu")


def memory_measurement(device: torch.device) -> dict:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"peak_process_rss_bytes": int(raw if sys.platform == "darwin" else raw * 1024)}
    if device.type == "mps":
        torch.mps.synchronize()
        result["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    return result


def save_model(model: ProofModel, directory: Path, metadata: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "weights.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, weights)
    manifest = {"schema": 1, "name": "Valentyn", "config": asdict(model.config),
                "vocabulary": list(VOCAB), "parameters": model.parameter_count,
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "metadata": metadata}
    (directory / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_model(directory: Path, device: torch.device) -> tuple[ProofModel, dict]:
    manifest = json.loads((directory / "model.json").read_text())
    if manifest["schema"] != 1 or manifest["vocabulary"] != list(VOCAB):
        raise ValueError("Incompatible model manifest")
    weights = directory / "weights.pt"
    if weights.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Model checkpoint exceeds 32 MiB local limit")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    model = ProofModel(ModelConfig(**manifest["config"]))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    return model, manifest


def encode_example(row: dict, context: int) -> tuple[list[int], list[int]]:
    prefix = prompt(row["formula"])
    sequence = [*prefix, *tokens(row["proof"]), IDS["<eos>"]]
    if len(sequence) - 1 > context:
        raise ValueError("Training example exceeds model context")
    targets = sequence[1:]
    targets[:len(prefix) - 1] = [-100] * (len(prefix) - 1)
    return sequence[:-1], targets


def batch(examples: list[tuple[list[int], list[int]]], device: torch.device):
    length = max(len(ids) for ids, _ in examples)
    ids = torch.zeros(len(examples), length, dtype=torch.long)
    targets = torch.full((len(examples), length), -100, dtype=torch.long)
    for index, (source, target) in enumerate(examples):
        ids[index, :len(source)] = torch.tensor(source)
        targets[index, :len(target)] = torch.tensor(target)
    return ids.to(device), targets.to(device)


def train_model(data: dict, output: Path, *, config: ModelConfig = ModelConfig(),
                epochs: int = 30, batch_size: int = 32, learning_rate: float = 0.001,
                seed: int = 7, device_name: str = "cpu", initial: Path | None = None,
                progress=None) -> dict:
    if not 1 <= epochs <= 1000 or not 1 <= batch_size <= 128 or not 0 < learning_rate <= 0.1:
        raise ValueError("Training configuration outside resource limits")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = select_device(device_name)
    start = time.monotonic()
    if initial:
        model, parent = load_model(initial, device)
        config = model.config
        initialization = {"parent_weights": parent["weights_sha256"]}
    else:
        model = ProofModel(config).to(device)
        initialization = {"random_seed": seed, "pretrained_weights": False}
    training = [encode_example(row, config.context) for row in data["train"]]
    validation = [encode_example(row, config.context) for row in data["validation"]]
    training_tokens = sum(sum(t != -100 for t in target) for _, target in training)
    common = {"dataset_digest": data["digest"], "initialization": initialization,
              "torch_version": str(torch.__version__), "device": str(device)}
    save_model(model, output / "initial", {**common, "stage": "before_training"})
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    from .resources import ResourceBudget
    budget = ResourceBudget()
    resource_plan = budget.preflight(model.parameter_count, batch_size, config.context,
                                     config.width, config.layers, config.heads)
    history = []
    best_validation = math.inf
    best_epoch = 0

    def loss_on(examples) -> float:
        model.eval()
        total_loss = total_tokens = 0
        with torch.inference_mode():
            for index in range(0, len(examples), batch_size):
                ids, targets = batch(examples[index:index + batch_size], device)
                logits = model(ids)
                total_loss += F.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1),
                                             ignore_index=-100, reduction="sum").item()
                total_tokens += int((targets != -100).sum().item())
        return total_loss / total_tokens

    initial_loss = loss_on(validation)
    for epoch in range(epochs):
        model.train()
        order = rng.sample(range(len(training)), len(training))
        running = count = 0
        for index in range(0, len(order), batch_size):
            ids, targets = batch([training[i] for i in order[index:index + batch_size]], device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; refusing checkpoint")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            budget.guard(device)
            running += loss.item()
            count += 1
        validation_loss = loss_on(validation)
        entry = {"epoch": epoch + 1, "training_loss": running / count,
                 "validation_loss": validation_loss, "elapsed_seconds": time.monotonic() - start}
        history.append(entry)
        if validation_loss < best_validation:
            best_validation, best_epoch = validation_loss, epoch + 1
            save_model(model, output, {**common, "stage": "trained", "best_epoch": best_epoch})
        if progress:
            progress(entry)
    report = {"kind": "neural_training", **common, "architecture": asdict(config),
              "parameters": model.parameter_count, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "best_epoch": best_epoch,
              "examples": len(training), "unique_supervised_tokens": training_tokens,
              "supervised_tokens_seen": training_tokens * epochs,
              "initial_validation_loss": initial_loss, "best_validation_loss": best_validation,
              "training_seconds": time.monotonic() - start,
              "memory": memory_measurement(device), "history": history,
              "resource_plan": resource_plan,
              "selection": "Lowest validation token loss; test set never used for selection.",
              "scope": "Autoregressive proof generation, learned from random weights unless parent explicitly provided."}
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@torch.inference_mode()
def generate(model: ProofModel, formula: str, *, beam: int = 1, max_tokens: int = 100,
             timeout: float = 15.0, use_cache: bool = True) -> dict:
    if not 1 <= beam <= 16 or not 1 <= max_tokens <= 256 or not 0 < timeout <= 300:
        raise ValueError("Generation budget outside limits")
    model.eval()
    device = next(model.parameters()).device
    prefix = prompt(formula)
    if len(prefix) >= model.config.context:
        raise ValueError("Theorem exceeds model context")
    from .attention import CORE_KINDS, cache_bytes
    if use_cache and beam == 1 and model.config.kind in CORE_KINDS:
        start = time.monotonic()
        logits, cache = model.forward_cached(torch.tensor([prefix], device=device))
        calls = 1
        produced = []
        completed = False
        for index in range(min(max_tokens, model.config.context - len(prefix))):
            if time.monotonic() - start >= timeout:
                break
            allowed = torch.tensor(PROOF_IDS, device=device)
            token = PROOF_IDS[int(logits[0, -1, allowed].argmax().item())]
            produced.append(token)
            if token == IDS["<eos>"]:
                completed = True
                break
            if index + 1 < min(max_tokens, model.config.context - len(prefix)):
                logits, cache = model.forward_cached(torch.tensor([[token]], device=device), cache)
                calls += 1
        text = " ".join(VOCAB[i] for i in produced[:-1]) if completed else ""
        return {"proofs": [text] if text else [], "inference_calls": calls,
                "expanded_tokens": len(produced), "elapsed_seconds": time.monotonic() - start,
                "completed": int(completed), "timed_out": time.monotonic() - start >= timeout,
                "cache_used": True, "retained_cache_bytes": cache_bytes(cache)}
    active = [(prefix, 0.0)]
    finished = []
    calls = generated = 0
    start = time.monotonic()
    # Global beam search by cumulative token log probability; no theorem-specific
    # symbolic rules, reference proofs, or type-directed search are consulted.
    for _ in range(min(max_tokens, model.config.context - len(prefix))):
        if not active or time.monotonic() - start >= timeout:
            break
        ids = torch.tensor([sequence for sequence, _ in active], dtype=torch.long, device=device)
        logits = model(ids)[:, -1, :]
        calls += 1
        mask = torch.full_like(logits, float("-inf"))
        mask[:, PROOF_IDS] = logits[:, PROOF_IDS]
        probabilities = F.log_softmax(mask, dim=-1)
        choices = []
        for index, (sequence, score) in enumerate(active):
            values, indices = probabilities[index].topk(beam)
            for logp, token in zip(values.tolist(), indices.tolist()):
                choices.append((sequence + [token], score + logp))
        choices.sort(key=lambda item: item[1], reverse=True)
        active = []
        for sequence, score in choices[:beam]:
            generated += 1
            if sequence[-1] == IDS["<eos>"]:
                finished.append((sequence, score))
            else:
                active.append((sequence, score))
    finished.sort(key=lambda item: item[1], reverse=True)
    proofs = []
    for sequence, score in finished:
        text = " ".join(VOCAB[i] for i in sequence[len(prefix):-1])
        if text and text not in proofs:
            proofs.append(text)
    return {"proofs": proofs[:beam], "inference_calls": calls, "expanded_tokens": generated,
            "elapsed_seconds": time.monotonic() - start, "completed": len(finished),
            "timed_out": time.monotonic() - start >= timeout}
