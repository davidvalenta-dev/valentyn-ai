"""A from-scratch neural policy constructs proof trees from typed local states."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from .logic import Formula, parse
from .neural import IDS, VOCAB, memory_measurement, select_device, tokens
from .proof_data import ascii_formula
from .proof_states import ACTIONS, Term, action_label, available, extract_steps, premises, with_truth


def state_tokens(goal: Formula, context, max_length: int = 512) -> list[int]:
    sequence = [IDS["<bos>"]]
    for index, (formula, _) in enumerate(with_truth(context)):
        sequence.extend([IDS[f"h{index}"], *tokens(ascii_formula(formula)), IDS["<proof>"]])
    sequence.extend([*tokens(ascii_formula(goal)), IDS["<eos>"]])
    if len(sequence) > max_length:
        raise ValueError("Factorized state exceeds context budget")
    return sequence


class StatePolicy(nn.Module):
    def __init__(self, width: int = 96):
        super().__init__()
        if width not in (32, 64, 96, 128, 192):
            raise ValueError("Unsupported state policy width")
        self.width = width
        self.embedding = nn.Embedding(len(VOCAB), width, padding_idx=0)
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(ACTIONS)))

    def forward(self, ids, lengths):
        hidden, _ = self.recurrent(self.embedding(ids))
        last = hidden[torch.arange(ids.shape[0], device=ids.device), lengths - 1]
        return self.output(last)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def state_batch(rows, device):
    lengths = torch.tensor([len(row[0]) for row in rows], device=device)
    ids = torch.zeros(len(rows), int(lengths.max().item()), dtype=torch.long)
    for index, (sequence, _) in enumerate(rows):
        ids[index, :len(sequence)] = torch.tensor(sequence)
    return ids.to(device), lengths, torch.tensor([label for _, label in rows], device=device)


def save_policy(model, path: Path, metadata):
    path.mkdir(parents=True, exist_ok=True)
    weights = path / "weights.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights)
    manifest = {"schema": 1, "name": "Valentyn", "architecture": "factorized-gru-policy",
                "width": model.width, "parameters": model.parameter_count,
                "actions": list(ACTIONS), "vocabulary": list(VOCAB),
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "metadata": metadata}
    (path / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_policy(path: Path, device):
    manifest = json.loads((path / "model.json").read_text())
    if manifest.get("architecture") == "relational-action-policy":
        from .relational import load
        return load(path, device)
    if manifest["schema"] != 1 or manifest["actions"] != list(ACTIONS) or manifest["vocabulary"] != list(VOCAB):
        raise ValueError("Incompatible factorized policy")
    weights = path / "weights.pt"
    if weights.stat().st_size > 8 * 1024 * 1024 or hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Invalid factorized checkpoint")
    model = StatePolicy(manifest["width"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval(), manifest


def train_policy(data: dict, output: Path, *, epochs=15, seed=7, device_name="cpu",
                 width=96, batch_size=64, learning_rate=0.002, progress=None):
    if not 1 <= epochs <= 300:
        raise ValueError("Epochs must be 1..300")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = select_device(device_name)
    rng = random.Random(seed)
    start = time.monotonic()
    if not 1 <= batch_size <= 128 or not 0 < learning_rate <= 0.1:
        raise ValueError("Invalid training batch size or learning rate")
    model = StatePolicy(width).to(device)
    rows = {}
    skipped = {}
    for split in ("train", "validation"):
        rows[split] = []
        skipped[split] = []
        for task in data[split]:
            try:
                steps = extract_steps(parse(task["formula"]), task["proof"])
                encoded = [(state_tokens(step["goal"], step["context"]), step["action"]) for step in steps]
            except ValueError as error:
                skipped[split].append({"task_id": task["id"], "reason": str(error)})
                continue
            rows[split].extend(encoded)
    if not rows["train"] or not rows["validation"]:
        raise ValueError("No usable proof-state examples")
    metadata = {"dataset_digest": data["digest"], "random_seed": seed, "pretrained_weights": False}
    save_policy(model, output / "initial", metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    from .resources import ResourceBudget
    budget = ResourceBudget()
    resource_plan = budget.preflight(model.parameter_count, batch_size, 512, model.width, 1, 1)
    history = []
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        training = rng.sample(rows["train"], len(rows["train"]))
        for index in range(0, len(training), batch_size):
            ids, lengths, labels = state_batch(training[index:index + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            budget.guard(device)
        model.eval()
        total = correct = count = 0
        with torch.inference_mode():
            for index in range(0, len(rows["validation"]), batch_size):
                ids, lengths, labels = state_batch(rows["validation"][index:index + batch_size], device)
                logits = model(ids, lengths)
                total += F.cross_entropy(logits, labels, reduction="sum").item()
                correct += int((logits.argmax(-1) == labels).sum().item())
                count += labels.numel()
        entry = {"epoch": epoch + 1, "validation_loss": total / count,
                 "validation_action_accuracy": correct / count, "elapsed_seconds": time.monotonic() - start}
        history.append(entry)
        if entry["validation_loss"] < best:
            best = entry["validation_loss"]
            save_policy(model, output, {**metadata, "best_epoch": epoch + 1})
        if progress:
            progress(entry)
    report = {"kind": "factorized_training", **metadata, "parameters": model.parameter_count,
              "architecture": "GRU policy over typed local proof states", "epochs": epochs,
              "batch_size": batch_size, "learning_rate": learning_rate,
              "training_states": len(rows["train"]), "validation_states": len(rows["validation"]),
              "training_state_tokens": sum(len(row[0]) for row in rows["train"]),
              "skipped_theorems": skipped, "history": history,
              "training_seconds": time.monotonic() - start, "memory": memory_measurement(device),
              "device": str(device), "torch_version": str(torch.__version__)}
    report["resource_plan"] = resource_plan
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@dataclass(frozen=True)
class Goal:
    identifier: int
    formula: Formula
    context: tuple
    ancestry: frozenset = frozenset()


def expand(goal: Goal, action: int, serial: int):
    target, context = goal.formula, with_truth(goal.context)
    label = action_label(action)
    children = []

    def hole(formula, facts=context):
        nonlocal serial
        serial += 1
        children.append(Goal(serial, formula, tuple(facts), goal.ancestry |
                             {(target, frozenset(f for f, _ in context))}))
        return f"X{serial}"

    if label == "intro":
        variable = f"h{3 * serial}"
        expression = f"(fun {variable} => {hole(target.args[1], (*context, (target.args[0], Term(variable))))})"
    elif label == "pair":
        expression = f"(And.intro {hole(target.args[0])} {hole(target.args[1])})"
    elif label in ("left", "right"):
        index = 0 if label == "left" else 1
        expression = f"({'Or.inl' if index == 0 else 'Or.inr'} {hole(target.args[index])})"
    elif label == "absurd":
        false = next((t.lean() for f, t in context if f.op == "False"), None)
        expression = f"(False.elim {false or hole(Formula('False'))})"
    else:
        operation, index = label.split(":")
        formula, term = context[int(index)]
        if operation == "derive":
            premise = next(t for f, t in context if f == formula.args[0])
            evidence = Term("call", (term, premise))
            expression = hole(target, (*context, (formula.args[1], evidence)))
        elif operation == "exact":
            expression = term.lean()
        elif operation == "apply":
            needed = premises(formula, target)
            expression = "(" + " ".join([term.lean(), *(hole(g) for g in needed)]) + ")"
        else:
            remainder = tuple((f, t) for f, t in context if f != formula)
            left_variable, right_variable = f"h{3 * serial + 1}", f"h{3 * serial + 2}"
            left = hole(target, (*remainder, (formula.args[0], Term(left_variable))))
            right = hole(target, (*remainder, (formula.args[1], Term(right_variable))))
            expression = (f"(Or.elim {term.lean()} (fun {left_variable} => {left}) "
                          f"(fun {right_variable} => {right}))")
    return expression, tuple(children), serial


@torch.inference_mode()
def generate_proof(model, formula: str, *, max_nodes=200, beam=4, timeout=10.0, random_seed=None, prefer_closing=False):
    if not 1 <= max_nodes <= 10000 or not 1 <= beam <= 16 or not 0 < timeout <= 300:
        raise ValueError("Invalid factorized search budget")
    device = next(model.parameters()).device
    model.eval()
    rng = random.Random(random_seed)
    start = time.monotonic()
    # Heap entries: negative model log probability, unique tie-breaker, open goals,
    # assignments, next identifier. Every expansion is chosen by a policy score.
    frontier = [(0.0, 0, (Goal(0, parse(formula), ()),), {}, 0)]
    counter = nodes = calls = 0
    proofs = []
    while frontier and nodes < max_nodes and time.monotonic() - start < timeout:
        cost, _, goals, assignments, serial = heapq.heappop(frontier)
        if not goals:
            def render(identifier):
                return re.sub(r"\bX([0-9]+)\b", lambda m: render(int(m[1])), assignments[identifier])
            proof = render(0)
            if len(proof) <= 100000:
                proofs.append(proof)
            break
        current, *rest = goals
        context = with_truth(current.context)
        key = (current.formula, frozenset(f for f, _ in context))
        if key in current.ancestry:
            continue
        choices = available(current.formula, context, forward_actions=getattr(model, "forward_actions", False))
        if not choices:
            continue
        nodes += 1
        if random_seed is None:
            if hasattr(model, "score_actions"):
                candidate_logits = model.score_actions(current.formula, context, choices)
            else:
                try:
                    sequence = state_tokens(current.formula, context)
                except ValueError:
                    continue
                ids, lengths, _ = state_batch([(sequence, 0)], device)
                logits = model(ids, lengths)[0]
                selected = torch.tensor(choices, device=device)
                candidate_logits = logits[selected]
            calls += 1
            probabilities = F.log_softmax(candidate_logits, dim=0).tolist()
            ranked = sorted(zip(choices, probabilities), key=lambda pair: -pair[1])[:beam]
        else:
            if prefer_closing:
                closed = [a for a in choices if action_label(a).startswith("exact:")]
                if closed:
                    choices = closed
            rng.shuffle(choices)
            # Uniform local policy. Random order breaks ties; the probability is
            # normalized over ALL legal actions, like the learned policy above.
            ranked = [(action, -math.log(len(choices))) for action in choices[:beam]]
        for action, score in ranked:
            expression, children, next_serial = expand(current, action, serial)
            counter += 1
            heapq.heappush(frontier, (cost - score, counter, (*children, *rest),
                                     {**assignments, current.identifier: expression}, next_serial))
        # Explicit frontier bound limits memory; all policies share this cap.
        if len(frontier) > 64:
            frontier = heapq.nsmallest(64, frontier)
            heapq.heapify(frontier)
    return {"proofs": proofs, "nodes": nodes, "inference_calls": calls,
            "elapsed_seconds": time.monotonic() - start, "random_policy_seed": random_seed,
            "symbolic_solver_fallback": False, "prefer_closing": prefer_closing}
