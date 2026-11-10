"""Permutation-invariant learned scoring of legal proof actions.

Feature extraction inspects syntax and direct facts only: it performs no proof
search, implication saturation, or calls to the teacher. Handwritten controls use
exactly these same features and the same proof-construction environment.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .logic import Formula, parse
from .proof_states import ACTIONS, action_label, extract_steps, premises, with_truth

VERSION = 1
KINDS = ('intro', 'pair', 'left', 'right', 'absurd', 'exact', 'apply', 'cases')


def action_features(goal, context, action, *, forward_actions=False):
    context = with_truth(context)
    label = action_label(action)
    kind = label.split(':')[0]
    known = {f for f, _ in context}
    source = context[int(label.split(':')[1])][0] if ':' in label else goal
    if kind == 'intro':
        required = [goal.args[1]]
        known = known | {goal.args[0]}
    elif kind == 'pair':
        required = list(goal.args)
    elif kind in ('left', 'right'):
        required = [goal.args[kind == 'right']]
    elif kind == 'derive':
        required = [source.args[0]]
    elif kind == 'apply':
        required = premises(source, goal)
    elif kind == 'cases':
        required = [goal, goal]
    elif kind == 'absurd':
        required = [] if Formula('False') in known else [Formula('False')]
    else:
        required = []
    unknown = [f for f in required if f not in known]
    atoms = set().union(*(f.atoms() for f in known))
    total = sum(f.size() for f in required)
    result = [float(("apply" if kind == "derive" else kind) == k) for k in KINDS] + [
        len(required) / 8, len(unknown) / 8, float(not unknown),
        math.log1p(total) / 5, math.log1p(sum(f.size() for f in unknown)) / 5,
        math.log1p(goal.size()) / 5, math.log1p(source.size()) / 5,
        len(context) / 32, len(goal.atoms()) / 11,
        float(goal in known), float(Formula('False') in known),
        sum(f.op == 'imp' for f in known) / 32,
        sum(f.op == 'or' for f in known) / 32,
        sum(f.op == 'and' for f in known) / 32,
        sum(bool(f.atoms() - atoms) for f in unknown) / 8,
        sum(f.op == 'imp' for f in unknown) / 8,
        sum(f.op == 'and' for f in unknown) / 8,
        sum(f.op == 'or' for f in unknown) / 8,
        float(kind == 'cases' and goal in source.args),
        float(kind == 'intro' and goal.args[1] == goal.args[0]),
    ]

    if forward_actions:
        def projects(formula):
            return formula == goal or (formula.op == 'and' and any(projects(f) for f in formula.args))
        conclusion = source.args[1] if kind == 'derive' else source
        result += [float(kind == 'derive'), float(kind == 'derive' and projects(conclusion)),
                   float(kind == 'derive' and conclusion.op == 'imp')]
    return result

FEATURES = 28


class RelationalPolicy(nn.Module):
    def __init__(self, width=64, forward_actions=False):
        super().__init__()
        if width not in (32, 64, 96):
            raise ValueError('Unsupported relational width')
        self.width = width
        self.forward_actions = forward_actions
        self.network = nn.Sequential(nn.Linear(FEATURES + 3 * forward_actions, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)

    def score_actions(self, goal, context, choices):
        device = next(self.parameters()).device
        return self(torch.tensor([action_features(goal, context, a, forward_actions=self.forward_actions) for a in choices], device=device))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def heuristic_scores(goal, context, choices, variant='structural', forward_actions=False):
    """Explicit strong controls; these are NOT learned model outputs."""
    output = []
    for action in choices:
        row = action_features(goal, context, action, forward_actions=forward_actions)
        kind = action_label(action).split(':')[0]
        kind = "apply" if kind == "derive" else kind
        if variant == 'ordered':
            score = {'exact': 10, 'absurd': 8, 'intro': 7, 'pair': 6,
                     'apply': 5, 'left': 4, 'right': 4, 'cases': 3}[kind]
        elif variant in ('structural', 'relevance'):
            score = 20 * row[10] - 8 * row[9] - row[12]
            score += {'exact': 10, 'absurd': 1, 'intro': 3, 'pair': 2,
                      'apply': 1, 'left': 0, 'right': 0, 'cases': -3}[kind]
            if variant == 'relevance' and forward_actions:
                score += 20 * row[29] + 2 * row[30] - 2 * row[28]
        else:
            raise ValueError('Unknown heuristic control')
        output.append(score)
    return torch.tensor(output, dtype=torch.float32)


class HeuristicPolicy(RelationalPolicy):
    def __init__(self, variant, forward_actions=False):
        super().__init__(32, forward_actions=forward_actions)
        self.variant = variant

    def score_actions(self, goal, context, choices):
        return heuristic_scores(goal, context, choices, self.variant, self.forward_actions)


def save(model, path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights = path / 'weights.pt'
    torch.save(model.state_dict(), weights)
    manifest = {'schema': VERSION, 'architecture': 'relational-action-policy', 'width': model.width,
                'features': FEATURES + 3 * model.forward_actions, 'forward_actions': model.forward_actions, 'parameters': model.parameter_count,
                'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'metadata': metadata}
    (path / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load(path, device='cpu'):
    path = Path(path)
    manifest = json.loads((path / 'model.json').read_text())
    if manifest['schema'] != VERSION or manifest['architecture'] != 'relational-action-policy' or manifest['features'] != FEATURES + 3 * manifest.get('forward_actions', False):
        raise ValueError('Incompatible relational checkpoint')
    weights = path / 'weights.pt'
    if weights.stat().st_size > 1024**2 or hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['weights_sha256']:
        raise ValueError('Invalid relational weights')
    model = RelationalPolicy(manifest['width'], forward_actions=manifest.get('forward_actions', False))
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    return model.to(device).eval(), manifest


def training_rows(data, split):
    from .proof_states import available
    rows = []
    for task in data[split]:
        for step in extract_steps(parse(task['formula']), task['proof']):
            choices = available(step['goal'], step['context'])
            rows.append(([action_features(step['goal'], step['context'], a) for a in choices],
                         choices.index(step['action'])))
    return rows


def pack(rows):
    features = torch.zeros(len(rows), max(len(r[0]) for r in rows), len(rows[0][0][0]))
    mask = torch.zeros(features.shape[:2], dtype=torch.bool)
    for index, (values, _) in enumerate(rows):
        features[index, :len(values)] = torch.tensor(values)
        mask[index, :len(values)] = True
    return features, mask, torch.tensor([r[1] for r in rows])


def train(data, output, *, seed=7, epochs=80, width=64, batch_size=128, learning_rate=.002, progress=None):
    if not 1 <= epochs <= 300 or not 1 <= batch_size <= 512 or not 0 < learning_rate <= .1:
        raise ValueError('Invalid training budget')
    from .resources import ResourceBudget
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    start = time.monotonic()
    rows = {split: training_rows(data, split) for split in ('train', 'validation')}
    if not all(rows.values()):
        raise ValueError('Empty fitting split')
    model = RelationalPolicy(width)
    budget = ResourceBudget()
    budget.preflight(model.parameter_count, batch_size, 32, width, 2, 1)
    metadata = {'seed': seed, 'dataset_digest': data['digest'], 'pretrained_weights': False,
                'selection': 'validation masked action loss only', 'feature_version': VERSION}
    save(model, Path(output) / 'initial', metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    best, history = float('inf'), []
    validation = pack(rows['validation'])
    for epoch in range(epochs):
        model.train()
        ordered = rng.sample(rows['train'], len(rows['train']))
        for index in range(0, len(ordered), batch_size):
            x, mask, labels = pack(ordered[index:index + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).masked_fill(~mask, -1e9), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        budget.guard(torch.device('cpu'))
        model.eval()
        with torch.inference_mode():
            x, mask, labels = validation
            logits = model(x).masked_fill(~mask, -1e9)
            loss = F.cross_entropy(logits, labels).item()
            accuracy = (logits.argmax(-1) == labels).float().mean().item()
        history.append({'epoch': epoch+1, 'validation_loss': loss, 'validation_action_accuracy': accuracy})
        if loss < best:
            best = loss
            save(model, output, {**metadata, 'best_epoch': epoch+1})
        if progress and (epoch+1) % 10 == 0:
            progress(history[-1])
    report = {'kind': 'relational_training', **metadata, 'parameters': model.parameter_count,
              'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
              'training_states': len(rows['train']), 'validation_states': len(rows['validation']),
              'history': history, 'training_seconds': time.monotonic()-start}
    (Path(output) / 'training.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def train_cost(data, output, *, seed=7, epochs=100, width=64, batch_size=128,
               learning_rate=.002, progress=None):
    """Train from scratch using alternative-action labels on fitting states only."""
    if not 1 <= epochs <= 300 or not 1 <= batch_size <= 512 or not 0 < learning_rate <= .1:
        raise ValueError('Invalid training budget')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    start = time.monotonic()
    rows = {}
    for split in ('train', 'validation'):
        rows[split] = []
        for row in data[split]:
            if len(row['features']) != len(row['costs']) or not row['costs']:
                raise ValueError('Invalid cost supervision')
            finite = [c for c in row['costs'] if c is not None]
            if not finite:
                continue
            best = min(finite)
            rows[split].append((row['features'], [c == best for c in row['costs']]))
    if not all(rows.values()):
        raise ValueError('Empty cost fitting split')

    def batch(examples):
        x, mask, _ = pack([(x, 0) for x, _ in examples])
        good = torch.zeros(mask.shape, dtype=torch.bool)
        for i, (_, labels) in enumerate(examples):
            good[i, :len(labels)] = torch.tensor(labels)
        return x, mask, good

    def objective(logits, mask, good):
        logprob = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
        return -torch.logsumexp(logprob.masked_fill(~good, -1e9), dim=-1).mean()

    model = RelationalPolicy(width, forward_actions=data.get('forward_actions', False))
    from .resources import ResourceBudget
    budget = ResourceBudget()
    budget.preflight(model.parameter_count, batch_size, 32, width, 2, 1)
    metadata = {'seed': seed, 'dataset_digest': data['dataset_digest'], 'pretrained_weights': False,
                'objective': 'masked multi-optimal action likelihood from bounded cost oracle',
                'selection': 'validation oracle-action loss only', 'feature_version': VERSION}
    save(model, Path(output) / 'initial', metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    best, history = float('inf'), []
    validation = batch(rows['validation'])
    for epoch in range(epochs):
        model.train()
        ordered = rng.sample(rows['train'], len(rows['train']))
        for index in range(0, len(ordered), batch_size):
            x, mask, good = batch(ordered[index:index + batch_size])
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x), mask, good)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        budget.guard(torch.device('cpu'))
        model.eval()
        with torch.inference_mode():
            x, mask, good = validation
            logits = model(x).masked_fill(~mask, -1e9)
            loss = objective(logits, mask, good).item()
            accuracy = good.gather(1, logits.argmax(-1, keepdim=True)).float().mean().item()
        history.append({'epoch': epoch+1, 'validation_loss': loss, 'validation_action_accuracy': accuracy})
        if loss < best:
            best = loss
            save(model, output, {**metadata, 'best_epoch': epoch+1})
        if progress and (epoch+1) % 20 == 0:
            progress(history[-1])
    report = {'kind': 'relational_cost_training', **metadata, 'parameters': model.parameter_count,
              'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
              'training_states': len(rows['train']), 'validation_states': len(rows['validation']),
              'history': history, 'training_seconds': time.monotonic()-start}
    (Path(output) / 'training.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
