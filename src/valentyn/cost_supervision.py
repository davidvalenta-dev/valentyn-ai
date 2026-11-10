"""Bounded training-only proof-cost oracle and alternative-action supervision.

Unknown is censored search exhaustion, never a disproof. Only fitting splits are
accepted. The oracle is not imported or called by learned proof generation.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

from .factorized import Goal, expand
from .logic import parse
from .proof_states import available, extract_steps, with_truth
from .relational import action_features


class CostOracle:
    def __init__(self, max_nodes=256, max_depth=12, forward_actions=False):
        self.max_nodes, self.max_depth = max_nodes, max_depth
        self.nodes = 0
        self.forward_actions = forward_actions

    def estimate(self, goal, context, depth=None, path=frozenset()):
        depth = self.max_depth if depth is None else depth
        context = with_truth(context)
        key = (goal, frozenset(f for f, _ in context))
        if depth <= 0 or self.nodes >= self.max_nodes or key in path:
            return None
        choices = available(goal, context, forward_actions=self.forward_actions)
        if not choices:
            return None
        best = None
        for action in choices:
            self.nodes += 1
            if self.nodes > self.max_nodes:
                break
            _, children, _ = expand(Goal(0, goal, context), action, 0)
            cost = 1
            for child in children:
                value = self.estimate(child.formula, child.context, depth-1, path | {key})
                if value is None:
                    cost = None
                    break
                cost += value
            if cost is not None:
                best = cost if best is None else min(best, cost)
                if best == 1:
                    break
        return best

    def actions(self, goal, context):
        choices = available(goal, context, forward_actions=self.forward_actions)
        values = []
        for action in choices:
            self.nodes = 0
            _, children, _ = expand(Goal(0, goal, tuple(context)), action, 0)
            total = 1
            root = (goal, frozenset(f for f, _ in with_truth(context)))
            for child in children:
                cost = self.estimate(child.formula, child.context, path=frozenset({root}))
                if cost is None:
                    total = None
                    break
                total += cost
            values.append(total)
        return choices, values


def collect(data, *, seed=17, augment=1, max_states=12000, progress=None, forward_actions=False):
    rng = random.Random(seed)
    start = time.monotonic()
    report = {'schema': 1, 'dataset_digest': data['digest'], 'seed': seed,
              'oracle': {'max_nodes_per_action': 256, 'max_depth': 12},
              'test_accessed': False, 'forward_actions': forward_actions, 'train': [], 'validation': []}
    counts = {'unknown_actions': 0, 'skipped_no_closed_action': 0}
    oracle = CostOracle(forward_actions=forward_actions)
    for split in ('train', 'validation'):
        seen = set()
        for task in data[split]:
            states = extract_steps(parse(task['formula']), task['proof'])
            for step in states:
                context = step['context']
                pending = [(step['goal'], context)]
                if split == 'train':
                    choices = available(step['goal'], context, forward_actions=forward_actions)
                    for action in rng.sample(choices, min(augment, len(choices))):
                        _, children, _ = expand(Goal(0, step['goal'], context), action, 0)
                        pending.extend((c.formula, c.context) for c in children)
                for goal, context in pending:
                    key = (goal, frozenset(f for f, _ in with_truth(context)))
                    if key in seen:
                        continue
                    seen.add(key)
                    choices, values = oracle.actions(goal, context)
                    counts['unknown_actions'] += sum(v is None for v in values)
                    if not any(v is not None for v in values):
                        counts['skipped_no_closed_action'] += 1
                        continue
                    # Censored alternatives have no found proof within the oracle
                    # budget. Their penalty is a training heuristic, not falsity.
                    report[split].append({'features': [action_features(goal, context, a, forward_actions=forward_actions) for a in choices],
                                          'costs': values, 'source_task': task['id']})
                if len(report[split]) >= max_states:
                    break
            if len(report[split]) >= max_states:
                break
        if progress:
            progress(f'{split}: {len(report[split])} cost-labelled states')
    report.update(counts)
    report['collection_seconds'] = time.monotonic()-start
    return report
