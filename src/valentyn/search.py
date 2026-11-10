"""Bounded natural deduction with three independently selectable search orders."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import time

from .logic import Formula, evaluate

STRATEGIES = ("focused", "forward", "split")


@dataclass(frozen=True)
class SearchResult:
    status: str
    proof: str | None
    nodes: int
    elapsed: float
    strategy: str
    counterexample: dict[str, bool] | None = None


class Exhausted(Exception):
    pass


class Prover:
    def __init__(self, *, max_nodes: int = 10000, max_depth: int = 24,
                 timeout: float = 5.0):
        if not 1 <= max_nodes <= 1_000_000 or not 1 <= max_depth <= 64:
            raise ValueError("Search budget must be 1..1000000 nodes and 1..64 depth")
        if not 0 < timeout <= 300:
            raise ValueError("Search timeout must be greater than 0 and at most 300 seconds")
        self.max_nodes, self.max_depth, self.timeout = max_nodes, max_depth, timeout

    def solve(self, goal: Formula, strategy: str = "focused") -> SearchResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        start = time.monotonic()
        deadline = start + self.timeout
        nodes = 0
        serial = 0

        def tick() -> None:
            nonlocal nodes
            nodes += 1
            if nodes > self.max_nodes or time.monotonic() >= deadline:
                raise Exhausted

        def name() -> str:
            nonlocal serial
            serial += 1
            return f"h{serial}"

        def prove(target: Formula, context: dict[Formula, str], depth: int,
                  path: frozenset[tuple]) -> str | None:
            tick()
            if target in context:
                return context[target]
            if target.op == "True":
                return "True.intro"
            if depth <= 0:
                return None
            key = (target, frozenset(context))
            if key in path:
                return None
            path = path | {key}
            context = dict(context)
            # Saturate only cheap elimination rules, retaining proof terms.
            changed = True
            while changed:
                changed = False
                for formula, term in list(context.items()):
                    tick()
                    additions: list[tuple[Formula, str]] = []
                    if formula.op == "and":
                        additions = [(formula.args[0], f"(And.left {term})"),
                                     (formula.args[1], f"(And.right {term})")]
                    elif formula.op == "imp" and formula.args[0] in context:
                        additions = [(formula.args[1], f"({term} {context[formula.args[0]]})")]
                    for conclusion, evidence in additions:
                        if conclusion not in context:
                            if len(evidence) > 100000:
                                raise Exhausted
                            context[conclusion] = evidence
                            changed = True
                if target in context:
                    return context[target]
                if Formula("False") in context:
                    return f"(False.elim {context[Formula('False')]})"

            def introduce() -> str | None:
                if target.op == "imp":
                    variable = name()
                    body = prove(target.args[1], {**context, target.args[0]: variable},
                                 depth - 1, path)
                    return f"(fun {variable} => {body})" if body else None
                if target.op == "and":
                    left = prove(target.args[0], context, depth - 1, path)
                    if left:
                        right = prove(target.args[1], context, depth - 1, path)
                        if right:
                            return f"(And.intro {left} {right})"
                if target.op == "or":
                    for constructor, subgoal in zip(("Or.inl", "Or.inr"), target.args):
                        term = prove(subgoal, context, depth - 1, path)
                        if term:
                            return f"({constructor} {term})"
                return None

            def apply() -> str | None:
                implications = [(f, t) for f, t in context.items() if f.op == "imp"]
                implications.sort(key=lambda item: item[0].args[1] != target)
                for formula, term in implications:
                    if formula.args[1] in context:
                        continue
                    premise = prove(formula.args[0], context, depth - 1, path)
                    if premise:
                        evidence = f"({term} {premise})"
                        if len(evidence) > 100000:
                            raise Exhausted
                        result = prove(target, {**context, formula.args[1]: evidence},
                                       depth - 1, path)
                        if result:
                            return result
                return None

            def split() -> str | None:
                for formula, term in context.items():
                    if formula.op != "or":
                        continue
                    remainder = {f: t for f, t in context.items() if f != formula}
                    left_name, right_name = name(), name()
                    left = prove(target, {**remainder, formula.args[0]: left_name},
                                 depth - 1, path)
                    if left:
                        right = prove(target, {**remainder, formula.args[1]: right_name},
                                      depth - 1, path)
                        if right:
                            return (f"(Or.elim {term} (fun {left_name} => {left}) "
                                    f"(fun {right_name} => {right}))")
                return None

            orders = {"focused": (introduce, apply, split),
                      "forward": (apply, introduce, split),
                      "split": (split, introduce, apply)}
            for rule in orders[strategy]:
                result = rule()
                if result:
                    return result
            return None

        # A falsifying assignment is evidence against the goal, not a proof of it.
        atoms = sorted(goal.atoms())
        if len(atoms) <= 10:
            for values in product((False, True), repeat=len(atoms)):
                if time.monotonic() >= deadline:
                    return SearchResult("exhausted", None, nodes, time.monotonic() - start, strategy)
                assignment = dict(zip(atoms, values))
                if not evaluate(goal, assignment):
                    return SearchResult("counterexample", None, nodes,
                                        time.monotonic() - start, strategy, assignment)
        try:
            for depth in range(1, self.max_depth + 1):
                proof = prove(goal, {}, depth, frozenset())
                if proof:
                    if len(proof) > 100000:
                        raise Exhausted
                    return SearchResult("candidate", proof, nodes,
                                        time.monotonic() - start, strategy)
        except Exhausted:
            pass
        return SearchResult("exhausted", None, min(nodes, self.max_nodes),
                            time.monotonic() - start, strategy)
