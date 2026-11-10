"""Reproducible train/validation/test separation and controlled policy experiments."""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import random

from .engine import Engine
from .logic import Formula, parse
from .policy import Policy, reward
from .search import STRATEGIES


def canonical(goal: Formula) -> tuple:
    names: dict[str, int] = {}

    def visit(formula: Formula) -> tuple:
        if formula.args:
            return (formula.op, *(visit(arg) for arg in formula.args))
        if formula.op in ("True", "False"):
            return (formula.op,)
        return ("atom", names.setdefault(formula.op, len(names)))

    return visit(goal)


def corpus() -> dict:
    data = json.loads(files("valentyn").joinpath("data/problems.json").read_text())
    seen: dict[tuple, str] = {}
    identifiers: set[str] = set()
    for split in ("train", "validation", "test"):
        for task in data[split]:
            key = canonical(parse(task["formula"]))
            if key in seen:
                raise ValueError(f"Duplicate formula structure across corpus: {task['id']} and {seen[key]}")
            if task["id"] in identifiers:
                raise ValueError(f"Duplicate task identifier: {task['id']}")
            seen[key] = task["id"]
            identifiers.add(task["id"])
    return data


def digest(data: object) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def train(engine: Engine, *, epochs: int = 100, seed: int = 0) -> tuple[Policy, dict]:
    if not 1 <= epochs <= 10000:
        raise ValueError("Epochs must be 1..10000")
    tasks = corpus()["train"]
    policy = Policy()
    observations = []
    for task in tasks:
        for strategy in STRATEGIES:
            result = engine.attempt(task["formula"], strategy, task_id=task["id"], split="train")
            if result["status"] == "rejected":
                raise RuntimeError(f"Training halted: verification failed for {task['id']}: "
                                   f"{result['verification']['reason']}")
            observations.append((task["formula"], strategy,
                                 reward(result["status"] == "verified", result["nodes"],
                                        engine.prover.max_nodes)))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(observations)
        for formula, strategy, target in observations:
            policy.update(parse(formula), strategy, target)
    policy.training_digest = digest(tasks)
    report = {"kind": "train", "seed": seed, "epochs": epochs,
              "tasks": len(tasks), "search_observations": len(observations),
              "verified_observations": sum(target > 0 for _, _, target in observations),
              "budget_nodes_per_attempt": engine.prover.max_nodes,
              "corpus_digest": digest(tasks), "policy_digest": policy.digest(),
              "observations": observations,
              "note": "Observations include failures with zero reward; epochs reuse these observations."}
    engine.memory.experiment("train", report)
    return policy, report


def evaluate(engine: Engine, policy: Policy, *, split: str = "test") -> dict:
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    tasks = corpus()[split]
    rows = []
    for task in tasks:
        strategy = policy.choose(parse(task["formula"]))
        for label, selected in (("baseline", "focused"), ("fixed_forward", "forward"),
                                ("fixed_split", "split"), ("learned", strategy)):
            result = engine.attempt(task["formula"], selected, task_id=task["id"], split=split)
            rows.append({"arm": label, **result})
    summaries = {}
    for arm in ("baseline", "fixed_forward", "fixed_split", "learned"):
        selected = [r for r in rows if r["arm"] == arm]
        summaries[arm] = {
            "verified": sum(r["status"] == "verified" for r in selected),
            "counterexamples": sum(r["status"] == "counterexample" for r in selected),
            "exhausted": sum(r["status"] == "exhausted" for r in selected),
            "rejected": sum(r["status"] == "rejected" for r in selected),
            "total_nodes": sum(r["nodes"] for r in selected),
            "mean_reward": sum(reward(r["status"] == "verified", r["nodes"],
                                      engine.prover.max_nodes) for r in selected) / len(selected),
        }
    regressions = [task["id"] for task in tasks
                   if next(r for r in rows if r["task_id"] == task["id"] and r["arm"] == "baseline")["status"] == "verified"
                   and next(r for r in rows if r["task_id"] == task["id"] and r["arm"] == "learned")["status"] != "verified"]
    report = {"kind": "benchmark", "split": split, "tasks": len(tasks),
              "corpus_digest": digest(tasks), "policy_digest": policy.digest(),
              "budget": {"nodes": engine.prover.max_nodes, "depth": engine.prover.max_depth,
                         "timeout_seconds": engine.prover.timeout},
              "summary": summaries, "regressions": regressions, "results": rows,
              "limitations": ["Small public synthetic suite; no mathematical novelty claim.",
                              "Test is held out from fitting, not from human development.",
                              "Classically valid nonconstructive formulas may remain exhausted.",
                              "Equal search budgets; model fitting and evolution costs reported separately."]}
    engine.memory.experiment("benchmark", report)
    return report


def propose_formula(rng: random.Random, depth: int) -> Formula:
    if depth <= 0 or rng.random() < 0.25:
        return Formula(rng.choice(("p", "q", "r", "False", "True")))
    return Formula(rng.choice(("imp", "and", "or")),
                   (propose_formula(rng, depth - 1), propose_formula(rng, depth - 1)))


def discover(engine: Engine, policy: Policy, *, count: int = 20, seed: int = 0) -> dict:
    if not 1 <= count <= 10000:
        raise ValueError("Discovery count must be 1..10000")
    rng = random.Random(seed)
    results = []
    seen = set()
    for _ in range(count * 20):
        goal = propose_formula(rng, 3)
        key = canonical(goal)
        if key in seen:
            continue
        seen.add(key)
        formula = goal.lean().replace("∧", "&").replace("∨", "|").replace("→", "->")
        previously_known = engine.memory.contains(formula)
        result = engine.attempt(formula, policy.choose(goal), task_id=f"generated-{seed}-{len(results)}",
                                split="generated")
        result["new_to_local_archive"] = result["status"] == "verified" and not previously_known
        results.append(result)
        if len(results) == count:
            break
    report = {"kind": "discovery", "seed": seed, "generated": len(results),
              "verified": sum(r["status"] == "verified" for r in results),
              "new_to_local_archive": sum(r["new_to_local_archive"] for r in results),
              "novelty_status": "Not checked against mathematical literature; elementary propositions only.",
              "results": results}
    engine.memory.experiment("discovery", report)
    return report
