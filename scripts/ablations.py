"""Executable low-cost probes for portfolio, curriculum, and transfer hypotheses.

These development experiments use validation, never the final test split.
Every solved attempt still passes the normal Lean gate. No neural models needed.
"""

import argparse
import json
from pathlib import Path
import random

from valentyn.cli import provenance
from valentyn.engine import Engine
from valentyn.experiments import corpus, digest, evaluate
from valentyn.logic import parse
from valentyn.memory import Memory
from valentyn.policy import Policy, reward
from valentyn.search import STRATEGIES

ROOT = Path(__file__).resolve().parents[1]


def portfolio(memory: Memory, nodes: int) -> dict:
    tasks = corpus()["validation"]
    rows = []
    for task in tasks:
        single = Engine(ROOT, memory, nodes=nodes).attempt(
            task["formula"], "focused", task_id=task["id"], split="validation")
        attempts = []
        # Each member receives floor(B / 3) nodes. Total allocation never exceeds B.
        for strategy in STRATEGIES:
            result = Engine(ROOT, memory, nodes=nodes // len(STRATEGIES),
                            timeout=5 / len(STRATEGIES)).attempt(
                task["formula"], strategy, task_id=task["id"], split="validation")
            attempts.append(result)
            if result["status"] in ("verified", "counterexample"):
                break
        rows.append({"task": task["id"], "baseline": single, "portfolio": attempts})
    return {"experiment": "portfolio", "nodes_budget": nodes,
            "baseline_solved": sum(r["baseline"]["status"] == "verified" for r in rows),
            "portfolio_solved": sum(any(a["status"] == "verified" for a in r["portfolio"]) for r in rows),
            "baseline_nodes": sum(r["baseline"]["nodes"] for r in rows),
            "portfolio_nodes": sum(sum(a["nodes"] for a in r["portfolio"]) for r in rows),
            "note": "Sequential portfolio; each member also receives 1/3 of baseline search wall budget.",
            "results": rows}


def curriculum(memory: Memory, nodes: int, seed: int, mode: str) -> dict:
    engine = Engine(ROOT, memory, nodes=nodes)
    tasks = corpus()["train"]
    observations = []
    for task in tasks:
        for strategy in STRATEGIES:
            row = engine.attempt(task["formula"], strategy, task_id=task["id"], split="train")
            if row["status"] == "rejected":
                raise RuntimeError("Verification failure while collecting observations")
            observations.append((task["formula"], strategy,
                                 reward(row["status"] == "verified", row["nodes"], nodes)))
    rng = random.Random(seed)
    reports = {}
    if mode == "curriculum":
        schedules = {
            "random": rng.sample(observations, len(observations)),
            "easy_to_hard": sorted(observations, key=lambda item: parse(item[0]).size()),
            "hard_to_easy": sorted(observations, key=lambda item: -parse(item[0]).size()),
        }
    else:
        # Leave disjunction tasks out entirely, not merely their task identifiers.
        schedules = {"all": observations,
                     "without_disjunction": [row for row in observations if "|" not in row[0]]}
    for name, schedule in schedules.items():
        policy = Policy()
        # Equal update count even for differently sized training subsets.
        for index in range(360):
            formula, strategy, target = schedule[index % len(schedule)]
            policy.update(parse(formula), strategy, target)
        policy.training_digest = digest(schedule)
        reports[name] = evaluate(engine, policy, split="validation")
    return {"experiment": mode, "seed": seed, "updates_per_arm": 360,
            "corpus_digest": digest(tasks), "arms": reports,
            "limitation": "Development probe on six validation formulas; no significance or novelty claim."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("portfolio", "curriculum", "transfer"))
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.nodes < len(STRATEGIES):
        parser.error("Node budget must be at least 3")
    memory = Memory(ROOT / ".valentyn/ablations.sqlite3")
    report = (portfolio(memory, args.nodes) if args.experiment == "portfolio" else
              curriculum(memory, args.nodes, args.seed, args.experiment))
    report["provenance"] = provenance(ROOT)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "arms")}, indent=2))


if __name__ == "__main__":
    main()
