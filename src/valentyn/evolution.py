"""Validated policy mutation with regression gates on a separate selection split."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

from .engine import Engine
from .experiments import corpus, digest
from .policy import Policy


def isolated_search(engine: Engine, policy: Policy, tasks: list[dict]) -> list[dict]:
    request = {"policy": policy.to_dict(), "tasks": tasks, "nodes": engine.prover.max_nodes,
               "depth": engine.prover.max_depth, "timeout": engine.prover.timeout}
    # -I excludes PYTHONPATH and user site packages. Install Valentyn into a venv.
    environment = {"PATH": os.defpath, "LANG": "C", "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryDirectory(prefix="valentyn-worker-") as directory:
        try:
            completed = subprocess.run([sys.executable, "-I", "-m", "valentyn.worker"],
                                       input=json.dumps(request), text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       cwd=directory, env=environment, timeout=30, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Strategy evaluation worker exceeded its wall time") from error
    if completed.returncode:
        raise RuntimeError(f"Strategy worker failed: {completed.stderr[-2000:]}")
    if len(completed.stdout) > 2_000_000:
        raise RuntimeError("Strategy worker output exceeds limit")
    results = json.loads(completed.stdout)
    if not isinstance(results, list) or len(results) != len(tasks):
        raise RuntimeError("Strategy worker returned an invalid result count")
    for task, result in zip(tasks, results):
        if result["task_id"] != task["id"] or result["formula"] != task["formula"]:
            raise RuntimeError("Strategy worker changed the evaluation task")
    return results


def evolve(engine: Engine, policy: Policy, *, generations: int = 8, seed: int = 0) -> tuple[Policy, dict]:
    if not 1 <= generations <= 1000:
        raise ValueError("Generations must be 1..1000")
    tasks = corpus()["validation"]
    rng = random.Random(seed)

    def assess(candidate: Policy) -> tuple[tuple[int, int], set[str], list[dict]]:
        rows = [engine.accept_search(row["formula"], row, task_id=row["task_id"], split="validation")
                for row in isolated_search(engine, candidate, tasks)]
        solved = {r["task_id"] for r in rows if r["status"] == "verified"}
        score = (len(solved), -sum(r["nodes"] for r in rows))
        if any(r["status"] == "rejected" for r in rows):
            raise RuntimeError("Verification failure during evolution; refusing promotion")
        return score, solved, rows

    incumbent = Policy.from_dict(policy.to_dict())
    score, solved, initial_rows = assess(incumbent)
    initial_score = score
    lineage = []
    for generation in range(generations):
        candidate = incumbent.mutate(rng, scale=0.2)
        proposed_score, proposed_solved, rows = assess(candidate)
        accepted = proposed_score > score and solved <= proposed_solved
        lineage.append({"generation": generation, "candidate_digest": candidate.digest(),
                        "score": list(proposed_score), "promoted": accepted,
                        "regressions": sorted(solved - proposed_solved),
                        "attempt_ids": [r["attempt_id"] for r in rows]})
        if accepted:
            incumbent, score, solved = candidate, proposed_score, proposed_solved
    report = {"kind": "evolution", "seed": seed, "generations": generations,
              "selection_split": "validation", "selection_digest": digest(tasks),
              "initial_score": list(initial_score), "final_score": list(score),
              "promotions": sum(r["promoted"] for r in lineage), "lineage": lineage,
              "policy_digest": incumbent.digest(),
              "initial_attempt_ids": [r["attempt_id"] for r in initial_rows],
              "scope": "Numeric strategy-selection policies only. No generated code is executed.",
              "isolation": "Fresh Python -I process, minimal environment, CPU/file/descriptor limits and wall timeout; not an OS sandbox.",
              "limitation": "Validation nonregression does not guarantee test or real-world nonregression."}
    engine.memory.experiment("evolution", report)
    return incumbent, report
