"""Evaluate model-chosen proof trees and a legality-matched random-policy control."""

from dataclasses import asdict
from pathlib import Path
import time

import torch

from .factorized import generate_proof, load_policy
from .logic import parse
from .neural import memory_measurement, select_device
from .proof_data import TEMPLATES
from .verifier import LeanVerifier


def attempt(model, formula, verifier, memory, *, task_id="interactive", split="interactive",
            nodes=200, beam=1, random_seed=None, policy_name=None):
    generation = generate_proof(model, formula, max_nodes=nodes, beam=beam, random_seed=random_seed,
                                prefer_closing=policy_name == "factorized-random-closed")
    attempts = []
    for proof in generation["proofs"]:
        checked = verifier.verify(parse(formula), proof)
        row = {"task_id": task_id, "split": split, "formula": formula,
               "strategy": policy_name or ("factorized-neural" if random_seed is None else "factorized-random"),
               "status": "verified" if checked.accepted else "rejected", "proof": proof,
               "nodes": generation["nodes"], "verification": asdict(checked)}
        row["attempt_id"] = memory.record(row)
        attempts.append(row)
        if not checked.accepted:
            raise RuntimeError(f"Typed construction failed Lean; refusing results: {checked.diagnostics}")
    if not attempts:
        memory.record({"task_id": task_id, "split": split, "formula": formula,
                       "strategy": policy_name or ("factorized-neural" if random_seed is None else "factorized-random"),
                       "status": "exhausted", "nodes": generation["nodes"],
                       "proof": None, "verification": None, "generation": generation})
    return {"task_id": task_id, "formula": formula, "solved": bool(attempts),
            "attempts": attempts, "generation": generation, "symbolic_fallback": False}


def evaluate(project, memory, checkpoint: Path, data, *, split="test", beam=1, nodes=200,
             random_seed=None, device_name="cpu", limit=None, progress=None, control=None, verifier=None):
    if split not in ("train", "validation", "test") or limit is not None and not 1 <= limit <= len(data[split]):
        raise ValueError("Invalid factorized evaluation split or limit")
    torch.set_num_threads(2)
    device = select_device(device_name)
    model, manifest = load_policy(checkpoint, device)
    if control is not None:
        from .relational import HeuristicPolicy
        if control not in ("ordered", "structural", "relevance", "random-closed"):
            raise ValueError("Unknown heuristic control")
        if control == "random-closed":
            random_seed = 0 if random_seed is None else random_seed
        else:
            model = HeuristicPolicy(control, forward_actions=getattr(model, "forward_actions", False)).to(device).eval()
    verifier = verifier or LeanVerifier(project)
    smoke = verifier.verify(parse("p -> p"), "(fun h0 => h0)")
    if not smoke.accepted:
        raise RuntimeError("Lean preflight failed")
    start = time.monotonic()
    rows = []
    for index, task in enumerate(data[split][:limit]):
        row = attempt(model, task["formula"], verifier, memory, task_id=task["id"],
                      split=f"factorized-{split}", nodes=nodes, beam=beam,
                      random_seed=None if random_seed is None else random_seed + index,
                      policy_name=f"factorized-{control}" if control else None)
        row["family"] = task["family"]
        row["policy"] = control or ("random" if random_seed is not None else "learned")
        rows.append(row)
        if progress and (index + 1) % 10 == 0:
            progress(f"{index + 1} evaluated; {sum(r['solved'] for r in rows)} verified")
    report = {"kind": "factorized_evaluation", "split": split,
              "dataset_digest": data["digest"], "weights_sha256": manifest["weights_sha256"],
              "parameters": manifest["parameters"], "device": str(device), "torch_version": str(torch.__version__),
              "tasks": len(rows), "solved": sum(r["solved"] for r in rows),
              "max_nodes": nodes, "beam": beam, "random_policy_seed": random_seed,
              "inference_calls": sum(r["generation"]["inference_calls"] for r in rows),
              "nodes": sum(r["generation"]["nodes"] for r in rows),
              "inference_seconds": sum(r["generation"]["elapsed_seconds"] for r in rows),
              "total_evaluation_seconds": time.monotonic() - start,
              "slices": {label: {"tasks": len(selected), "solved": sum(r["solved"] for r in selected)}
                         for label, selected in (("substitution", [r for r in rows if r["family"] in TEMPLATES]),
                           ("held_out_family", [r for r in rows if r["family"] not in TEMPLATES]))},
              "memory": memory_measurement(device), "results": rows,
              "scope": "Learned local action policy with typed inference rules; no calls to the symbolic search engine.",
              "limitations": ["Legality masks and conjunction projection supply symbolic structure.",
                              "Same typed environment must be used for untrained and random controls.",
                              "Synthetic family transfer is not novel mathematical discovery."]}
    memory.experiment("factorized_evaluation", {k: v for k, v in report.items() if k != "results"})
    return report
