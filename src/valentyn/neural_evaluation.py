"""Independent evaluation of model-produced terms, with no symbolic fallback."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .logic import parse
from .memory import Memory
from .neural import generate, load_model, memory_measurement, select_device, tokens
from .proof_data import TEMPLATES
from .verifier import LeanVerifier


def model_attempt(model, formula: str, verifier: LeanVerifier, memory: Memory, *,
                  task_id: str, split: str, beam: int = 1, max_tokens: int = 100) -> dict:
    generation = generate(model, formula, beam=beam, max_tokens=max_tokens)
    attempts = []
    for index, proof in enumerate(generation["proofs"]):
        checked = verifier.verify(parse(formula), proof)
        attempt = {"task_id": task_id, "split": split, "formula": formula,
                   "strategy": f"neural-{model.config.kind}",
                   "status": "verified" if checked.accepted else "rejected",
                   "nodes": generation["expanded_tokens"], "proof": proof,
                   "verification": asdict(checked), "candidate_rank": index}
        attempt["attempt_id"] = memory.record(attempt)
        attempts.append(attempt)
        if checked.accepted:
            break
        if checked.reason in ("lean_unavailable", "toolchain_mismatch", "lean_error"):
            raise RuntimeError(f"Evaluation infrastructure failed: {checked.reason}")
    if not attempts:
        memory.record({"task_id": task_id, "split": split, "formula": formula,
                       "strategy": f"neural-{model.config.kind}", "status": "exhausted",
                       "nodes": generation["expanded_tokens"], "proof": None,
                       "verification": None, "generation": generation})
    return {"task_id": task_id, "formula": formula, "solved": any(
        row["status"] == "verified" for row in attempts),
        "generation": generation, "attempts": attempts,
        "symbolic_fallback": False}


def evaluate_model(project: Path, memory: Memory, checkpoint: Path, data: dict, *,
                   split: str = "test", beam: int = 1, max_tokens: int = 100,
                   device_name: str = "cpu", limit: int | None = None,
                   progress=None) -> dict:
    if split not in ("validation", "test", "train"):
        raise ValueError("Invalid model evaluation split")
    if limit is not None and not 1 <= limit <= len(data[split]):
        raise ValueError("Evaluation limit outside dataset size")
    torch.set_num_threads(2)
    device = select_device(device_name)
    load_start = time.monotonic()
    model, manifest = load_model(checkpoint, device)
    loading_seconds = time.monotonic() - load_start
    verifier = LeanVerifier(project)
    smoke = verifier.verify(parse("p -> p"), "(fun h0 => h0)")
    if not smoke.accepted:
        raise RuntimeError(f"Lean preflight failed: {smoke.reason}")
    # Only formulas are passed to the model. Reference proofs are not consulted.
    tasks = data[split][:limit]
    start = time.monotonic()
    results = []
    for index, task in enumerate(tasks):
        result = model_attempt(model, task["formula"], verifier, memory,
                               task_id=task["id"], split=f"neural-{split}",
                               beam=beam, max_tokens=max_tokens)
        result["family"] = task["family"]
        results.append(result)
        if progress and (index + 1) % 10 == 0:
            progress(f"{index + 1}/{len(tasks)} evaluated; {sum(r['solved'] for r in results)} verified")
    slices = {}
    for label, selected in (("substitution", [r for r in results if r["family"] in TEMPLATES]),
                            ("held_out_family", [r for r in results if r["family"] not in TEMPLATES])):
        slices[label] = {"tasks": len(selected), "solved": sum(r["solved"] for r in selected)}
    report = {"kind": "neural_evaluation", "split": split, "dataset_digest": data["digest"],
              "weights_sha256": manifest["weights_sha256"], "architecture": manifest["config"],
              "parameters": manifest["parameters"], "device": str(device), "torch_version": str(torch.__version__),
              "tasks": len(tasks), "solved": sum(r["solved"] for r in results),
              "beam": beam, "max_generated_tokens": max_tokens,
              "inference_calls": sum(r["generation"]["inference_calls"] for r in results),
              "inference_seconds": sum(r["generation"]["elapsed_seconds"] for r in results),
              "total_evaluation_seconds": time.monotonic() - start,
              "loading_seconds": loading_seconds, "memory": memory_measurement(device),
              "slices": slices, "results": results, "symbolic_fallback": False,
              "note": "Valid proofs must pass Lean and its zero-axiom audit. Failed predictions remain in the log."}
    memory.experiment("neural_evaluation", {k: v for k, v in report.items() if k != "results"})
    return report


def retrieval_baseline(project: Path, memory: Memory, data: dict, *, split: str = "test") -> dict:
    """A deliberately strong copying control, using only training proof terms."""
    reference = [(set(tokens(row["formula"])), row["proof"]) for row in data["train"]]
    verifier = LeanVerifier(project)
    results = []
    start = time.monotonic()
    for task in data[split]:
        query = set(tokens(task["formula"]))
        _, proof = max(reference, key=lambda pair: len(query & pair[0]) / max(len(query | pair[0]), 1))
        checked = verifier.verify(parse(task["formula"]), proof)
        row = {"task_id": task["id"], "family": task["family"], "formula": task["formula"],
               "split": f"neural-{split}", "strategy": "training-proof-token-retrieval",
               "proof": proof, "status": "verified" if checked.accepted else "rejected",
               "nodes": 0, "verification": asdict(checked)}
        memory.record(row)
        results.append(row)
    return {"kind": "neural_retrieval_baseline", "split": split,
            "dataset_digest": data["digest"], "tasks": len(results),
            "solved": sum(r["status"] == "verified" for r in results),
            "total_evaluation_seconds": time.monotonic() - start, "results": results}
