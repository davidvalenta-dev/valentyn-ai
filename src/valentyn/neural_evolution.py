"""Controlled model-driven search improvement, with unchanged model and verifier."""

from pathlib import Path



def promotion_gate(baseline: dict, candidate: dict, comparisons: int) -> dict:
    if any(baseline[key] != candidate[key] for key in
           ("dataset_digest", "weights_sha256", "split", "tasks", "max_generated_tokens")):
        raise ValueError("Incomparable search experiments")
    if baseline["split"] != "validation":
        raise ValueError("Search selection must not use test data")
    before = {row["task_id"]: row["solved"] for row in baseline["results"]}
    after = {row["task_id"]: row["solved"] for row in candidate["results"]}
    if set(before) != set(after) or not 1 <= comparisons <= 16:
        raise ValueError("Invalid paired evaluation")
    wins = sum(after[key] and not before[key] for key in before)
    losses = sum(before[key] and not after[key] for key in before)
    # Exact two-sided sign probability when there are no regressions.
    # Any regression blocks promotion regardless of the aggregate score.
    probability = min(1.0, 2.0 ** (1 - wins)) if losses == 0 else 1.0
    adjusted = min(1.0, comparisons * probability)
    before_cost = baseline["inference_seconds"] / max(baseline["solved"], 1)
    after_cost = candidate["inference_seconds"] / max(candidate["solved"], 1)
    promoted = wins > 0 and losses == 0 and adjusted <= 0.05 and after_cost <= before_cost
    return {"promoted": promoted, "new_solved": wins, "regressions": losses,
            "adjusted_sign_probability": adjusted,
            "baseline_inference_seconds_per_solve": before_cost,
            "candidate_inference_seconds_per_solve": after_cost,
            "criterion": "No lost solves, adjusted paired sign probability <= .05, and no higher measured inference cost per solve.",
            "caveat": "Exploratory: synthetic theorem dependencies and timing noise limit statistical interpretation; no claim of broad significance."}


def evolve_decoding(project, memory, checkpoint: Path, data: dict, *, beams=(2, 4),
                    max_tokens: int = 100, device_name="cpu", progress=None) -> dict:
    from .neural_evaluation import evaluate_model

    if not beams or len(beams) > 4 or any(type(b) is not int or not 2 <= b <= 16 for b in beams):
        raise ValueError("Propose one to four beam widths from 2 through 16")
    baseline = evaluate_model(project, memory, checkpoint, data, split="validation", beam=1,
                              max_tokens=max_tokens, device_name=device_name, progress=progress)
    archive = []
    best_beam, best_solved = 1, baseline["solved"]
    for beam in dict.fromkeys(beams):
        candidate = evaluate_model(project, memory, checkpoint, data, split="validation", beam=beam,
                                   max_tokens=max_tokens, device_name=device_name, progress=progress)
        gate = promotion_gate(baseline, candidate, len(set(beams)))
        archive.append({"beam": beam, "gate": gate, "evaluation": candidate})
        if gate["promoted"] and candidate["solved"] > best_solved:
            best_beam, best_solved = beam, candidate["solved"]
    report = {"kind": "neural_search_evolution", "baseline": baseline, "archive": archive,
              "selected": {"beam": best_beam, "max_tokens": max_tokens,
                           "weights_sha256": baseline["weights_sha256"]},
              "promoted": best_beam != 1,
              "scope": "Model decoding configuration only; no weights, verifier, or executable source are modified."}
    memory.experiment("neural_search_evolution", {"selected": report["selected"], "promoted": report["promoted"]})
    return report
