"""Sequential, budget-matched Core architecture pilot and measured cache profiling."""

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def profile(checkpoint: Path, device_name="cpu"):
    import torch
    from .attention import cache_bytes
    from .neural import load_model, memory_measurement, select_device
    torch.set_num_threads(2)
    device = select_device(device_name)
    model, manifest = load_model(checkpoint, device)
    measurements = []

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    with torch.inference_mode():
        for length in (32, 128, 256):
            source = torch.ones(1, length, dtype=torch.long, device=device)
            model.forward_cached(source)
            durations = []
            for _ in range(7):
                sync(); start = time.perf_counter()
                _, cache = model.forward_cached(source)
                sync(); durations.append(time.perf_counter() - start)
            row = {"context": length, "prefill_seconds_median": statistics.median(durations),
                   "retained_cache_bytes": cache_bytes(cache)}
            if length < model.config.context:
                durations = []
                token = torch.ones(1, 1, dtype=torch.long, device=device)
                for _ in range(7):
                    sync(); start = time.perf_counter()
                    model.forward_cached(token, cache)
                    sync(); durations.append(time.perf_counter() - start)
                row["cached_token_seconds_median"] = statistics.median(durations)
            measurements.append(row)
    return {"architecture": manifest["config"], "parameters": manifest["parameters"],
            "parameter_bytes_float32": 4 * manifest["parameters"],
            "adam_parameter_gradient_moments_bytes_estimate": 16 * manifest["parameters"],
            "device": str(device), "measurements": measurements, "memory": memory_measurement(device),
            "note": "Batch one, seven timed runs after warmup. Cache bytes are actual retained tensors; optimizer bytes are an estimate."}


def compare(project: Path, state: Path, data_path: Path, *, kinds=("mha", "gqa", "mla", "hybrid"),
            epochs=12, width=64, batch_size=16, device="cpu", train_count=256,
            validation_count=32, test_count=32, seed=7, progress=None):
    from .proof_data import dataset_digest, load_dataset, validate_dataset
    if not kinds or len(set(kinds)) != len(kinds) or any(k not in ("mha", "gqa", "mla", "hybrid") for k in kinds):
        raise ValueError("Comparison supports distinct mha, gqa, mla, hybrid variants")
    data = load_dataset(data_path)
    if not 8 <= train_count <= len(data["train"]) or not 4 <= validation_count <= len(data["validation"]) or not 8 <= test_count <= len(data["test"]):
        raise ValueError("Comparison subset exceeds corpus")
    held = test_count // 4
    subset = {**data, "train": data["train"][:train_count],
              "validation": data["validation"][:validation_count],
              "test": data["test"][:test_count - held] + data["test"][-held:]}
    subset["digest"] = dataset_digest(subset)
    validate_dataset(subset)
    destination = state / "architecture-comparison"
    destination.mkdir(parents=True, exist_ok=True)
    corpus_path = (destination / "dataset.json").resolve()
    corpus_path.write_text(json.dumps(subset, indent=2) + "\n")
    results = []
    for kind in kinds:
        if progress:
            progress(f"Training {kind} sequentially on {train_count} examples for {epochs} epochs")
        output = (destination / kind).resolve()
        training_report, evaluation_report = output / "training-report.json", output / "evaluation-report.json"
        commands = [
            [sys.executable, "-m", "valentyn", "model-train", "--data", str(corpus_path),
             "--output", str(output), "--kind", kind, "--width", str(width),
             "--epochs", str(epochs), "--batch-size", str(batch_size), "--device", device,
             "--seed", str(seed), "--report", str(training_report)],
            [sys.executable, "-m", "valentyn", "model-evaluate", "--data", str(corpus_path),
             "--checkpoint", str(output), "--beam", "1", "--device", device,
             "--report", str(evaluation_report)],
            [sys.executable, "-m", "valentyn.comparison", "profile", str(output),
             str(output / "profile.json"), device],
        ]
        output.mkdir(parents=True, exist_ok=True)
        for command in commands:
            with (output / "execution.log").open("a") as log:
                completed = subprocess.run(command, cwd=project, stdout=log, stderr=log, timeout=900)
            if completed.returncode:
                raise RuntimeError(f"Architecture {kind} failed; inspect {output / 'execution.log'}")
        training = json.loads(training_report.read_text())
        evaluation = json.loads(evaluation_report.read_text())
        measurements = json.loads((output / "profile.json").read_text())
        results.append({"kind": kind, "training": training, "evaluation": evaluation,
                        "profile": measurements})
        if progress:
            progress(f"{kind}: {evaluation['solved']}/{evaluation['tasks']} Lean-verified model proofs")
    return {"kind": "architecture_comparison", "dataset_digest": subset["digest"],
            "seed": seed, "epochs": epochs, "batch_size": batch_size, "width": width,
            "train_examples": train_count, "validation_examples": validation_count,
            "test_examples": test_count, "device": device, "results": results,
            "limitations": ["Single seed, small synthetic pilot; not frontier superiority evidence.",
                            "Same width/layers/examples/updates; parameter counts and FLOPs differ and are reported.",
                            "Hybrid recurrence is a readable unfused prototype, not an optimized DeltaNet kernel.",
                            "A comparison does not automatically promote a new model."]}


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "profile":
        raise SystemExit("Use: python -m valentyn.comparison profile CHECKPOINT OUTPUT DEVICE")
    Path(sys.argv[3]).write_text(json.dumps(profile(Path(sys.argv[2]), sys.argv[4]), indent=2) + "\n")
