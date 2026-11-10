"""Strict data-only training configurations, using JSON-formatted YAML 1.2."""

import argparse
import json
import math
from pathlib import Path


def training_arguments(path: Path):
    if path.stat().st_size > 16384:
        raise ValueError("Training configuration exceeds 16 KiB")
    try:
        values = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("Use JSON-formatted YAML (as in the bundled configs); arbitrary YAML tags are not supported") from error
    allowed = {"data", "output", "kind", "epochs", "batch_size", "width", "seed",
               "learning_rate", "device", "report"}
    if not isinstance(values, dict) or set(values) - allowed or not {"data", "output"} <= set(values):
        raise ValueError("Unknown or missing training configuration fields")
    defaults = {"kind": "transformer", "epochs": 15, "batch_size": 32, "width": 96,
                "seed": 7, "learning_rate": 0.001, "device": "auto", "report": None}
    for key in ("data", "output", "report"):
        if values.get(key) is not None and (not isinstance(values[key], str) or not values[key]):
            raise ValueError(f"Configuration field {key} must be a path string")
    args = {**defaults, **values}
    for key in ("epochs", "batch_size", "width", "seed"):
        if type(args[key]) is not int:
            raise ValueError(f"Configuration field {key} must be an integer")
    if args["kind"] not in ("transformer", "gru", "state", "relational", "relational-cost", "mha", "gqa", "mla", "hybrid"):
        raise ValueError("Unsupported training architecture")
    if args["device"] not in ("cpu", "mps", "auto"):
        raise ValueError("Unsupported training device")
    if type(args["learning_rate"]) not in (int, float) or not math.isfinite(args["learning_rate"]):
        raise ValueError("Learning rate must be a finite number")
    for key in ("data", "output", "report"):
        args[key] = Path(args[key]) if args[key] else None
    return argparse.Namespace(command="model-train", initial=None, **args)
