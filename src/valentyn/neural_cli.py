"""Optional model commands. Import PyTorch only when a model command executes."""

import json
from pathlib import Path
import sys


def add_commands(sub) -> None:
    dataset = sub.add_parser("model-data", help="Generate Lean-verified neural training data")
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--train-count", type=int, default=1000)
    dataset.add_argument("--validation-count", type=int, default=80)
    dataset.add_argument("--test-count", type=int, default=100)
    dataset.add_argument("--seed", type=int, default=41)
    train = sub.add_parser("model-train", help="Train Valentyn's neural proof generator")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--kind", choices=("transformer", "gru", "state", "mha", "gqa", "mla", "hybrid"), default="transformer")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--width", type=int, default=96)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--initial", type=Path, help="Explicit parent checkpoint for continued training")
    train.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    train.add_argument("--report", type=Path)
    evaluation = sub.add_parser("model-evaluate", help="Evaluate model-generated proofs without solver fallback")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "test", "train"), default="test")
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--retrieval-baseline", action="store_true")
    evaluation.add_argument("--random-policy-seed", type=int,
                            help="Legality-matched random control for a factorized checkpoint")
    evaluation.add_argument("--nodes", type=int, default=200, help="Factorized proof construction budget")
    evolution = sub.add_parser("model-evolve", help="Test neural decoding improvements on validation only")
    evolution.add_argument("--data", type=Path, required=True)
    evolution.add_argument("--beams", type=int, nargs="+", default=[2, 4])
    evolution.add_argument("--output", type=Path, required=True)
    proof = sub.add_parser("model-prove", help="Ask the neural model to produce a proof, then check it")
    proof.add_argument("formula")
    proof.add_argument("--output", type=Path)
    proof.add_argument("--nodes", type=int, default=200, help="Factorized model construction budget")
    for command in (evaluation, proof, evolution):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
        if command != evolution:
            command.add_argument("--beam", type=int, default=1)
        command.add_argument("--max-tokens", type=int, default=100)
        command.add_argument("--report", type=Path)


def run(args, project: Path, state: Path) -> int:
    from .cli import provenance
    from .proof_data import build_dataset, load_dataset
    if args.command == "model-data":
        data = build_dataset(project, train_count=args.train_count,
                             validation_count=args.validation_count, test_count=args.test_count,
                             seed=args.seed, progress=lambda text: print(text, file=sys.stderr, flush=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({"digest": data["digest"], "splits": {k: len(data[k]) for k in
              ("train", "validation", "test")}, "output": str(args.output)}, indent=2))
        return 0
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install optional model support: python -m pip install -e '.[model]'") from error
    from .memory import Memory
    from .neural import ModelConfig, load_model, select_device, train_model
    from .neural_evaluation import evaluate_model, model_attempt, retrieval_baseline
    from .verifier import LeanVerifier
    memory = Memory(state / "neural.sqlite3")
    if args.command == "model-train":
        if args.kind == "state":
            from .factorized import train_policy
            if args.initial:
                raise ValueError("State policy continuation is not implemented")
            report = train_policy(load_dataset(args.data), args.output, epochs=args.epochs,
                                  width=args.width, batch_size=args.batch_size,
                                  learning_rate=args.learning_rate, seed=args.seed, device_name=args.device,
                                  progress=lambda entry: print(json.dumps(entry), file=sys.stderr, flush=True))
        else:
            report = train_model(load_dataset(args.data), args.output,
                             config=ModelConfig(kind=args.kind, width=args.width),
                             epochs=args.epochs, batch_size=args.batch_size,
                             learning_rate=args.learning_rate, seed=args.seed,
                             device_name=args.device, initial=args.initial,
                             progress=lambda entry: print(json.dumps(entry), file=sys.stderr, flush=True))
    elif args.command == "model-evolve":
        from .neural_evolution import evolve_decoding
        report = evolve_decoding(project, memory, args.checkpoint, load_dataset(args.data),
                                 beams=args.beams, max_tokens=args.max_tokens, device_name=args.device,
                                 progress=lambda text: print(text, file=sys.stderr, flush=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report["selected"], indent=2) + "\n")
    elif args.command == "model-evaluate":
        data = load_dataset(args.data)
        manifest = json.loads((args.checkpoint / "model.json").read_text())
        if manifest.get("architecture") == "factorized-gru-policy" and not args.retrieval_baseline:
            from .factorized_evaluation import evaluate
            report = evaluate(project, memory, args.checkpoint, data, split=args.split,
                              beam=args.beam, nodes=args.nodes, random_seed=args.random_policy_seed,
                              device_name=args.device, limit=args.limit,
                              progress=lambda text: print(text, file=sys.stderr, flush=True))
        else:
            if args.random_policy_seed is not None:
                raise ValueError("Random action control requires a factorized checkpoint")
            report = (retrieval_baseline(project, memory, data, split=args.split)
                  if args.retrieval_baseline else evaluate_model(
                      project, memory, args.checkpoint, data, split=args.split, beam=args.beam,
                      max_tokens=args.max_tokens, device_name=args.device, limit=args.limit,
                      progress=lambda text: print(text, file=sys.stderr, flush=True)))
    else:
        torch.set_num_threads(2)
        manifest = json.loads((args.checkpoint / "model.json").read_text())
        if manifest.get("architecture") == "factorized-gru-policy":
            from .factorized import load_policy
            from .factorized_evaluation import attempt
            model, _ = load_policy(args.checkpoint, select_device(args.device))
            report = attempt(model, args.formula, LeanVerifier(project), memory,
                             beam=args.beam, nodes=args.nodes)
        else:
            model, manifest = load_model(args.checkpoint, select_device(args.device))
            report = model_attempt(model, args.formula, LeanVerifier(project), memory,
                               task_id="interactive", split="interactive",
                               beam=args.beam, max_tokens=args.max_tokens)
        if args.output and report["solved"]:
            accepted = next(r for r in report["attempts"] if r["status"] == "verified")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(accepted["verification"]["source"])
    report["provenance"] = provenance(project)
    destination = args.report or state / f"{args.command}-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in
                     ("results", "history", "attempts", "baseline", "archive")}, indent=2))
    return 0 if args.command != "model-prove" or report["solved"] else 3
