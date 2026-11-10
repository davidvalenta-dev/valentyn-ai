"""Command-line interface for local, reproducible Valentyn research."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import sys

from . import __version__
from .engine import Engine
from .evolution import evolve
from .experiments import discover, evaluate, train
from .logic import parse
from .memory import Memory
from .policy import Policy
from .search import STRATEGIES
from .verifier import LeanVerifier, lean_binary


def project_root(explicit: str | None) -> Path:
    if explicit:
        result = Path(explicit).resolve()
    else:
        result = next((p for p in (Path.cwd(), *Path.cwd().parents)
                       if (p / "lean-toolchain").is_file()), Path.cwd())
    if not (result / "lean-toolchain").is_file():
        raise ValueError("Run inside the Valentyn repository, or supply --project PATH")
    return result


def provenance(project: Path) -> dict:
    digest = hashlib.sha256()
    paths = sorted((project / "src/valentyn").rglob("*.py"))
    paths += sorted((project / "src/valentyn/data").glob("*.json"))
    paths += [project / "lean-toolchain", project / "lakefile.toml", project / "Valentyn.lean",
              project / "pyproject.toml"]
    paths += sorted((project / "Valentyn").glob("*.lean"))
    paths += sorted((project / "scripts").glob("*.py"))
    paths += sorted((project / "scripts").glob("*.sh"))
    for path in paths:
        digest.update(str(path.relative_to(project)).encode())
        digest.update(path.read_bytes())
    return {"valentyn_version": __version__, "python": platform.python_version(),
            "platform": platform.platform(), "implementation_digest": digest.hexdigest(),
            "lean_toolchain": (project / "lean-toolchain").read_text().strip()}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="valentyn", description="Lean-verified reasoning experiments")
    result.add_argument("--project", help="Repository root (defaults to current repository)")
    result.add_argument("--state", help="State directory (default: PROJECT/.valentyn)")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check pinned Lean and verify a smoke-test theorem")
    proof = sub.add_parser("prove", help="Search for and verify a propositional proof")
    proof.add_argument("formula", help="Use p..z, True, False, &, |, ->, ! and parentheses")
    proof.add_argument("--strategy", choices=("auto", *STRATEGIES), default="auto")
    proof.add_argument("--output", type=Path, help="Write a verified .lean artifact")
    for name in ("train", "benchmark", "evolve", "discover", "prove"):
        command = proof if name == "prove" else sub.add_parser(name)
        command.add_argument("--nodes", type=int, default=10000)
        command.add_argument("--depth", type=int, default=24)
        command.add_argument("--timeout", type=float, default=5.0, help="Search time per attempt in seconds")
        command.add_argument("--policy", type=Path, help="Policy file (default: STATE/policy.json)")
        if name != "prove":
            command.add_argument("--report", type=Path, help="Write complete JSON report and proof artifacts")
        if name in ("train", "evolve", "discover"):
            command.add_argument("--seed", type=int, default=0)
        if name == "train":
            command.add_argument("--epochs", type=int, default=100)
        if name == "benchmark":
            command.add_argument("--split", choices=("validation", "test"), default="test")
        if name == "evolve":
            command.add_argument("--generations", type=int, default=8)
            command.add_argument("--output-policy", type=Path,
                                 help="Default: STATE/evolved-policy.json; original policy is preserved")
        if name == "discover":
            command.add_argument("--count", type=int, default=20)
    history = sub.add_parser("history")
    history.add_argument("--limit", type=int, default=10)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        project = project_root(args.project)
        state = Path(args.state).resolve() if args.state else project / ".valentyn"
        if args.command == "doctor":
            checked = LeanVerifier(project).verify(parse("p -> p"), "(fun h1 => h1)")
            print(json.dumps({"lean_binary": lean_binary(), "smoke_test": asdict(checked),
                              "provenance": provenance(project)}, indent=2))
            return 0 if checked.accepted else 1
        memory = Memory(state / "research.sqlite3")
        if args.command == "history":
            rows = memory.history(args.limit)
            print(json.dumps({"summary": memory.summary(), "recent": [
                {k: r[k] for k in ("id", "task_id", "split", "status", "strategy", "nodes")}
                for r in rows]}, indent=2))
            return 0
        engine = Engine(project, memory, nodes=args.nodes, depth=args.depth, timeout=args.timeout)
        policy_path = args.policy or state / "policy.json"
        if args.command == "train":
            policy, report = train(engine, epochs=args.epochs, seed=args.seed)
            policy.save(policy_path)
            report["policy_path"] = str(policy_path)
        else:
            if policy_path.exists():
                policy = Policy.load(policy_path)
            elif args.policy or args.command in ("benchmark", "evolve"):
                raise ValueError(f"Policy not found: {policy_path}. Run 'valentyn train' first.")
            else:
                policy = Policy()
            if args.command == "prove":
                goal = parse(args.formula)
                strategy = policy.choose(goal) if args.strategy == "auto" else args.strategy
                report = engine.attempt(args.formula, strategy)
                if args.output and report["status"] == "verified":
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(report["verification"]["source"])
                print(json.dumps(report, indent=2))
                return {"verified": 0, "counterexample": 2, "exhausted": 3, "rejected": 1}[report["status"]]
            if args.command == "benchmark":
                report = evaluate(engine, policy, split=args.split)
            elif args.command == "evolve":
                evolved, report = evolve(engine, policy, generations=args.generations, seed=args.seed)
                destination = args.output_policy or state / "evolved-policy.json"
                evolved.save(destination)
                report["policy_path"] = str(destination)
            elif args.command == "discover":
                report = discover(engine, policy, count=args.count, seed=args.seed)
        report["provenance"] = provenance(project)
        destination = args.report or state / f"{args.command}-report.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        summary = {k: v for k, v in report.items() if k not in
                   ("results", "observations", "lineage", "initial_attempt_ids")}
        print(json.dumps({**summary, "report": str(destination)}, indent=2))
        if args.command == "benchmark" and any(s["rejected"] for s in report["summary"].values()):
            return 1
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Valentyn: {error}", file=sys.stderr)
        return 1
