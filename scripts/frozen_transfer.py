"""Freeze model hashes BEFORE generating a fresh, independently seeded test suite."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re

from valentyn.experiments import canonical
from valentyn.logic import parse
from valentyn.proof_data import ascii_formula, dataset_digest, load_dataset, normalize_proof, substitute
from valentyn.search import Prover
from valentyn.verifier import LeanVerifier

FAMILIES = {
    "transfer_chain": "(A -> B) -> (B -> C) -> (C -> D) -> A -> D",
    "transfer_cases": "(A -> D) -> (B -> D) -> (C -> D) -> (A | (B | C)) -> D",
    "transfer_pair_reverse": "((A & B) & (C & D)) -> ((D & C) & (B & A))",
    "transfer_cases_pair": "(A -> (C & D)) -> (B -> (C & D)) -> (A | B) -> (D & C)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="Also exclude previously examined suites")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    if not 8 <= args.count <= 2000:
        parser.error("Count must be 8..2000")
    base = load_dataset(args.data)
    frozen = {}
    for path in args.checkpoints:
        manifest = json.loads((path / "model.json").read_text())
        checksum = hashlib.sha256((path / "weights.pt").read_bytes()).hexdigest()
        if checksum != manifest["weights_sha256"]:
            raise SystemExit("Invalid model checkpoint")
        frozen[str(path)] = checksum
    seen = {canonical(parse(row["formula"])) for split in ("train", "validation", "test") for row in base[split]}
    excluded = {}
    for path in args.exclude:
        previous = load_dataset(path)
        excluded[str(path)] = previous["digest"]
        seen.update(canonical(parse(row["formula"])) for split in ("train", "validation", "test") for row in previous[split])
    data = {**base, "test": [], "fresh_test_protocol": {
        "seed": args.seed, "frozen_checkpoints": frozen, "excluded_datasets": excluded,
        "base_dataset_digest": base["digest"],
        "purpose": "New structural samples after architecture development. Same synthetic problem domain; not open mathematics."}}
    rng = random.Random(args.seed)
    root = Path(__file__).resolve().parents[1]
    verifier = LeanVerifier(root)
    for trial in range(args.count * 200):
        if len(data["test"]) == args.count:
            break
        family = list(FAMILIES)[len(data["test"]) % len(FAMILIES)]
        goal = parse(substitute(FAMILIES[family], rng))
        key = canonical(goal)
        if key in seen or goal.size() > 100 or len(re.findall(r"->|[A-Za-z]+|\S", ascii_formula(goal))) > 150:
            continue
        result = Prover(max_nodes=5000).solve(goal, "split")
        if result.proof is None:
            continue
        proof = normalize_proof(result.proof)
        verification = verifier.verify(goal, proof)
        if not verification.accepted:
            raise SystemExit(verification.diagnostics)
        seen.add(key)
        data["test"].append({"id": f"fresh-{len(data['test'])}", "family": family,
                             "formula": ascii_formula(goal), "proof": proof,
                             "verification": asdict(verification)})
        if len(data["test"]) % 32 == 0:
            print(f"{len(data['test'])}/{args.count} fresh problems verified", flush=True)
    if len(data["test"]) != args.count:
        raise SystemExit("Could not fill frozen suite")
    for path, checksum in frozen.items():
        if hashlib.sha256((Path(path) / "weights.pt").read_bytes()).hexdigest() != checksum:
            raise SystemExit("Model changed during test construction")
    data["digest"] = dataset_digest(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["fresh_test_protocol"], indent=2))


if __name__ == "__main__":
    main()
