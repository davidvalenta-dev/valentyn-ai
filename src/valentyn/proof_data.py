"""Generate structurally separated, Lean-verified supervision for proof models."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re

from .experiments import canonical
from .logic import Formula, parse
from .search import Prover
from .verifier import LeanVerifier

TEMPLATES = {
    "identity": "A -> A",
    "weaken": "A -> B -> A",
    "swap_pair": "(A & B) -> (B & A)",
    "first": "(A & B) -> A",
    "pair": "A -> B -> (A & B)",
    "compose": "(A -> B) -> (B -> C) -> A -> C",
    "apply": "(A -> B) -> A -> B",
    "inject_left": "A -> (A | B)",
    "inject_right": "B -> (A | B)",
    "swap_or": "(A | B) -> (B | A)",
    "cases": "(A -> C) -> (B -> C) -> (A | B) -> C",
    "map_pair": "(A -> C) -> (B -> D) -> (A & B) -> (C & D)",
    "curry": "((A & B) -> C) -> A -> B -> C",
    "uncurry": "(A -> B -> C) -> (A & B) -> C",
    "distribute": "(A & (B | C)) -> ((A & B) | (A & C))",
    "contrapositive": "(A -> B) -> !B -> !A",
}
HELD_OUT = {
    "long_chain": "(A -> B) -> (B -> C) -> (C -> D) -> A -> D",
    "nested_cases": "(A -> D) -> (B -> D) -> (C -> D) -> (A | (B | C)) -> D",
}


def normalize_proof(proof: str) -> str:
    """Injective alpha-renaming; does not simplify or invent proof steps."""
    names: dict[str, str] = {}
    return re.sub(r"\bh[0-9]+\b", lambda m: names.setdefault(m[0], f"h{len(names)}"), proof)


def ascii_formula(goal: Formula) -> str:
    return goal.lean().replace("∧", "&").replace("∨", "|").replace("→", "->")


def substitute(template: str, rng: random.Random) -> str:
    def atom(depth: int) -> Formula:
        if depth == 0 or rng.random() < 0.7:
            return Formula(rng.choice(("p", "q", "r", "s", "t", "u")))
        return Formula(rng.choice(("and", "or", "imp")), (atom(depth - 1), atom(depth - 1)))
    mapping = {name: ascii_formula(atom(2)) for name in "ABCD"}
    return re.sub(r"[ABCD]", lambda m: f"({mapping[m[0]]})", template)


def dataset_digest(data: dict) -> str:
    return hashlib.sha256(json.dumps({k: data[k] for k in ("train", "validation", "test")},
                                     sort_keys=True).encode()).hexdigest()


def validate_dataset(data: dict) -> None:
    seen = set()
    for split in ("train", "validation", "test"):
        for row in data[split]:
            key = canonical(parse(row["formula"]))
            if key in seen:
                raise ValueError("Dataset contains structurally duplicate theorems")
            seen.add(key)
            if split != "test" and row["family"] in HELD_OUT:
                raise ValueError("Held-out theorem family leaked into fitting data")
            if row.get("proof") is not None and not row["verification"]["accepted"]:
                raise ValueError("Training proof was not verified")
    if data.get("digest") != dataset_digest(data):
        raise ValueError("Dataset digest mismatch")


def build_dataset(project: Path, *, train_count: int = 1000, validation_count: int = 80,
                  test_count: int = 100, seed: int = 41, progress=None) -> dict:
    if not 16 <= train_count <= 20000 or not 4 <= validation_count <= 2000 or not 8 <= test_count <= 2000:
        raise ValueError("Dataset sizes outside supported bounds")
    rng = random.Random(seed)
    prover = Prover(max_nodes=5000, max_depth=24)
    verifier = LeanVerifier(project)
    seen = set()
    data = {"schema": 1, "seed": seed, "generator": "typed-template-substitution-v1",
            "train": [], "validation": [], "test": [],
            "limitations": ["Synthetic elementary propositional theorems, not research mathematics.",
                            "Canonical atom-renaming deduplication is not logical-equivalence deduplication.",
                            "Template substitutions share proof patterns; family holdout is reported separately."]}
    for split, count in (("train", train_count), ("validation", validation_count), ("test", test_count)):
        for trial in range(count * 200):
            index = len(data[split])
            if index == count:
                break
            families = HELD_OUT if split == "test" and index >= count * 3 // 4 else TEMPLATES
            family = list(families)[trial % len(families)]
            formula = substitute(families[family], rng)
            goal = parse(formula)
            key = canonical(goal)
            if key in seen or goal.size() > 100:
                continue
            result = prover.solve(goal, "split")
            if result.proof is None:
                continue
            proof = normalize_proof(result.proof)
            # Bound token context before paying verification cost.
            if len(re.findall(r"->|=>|[A-Za-z][A-Za-z0-9_.]*|\S", ascii_formula(goal) + proof)) > 220:
                continue
            checked = verifier.verify(goal, proof)
            if not checked.accepted:
                raise RuntimeError(f"Teacher proof failed Lean: {checked.diagnostics}")
            seen.add(key)
            data[split].append({"id": f"{split}-{index}", "family": family,
                                "formula": ascii_formula(goal), "proof": proof,
                                "verification": asdict(checked)})
            if progress and (index + 1) % 50 == 0:
                progress(f"{split}: {index + 1}/{count} Lean-verified examples")
        if len(data[split]) != count:
            raise RuntimeError(f"Could only generate {len(data[split])}/{count} {split} examples")
    data["digest"] = dataset_digest(data)
    validate_dataset(data)
    return data


def load_dataset(path: Path) -> dict:
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Dataset exceeds 128 MiB")
    data = json.loads(path.read_text())
    validate_dataset(data)
    return data
