"""A tiny contextual linear strategy learner trained from verified experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random

from .logic import Formula
from .search import STRATEGIES

FEATURES = ("bias", "size", "implications", "conjunctions", "disjunctions", "false", "atoms")


def features(goal: Formula) -> list[float]:
    counts = {"imp": 0, "and": 0, "or": 0, "False": 0}

    def visit(formula: Formula) -> None:
        if formula.op in counts:
            counts[formula.op] += 1
        for arg in formula.args:
            visit(arg)

    visit(goal)
    return [1.0, min(goal.size() / 32, 4), counts["imp"] / 16,
            counts["and"] / 16, counts["or"] / 16, counts["False"] / 16,
            len(goal.atoms()) / 10]


@dataclass
class Policy:
    weights: dict[str, list[float]] = field(
        default_factory=lambda: {s: [0.0] * len(FEATURES) for s in STRATEGIES})
    observations: int = 0
    training_digest: str = ""

    def choose(self, goal: Formula) -> str:
        vector = features(goal)
        return max(STRATEGIES, key=lambda s: sum(a * b for a, b in zip(self.weights[s], vector)))

    def update(self, goal: Formula, strategy: str, reward: float, rate: float = 0.05) -> None:
        if strategy not in STRATEGIES or not 0 <= reward <= 1:
            raise ValueError("Invalid learning observation")
        vector = features(goal)
        weights = self.weights[strategy]
        error = reward - sum(a * b for a, b in zip(weights, vector))
        for index, value in enumerate(vector):
            weights[index] = max(-10.0, min(10.0, weights[index] + rate * error * value))
        self.observations += 1

    def mutate(self, rng: random.Random, scale: float = 0.1) -> Policy:
        result = Policy.from_dict(self.to_dict())
        strategy = rng.choice(STRATEGIES)
        index = rng.randrange(len(FEATURES))
        result.weights[strategy][index] = max(-10.0, min(
            10.0, result.weights[strategy][index] + rng.gauss(0, scale)))
        return result

    def to_dict(self) -> dict:
        return {"schema": 1, "features": list(FEATURES), "weights": self.weights,
                "observations": self.observations, "training_digest": self.training_digest}

    @classmethod
    def from_dict(cls, data: dict) -> Policy:
        if not isinstance(data, dict) or set(data) != {
            "schema", "features", "weights", "observations", "training_digest"
        }:
            raise ValueError("Invalid policy schema")
        if data["schema"] != 1 or data["features"] != list(FEATURES):
            raise ValueError("Unsupported policy features")
        weights = data["weights"]
        if not isinstance(weights, dict) or set(weights) != set(STRATEGIES):
            raise ValueError("Policy must contain every known strategy")
        for vector in weights.values():
            if not isinstance(vector, list) or len(vector) != len(FEATURES):
                raise ValueError("Invalid weight dimension")
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 10 for v in vector):
                raise ValueError("Policy weights must be finite and bounded by 10")
        if type(data["observations"]) is not int or data["observations"] < 0:
            raise ValueError("Invalid observation count")
        if not isinstance(data["training_digest"], str) or len(data["training_digest"]) > 128:
            raise ValueError("Invalid training digest")
        return cls({s: list(v) for s, v in weights.items()}, data["observations"],
                   data["training_digest"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Policy:
        if path.stat().st_size > 16384:
            raise ValueError("Policy exceeds 16 KiB")
        return cls.from_dict(json.loads(path.read_text()))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def reward(verified: bool, nodes: int, budget: int) -> float:
    """Success dominates; within successful attempts fewer nodes are rewarded."""
    return (1.0 - 0.25 * min(nodes / budget, 1.0)) if verified else 0.0
