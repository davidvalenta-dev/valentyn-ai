"""The candidate → verify → archive loop, shared by all commands."""

from dataclasses import asdict
from pathlib import Path

from .logic import parse
from .memory import Memory
from .search import Prover
from .verifier import LeanVerifier


class Engine:
    def __init__(self, project: Path, memory: Memory, *, nodes: int = 10000,
                 depth: int = 24, timeout: float = 5):
        self.prover = Prover(max_nodes=nodes, max_depth=depth, timeout=timeout)
        self.verifier = LeanVerifier(project)
        self.memory = memory

    def attempt(self, formula: str, strategy: str, *, task_id: str = "interactive",
                split: str = "interactive") -> dict:
        goal = parse(formula)
        search = self.prover.solve(goal, strategy)
        return self.accept_search(formula, asdict(search), task_id=task_id, split=split)

    def accept_search(self, formula: str, search: dict, *, task_id: str, split: str) -> dict:
        verification = None
        status = search["status"]
        if search["proof"] is not None:
            checked = self.verifier.verify(parse(formula), search["proof"])
            verification = asdict(checked)
            status = "verified" if checked.accepted else "rejected"
        attempt = {**search, "formula": formula, "task_id": task_id,
                   "split": split, "status": status, "verification": verification}
        attempt["attempt_id"] = self.memory.record(attempt)
        return attempt
