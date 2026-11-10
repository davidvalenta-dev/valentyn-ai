"""Small, explicit propositional language. No Lean source is accepted as input."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Formula:
    op: str
    args: tuple[Formula, ...] = ()

    def lean(self) -> str:
        if not self.args:
            return self.op
        symbol = {"and": "∧", "or": "∨", "imp": "→"}[self.op]
        return f"({self.args[0].lean()} {symbol} {self.args[1].lean()})"

    def atoms(self) -> set[str]:
        if not self.args:
            return set() if self.op in ("True", "False") else {self.op}
        return set().union(*(arg.atoms() for arg in self.args))

    def size(self) -> int:
        return 1 + sum(arg.size() for arg in self.args)


def parse(text: str) -> Formula:
    """Parse ASCII &, |, ->, !, parentheses; implication associates right."""
    if len(text) > 4096:
        raise ValueError("Formula exceeds 4096 characters")
    tokens = re.findall(r"->|[()&|!]|[A-Za-z][A-Za-z0-9_]*|\S", text)
    index = 0

    def expression(level: int = 0, depth: int = 0) -> Formula:
        nonlocal index
        if depth > 64:
            raise ValueError("Formula nesting exceeds 64")
        if level < 3:
            left = expression(level + 1, depth + 1)
            operator = ("->", "|", "&")[level]
            while index < len(tokens) and tokens[index] == operator:
                index += 1
                right = expression(level if level == 0 else level + 1, depth + 1)
                left = Formula(("imp", "or", "and")[level], (left, right))
            return left
        if index >= len(tokens):
            raise ValueError("Expected a proposition")
        token = tokens[index]
        index += 1
        if token == "!":
            return Formula("imp", (expression(3, depth + 1), Formula("False")))
        if token == "(":
            result = expression(0, depth + 1)
            if index >= len(tokens) or tokens[index] != ")":
                raise ValueError("Missing closing parenthesis")
            index += 1
            return result
        if token in ("True", "False") or re.fullmatch(r"[p-z][0-9]*", token):
            return Formula(token)
        raise ValueError(f"Invalid proposition {token!r}; use p through z, optionally numbered")

    result = expression()
    if index != len(tokens):
        raise ValueError(f"Unexpected token {tokens[index]!r}")
    if result.size() > 128:
        raise ValueError("Formula exceeds 128 syntax nodes")
    return result


def evaluate(formula: Formula, assignment: dict[str, bool]) -> bool:
    """Classical truth evaluation, used only for counterexamples, never proofs."""
    if formula.op == "True":
        return True
    if formula.op == "False":
        return False
    if not formula.args:
        return assignment[formula.op]
    a, b = (evaluate(arg, assignment) for arg in formula.args)
    return {"and": lambda: a and b, "or": lambda: a or b,
            "imp": lambda: not a or b}[formula.op]()
