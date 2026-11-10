"""Tiny arithmetic/algebra curriculum with an independent Lean ``rfl`` gate.

This is deliberately separate from the propositional proof language.  The
first milestone is closed natural-number evaluation and identities whose
normal form is definitionally equal in Lean; no numerical answer is accepted
without compiling the emitted theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
import subprocess
import tempfile
import time


@dataclass(frozen=True)
class Expr:
    op: str
    left: "Expr | None" = None
    right: "Expr | None" = None
    value: int | None = None

    def lean(self) -> str:
        if self.op == "num":
            return str(self.value)
        if self.op == "var":
            return str(self.value)
        return f"({self.left.lean()} {self.op} {self.right.lean()})"

    def evaluate(self, env=None) -> int:
        env = env or {}
        if self.op == "num":
            return int(self.value)
        if self.op == "var":
            return int(env[self.value])
        left, right = self.left.evaluate(env), self.right.evaluate(env)
        return {"+": lambda: left + right, "*": lambda: left * right,
                "-": lambda: left - right}[self.op]()


def parse_expression(source: str) -> Expr:
    if len(source) > 512:
        raise ValueError("Arithmetic expression is too long")
    parts = re.findall(r"[()]|[+*-]|[A-Za-z][A-Za-z0-9_]*|[0-9]+", source)
    index = 0

    def atom():
        nonlocal index
        if index >= len(parts):
            raise ValueError("Expected arithmetic atom")
        token = parts[index]
        index += 1
        if token == "(":
            result = expression(0)
            if index >= len(parts) or parts[index] != ")":
                raise ValueError("Missing arithmetic closing parenthesis")
            index += 1
            return result
        if token.isdigit():
            value = int(token)
            if value > 1000000:
                raise ValueError("Numeral exceeds curriculum bound")
            return Expr("num", value=value)
        if re.fullmatch(r"[a-z][a-z0-9_]*", token):
            return Expr("var", value=token)
        raise ValueError(f"Invalid arithmetic token {token!r}")

    def product():
        nonlocal index
        left = atom()
        while index < len(parts) and parts[index] == "*":
            index += 1
            left = Expr("*", left, atom())
        return left

    def expression(_level=0):
        nonlocal index
        left = product()
        while index < len(parts) and parts[index] in ("+", "-"):
            op = parts[index]
            index += 1
            left = Expr(op, left, product())
        return left

    result = expression(0)
    if index != len(parts):
        raise ValueError(f"Unexpected arithmetic token {parts[index]!r}")
    return result


def parse_identity(source: str):
    parts = source.split("=")
    if len(parts) != 2:
        raise ValueError("Arithmetic identity must contain one equality")
    return parse_expression(parts[0].strip()), parse_expression(parts[1].strip())


def lean_verify_identity(project: Path, identity: str):
    left, right = parse_identity(identity)
    variables = sorted(left_vars(left) | left_vars(right))
    binders = " ".join(f"({name} : Nat)" for name in variables)
    source = ("import Init\nset_option autoImplicit false\n"
              f"example {binders} : {left.lean()} = {right.lean()} := by rfl\n")
    lean = Path.home() / ".elan/bin/lean"
    if not lean.exists():
        raise RuntimeError("Pinned Lean is unavailable")
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="valentyn-arithmetic-") as directory:
        path = Path(directory) / "Arithmetic.lean"
        path.write_text(source)
        result = subprocess.run([str(lean), "-DwarningAsError=true", str(path)],
                                cwd=project, capture_output=True, text=True, timeout=15)
    return {"identity": identity, "accepted": result.returncode == 0,
            "source": source, "diagnostics": result.stdout + result.stderr,
            "elapsed_seconds": time.monotonic() - start,
            "variables": variables, "reason": "verified_rfl" if result.returncode == 0 else "rejected"}


def left_vars(expr: Expr):
    if expr.op == "var":
        return {expr.value}
    if expr.op == "num":
        return set()
    return left_vars(expr.left) | left_vars(expr.right)


def generate_curriculum():
    return [
        "2 + 3 = 5", "4 * 6 = 24", "(2 + 3) * 4 = 20",
        "n + 0 = n", "(n + 1) + 2 = n + 3",
        "(2 * n) + 0 = 2 * n", "(n * 2) + 0 = n * 2",
    ]
