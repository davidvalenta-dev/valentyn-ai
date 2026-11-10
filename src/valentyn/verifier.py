"""Fail-closed Lean kernel gate. Never interprets model output as Lean commands."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

from .logic import Formula, parse

PINNED_VERSION = "4.24.0"
ALLOWED_IDENTIFIERS = {
    "fun", "True.intro", "False.elim", "And.intro", "And.left", "And.right",
    "Or.inl", "Or.inr", "Or.elim",
}


@dataclass(frozen=True)
class Verification:
    accepted: bool
    reason: str
    source_hash: str
    lean_version: str
    elapsed: float
    diagnostics: str
    source: str


def lean_binary() -> str | None:
    return shutil.which("lean") or (
        str(Path.home() / ".elan/bin/lean")
        if (Path.home() / ".elan/bin/lean").exists() else None
    )


def checked_term(proof: str) -> str:
    """Only the tiny term language emitted by our prover crosses this boundary."""
    if not proof or len(proof) > 100000:
        raise ValueError("Empty or oversized proof term")
    tokens = re.findall(r"=>|[()]|[A-Za-z][A-Za-z0-9_.]*|\S", proof)
    for token in tokens:
        if token not in ALLOWED_IDENTIFIERS and token not in ("(", ")", "=>"):
            if not re.fullmatch(r"h[0-9]+", token):
                raise ValueError(f"Proof term contains forbidden token: {token!r}")
    return " ".join(tokens)


def render(goal: Formula, proof: str) -> str:
    # Round-trip the AST to prevent callers constructing arbitrary Lean source
    # through Formula's public dataclass constructor.
    symbols = goal.lean().replace("∧", "&").replace("∨", "|").replace("→", "->")
    if parse(symbols) != goal:
        raise ValueError("Invalid formula tree")
    binders = " ".join(sorted(goal.atoms()))
    arguments = f" ({binders} : Prop)" if binders else ""
    return ("import Init\nset_option autoImplicit false\n"
            "set_option linter.unusedVariables false\n"
            f"theorem valentyn_candidate{arguments} : {goal.lean()} :=\n"
            f"  {checked_term(proof)}\n"
            "#print axioms valentyn_candidate\n")


def _run(command: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
    environment = {**os.environ, "LEAN_PATH": "", "LEAN_SRC_PATH": ""}
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return -1, "Lean verification timed out"
        output.seek(0)
        return process.returncode, output.read(65536).decode("utf-8", errors="replace")


class LeanVerifier:
    def __init__(self, project: Path, timeout: float = 15):
        self.project = project.resolve()
        if not 0 < timeout <= 300:
            raise ValueError("Verification timeout must be 0..300 seconds, exclusive of 0")
        self.timeout = timeout

    def verify(self, goal: Formula, proof: str) -> Verification:
        start = time.monotonic()
        try:
            source = render(goal, proof)
        except (ValueError, KeyError, RecursionError) as error:
            return Verification(False, "invalid_input", "", "", 0, str(error), "")
        digest = hashlib.sha256(source.encode()).hexdigest()
        binary = lean_binary()
        if not binary:
            return Verification(False, "lean_unavailable", digest, "", 0,
                                "Install Lean with scripts/setup.sh", source)
        try:
            code, version = _run([binary, "--version"], self.project, self.timeout)
            if code != 0 or not re.search(r"\bversion " + re.escape(PINNED_VERSION) + r"\b", version):
                return Verification(False, "toolchain_mismatch", digest, version.strip(),
                                    time.monotonic() - start, version, source)
            with tempfile.TemporaryDirectory(prefix="valentyn-proof-") as directory:
                path = Path(directory) / "Candidate.lean"
                path.write_text(source)
                code, output = _run([binary, "-DwarningAsError=true", "-DmaxRecDepth=1024",
                                     "-DmaxHeartbeats=200000", str(path)],
                                    self.project, self.timeout)
        except OSError as error:
            return Verification(False, "lean_error", digest, "", time.monotonic() - start,
                                str(error), source)
        # Successful compilation alone is insufficient: reject all axioms.
        accepted = (code == 0 and
                    "'valentyn_candidate' does not depend on any axioms" in output and
                    not re.search(r"\b(?:warning|error):", output))
        return Verification(accepted, "verified" if accepted else "lean_rejected",
                            digest, version.strip(), time.monotonic() - start, output, source)
