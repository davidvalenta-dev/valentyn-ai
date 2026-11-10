"""Compile an axiom audit for every project theorem (not just grep for sorry)."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from valentyn.verifier import lean_binary  # noqa: E402

theorems = []
for path in sorted((root / "Valentyn").glob("*.lean")):
    text = path.read_text()
    if re.search(r"\b(sorry|admit|axiom|unsafe|native_decide)\b", text):
        # Comments count here too, intentionally keeping proof modules simple.
        raise SystemExit(f"Forbidden declaration or proof escape in {path}")
    theorems.extend(re.findall(r"^theorem\s+([a-zA-Z0-9_]+)", text, re.MULTILINE))
with tempfile.TemporaryDirectory(prefix="valentyn-audit-") as directory:
    audit = Path(directory) / "Audit.lean"
    audit.write_text("import Valentyn\n" + "\n".join(
        f"#print axioms Valentyn.{name}" for name in theorems) + "\n")
    binary = lean_binary()
    if binary is None:
        raise SystemExit("Lean unavailable")
    completed = subprocess.run([str(Path(binary).with_name("lake")), "env", "lean",
                                "-DwarningAsError=true", str(audit)],
                               cwd=root, text=True, capture_output=True, timeout=120)
    print(completed.stdout, end="")
    if completed.returncode:
        raise SystemExit(completed.stderr or "Lean axiom audit failed")
    for name in theorems:
        expected = f"'Valentyn.{name}' does not depend on any axioms"
        if expected not in completed.stdout:
            raise SystemExit(f"Unexpected axiom dependencies for {name}")
print(f"Audited {len(theorems)} project theorems: no axiom dependencies.")
