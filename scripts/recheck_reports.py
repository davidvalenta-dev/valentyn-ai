"""Independently rerun Lean on every distinct accepted proof in saved reports."""

import argparse
import hashlib
import json
from pathlib import Path

from valentyn.cli import provenance
from valentyn.logic import parse
from valentyn.verifier import LeanVerifier, render

ROOT = Path(__file__).resolve().parents[1]


def accepted_attempts(value):
    if isinstance(value, dict):
        if value.get("status") == "verified" and "verification" in value:
            yield value
        for child in value.values():
            yield from accepted_attempts(child)
    elif isinstance(value, list):
        for child in value:
            yield from accepted_attempts(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()
    verifier = LeanVerifier(ROOT)
    seen = set()
    occurrences = 0
    for path in sorted(args.reports.glob("*.json")):
        for attempt in accepted_attempts(json.loads(path.read_text())):
            occurrences += 1
            archive = attempt["verification"]
            source = render(parse(attempt["formula"]), attempt["proof"])
            digest = hashlib.sha256(source.encode()).hexdigest()
            if not archive["accepted"] or archive["source_hash"] != digest or archive["source"] != source:
                raise SystemExit(f"Proof provenance mismatch in {path}: {attempt['task_id']}")
            if digest in seen:
                continue
            checked = verifier.verify(parse(attempt["formula"]), attempt["proof"])
            if not checked.accepted:
                raise SystemExit(f"Lean recheck failed: {path}: {checked.diagnostics}")
            seen.add(digest)
            if args.export_dir:
                args.export_dir.mkdir(parents=True, exist_ok=True)
                (args.export_dir / f"Proof_{digest[:16]}.lean").write_text(source)
    if not seen:
        raise SystemExit("No accepted proofs found; refusing an empty audit")
    report = {"kind": "proof_recheck", "accepted_occurrences": occurrences,
              "distinct_proofs_rechecked": len(seen), "failures": 0,
              "source_hashes": sorted(seen), "provenance": provenance(ROOT)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
