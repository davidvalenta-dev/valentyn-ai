"""Evaluate forward-action checkpoints on the already frozen transfer suite."""
import json
from pathlib import Path

from valentyn.factorized_evaluation import evaluate
from valentyn.memory import Memory
from valentyn.proof_data import load_dataset
from valentyn.verification_cache import VerificationCache


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / ".valentyn/relational-forward"
    data = load_dataset(root / ".valentyn/relational/frozen-transfer-v2.json")
    memory = Memory(base / "frozen.sqlite3")
    verifier = VerificationCache(root)
    for seed in (7, 11, 23):
        checkpoint = base / f"cost-seed{seed}"
        for beam in (1, 4):
            report = evaluate(root, memory, checkpoint, data, beam=beam,
                              verifier=verifier)
            destination = base / f"frozen-cost-{seed}-beam{beam}.json"
            destination.write_text(json.dumps(report, indent=2) + "\n")
            print(seed, beam, report["solved"], report["nodes"], flush=True)


if __name__ == "__main__":
    main()
