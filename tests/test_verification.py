from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from valentyn.engine import Engine
from valentyn.evolution import isolated_search
from valentyn.experiments import corpus
from valentyn.logic import Formula, parse
from valentyn.memory import Memory
from valentyn.policy import Policy
from valentyn.search import Prover, STRATEGIES
from valentyn.verifier import checked_term, LeanVerifier, lean_binary, render

ROOT = Path(__file__).resolve().parents[1]


class BoundaryTests(unittest.TestCase):
    def test_forbidden_terms(self):
        for term in ("sorry", "by exact True.intro", "False.elim (by sorry)",
                     "True.intro\naxiom bad : False", "native_decide", "admit",
                     "True.intro -- hidden", "True.intro /-hidden-/", "#eval 1"):
            with self.subTest(term=term), self.assertRaises(ValueError):
                checked_term(term)

    def test_malicious_formula_tree(self):
        with self.assertRaises(ValueError):
            render(Formula("True\naxiom bad : False"), "True.intro")

    def test_missing_lean_fails_closed(self):
        with patch("valentyn.verifier.lean_binary", return_value=None):
            result = LeanVerifier(ROOT).verify(parse("True"), "True.intro")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "lean_unavailable")

    def test_wrong_toolchain_fails_closed(self):
        with patch("valentyn.verifier._run", return_value=(0, "Lean (version 9.9.9)")):
            result = LeanVerifier(ROOT).verify(parse("True"), "True.intro")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "toolchain_mismatch")

    def test_axiom_audit_is_required(self):
        for output in ("", "'valentyn_candidate' depends on axioms: [sorryAx]",
                       "warning: test\n'valentyn_candidate' does not depend on any axioms"):
            with patch("valentyn.verifier._run", side_effect=[
                (0, "Lean (version 4.24.0)"), (0, output)
            ]):
                result = LeanVerifier(ROOT).verify(parse("True"), "True.intro")
                self.assertFalse(result.accepted)


@unittest.skipUnless(lean_binary(), "Lean not installed; integration checks require scripts/setup.sh")
class LeanIntegrationTests(unittest.TestCase):
    def test_all_inference_constructors_with_real_lean(self):
        sources = ["(p & q) -> (q & p)", "(p -> r) -> (q -> r) -> (p | q) -> r",
                   "p -> !p -> q", "True", "p -> (q | p)",
                   "((p -> q) -> r) -> (q -> r)"]
        for formula in sources:
            with self.subTest(formula=formula):
                result = Prover().solve(parse(formula))
                self.assertIsNotNone(result.proof)
                checked = LeanVerifier(ROOT).verify(parse(formula), result.proof)
                self.assertTrue(checked.accepted, checked.diagnostics)

    def test_type_incorrect_but_lexically_allowed_proof_is_rejected(self):
        result = LeanVerifier(ROOT).verify(parse("p -> q"), "(fun h1 => h1)")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "lean_rejected")

    def test_accepted_proof_is_archived_and_recheckable(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(Path(directory) / "research.sqlite3")
            result = Engine(ROOT, memory).attempt("p -> p", "focused")
            self.assertEqual(result["status"], "verified")
            self.assertTrue(memory.contains("p -> p"))
            self.assertEqual(memory.summary()["verified_proof_artifacts"], 1)

    def test_isolated_worker_proof_is_checked_in_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(Path(directory) / "research.sqlite3")
            engine = Engine(ROOT, memory)
            rows = isolated_search(engine, Policy(), [{"id": "worker", "formula": "p -> p"}])
            result = engine.accept_search(rows[0]["formula"], rows[0], task_id="worker", split="validation")
            self.assertEqual(result["status"], "verified")


if __name__ == "__main__":
    unittest.main()
