import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from valentyn.logic import parse
from valentyn.proof_states import ACTIONS, available, extract_steps, parse_term
from valentyn.search import Prover
from valentyn.verifier import LeanVerifier, lean_binary


class StateExtractionTests(unittest.TestCase):
    def test_term_parser_roundtrip(self):
        for source in ("(fun h0 => (h0 (fun h1 => h1)))",
                       "(Or.elim h0 (fun h1 => h1) (fun h2 => h2))"):
            term = parse_term(source)
            self.assertEqual(parse_term(term.lean()), term)

    def test_extract_inference_steps_from_verified_language(self):
        formulas = ["p -> p", "(p & q) -> (q & p)",
                    "(p -> r) -> (q -> r) -> (p | q) -> r",
                    "(p -> q -> r) -> (p & q) -> r", "p -> !p -> q"]
        for source in formulas:
            proof = Prover().solve(parse(source), "split").proof
            self.assertIsNotNone(proof)
            steps = extract_steps(parse(source), proof)
            self.assertGreater(len(steps), 0)
            for step in steps:
                self.assertIn(step["action"], available(step["goal"], step["context"]))

    def test_invalid_proof_term_is_not_silently_traced(self):
        with self.assertRaises(ValueError):
            extract_steps(parse("p"), "unknown")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional model dependencies absent")
class FactorizedModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        from valentyn.factorized import StatePolicy
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = StatePolicy(32)

    def test_random_control_does_not_use_model_or_solver(self):
        from valentyn.factorized import generate_proof
        with patch.object(self.model, "forward", side_effect=AssertionError("Model used")), \
             patch("valentyn.search.Prover.solve", side_effect=AssertionError("Solver used")):
            result = generate_proof(self.model, "p -> p", random_seed=0)
        self.assertEqual(result["inference_calls"], 0)
        self.assertEqual(len(result["proofs"]), 1)

    def test_model_generation_has_no_solver_fallback(self):
        from valentyn.factorized import generate_proof
        with patch("valentyn.search.Prover.solve", side_effect=AssertionError("Solver used")):
            result = generate_proof(self.model, "p -> p", beam=1)
        self.assertEqual(len(result["proofs"]), 1)
        self.assertGreater(result["inference_calls"], 0)

    def test_false_theorem_has_no_closed_construction(self):
        from valentyn.factorized import generate_proof
        result = generate_proof(self.model, "p -> q", beam=4, max_nodes=50)
        self.assertFalse(result["proofs"])

    def test_budget_is_enforced(self):
        from valentyn.factorized import generate_proof
        result = generate_proof(self.model, "p -> p", max_nodes=1)
        self.assertEqual(result["nodes"], 1)
        self.assertFalse(result["proofs"])

    @unittest.skipUnless(lean_binary(), "Lean unavailable")
    def test_constructed_terms_pass_real_lean(self):
        from valentyn.factorized import generate_proof
        verifier = LeanVerifier(Path(__file__).resolve().parents[1])
        for formula in ("(p & q) -> (q & p)", "(p | q) -> (q | p)",
                        "(p -> r) -> (q -> r) -> (p | q) -> r", "p -> !p -> q"):
            result = generate_proof(self.model, formula, beam=4, max_nodes=500, random_seed=0)
            self.assertTrue(result["proofs"], formula)
            for term in result["proofs"]:
                checked = verifier.verify(parse(formula), term)
                self.assertTrue(checked.accepted, checked.diagnostics)


if __name__ == "__main__":
    unittest.main()
