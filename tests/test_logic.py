import random
import unittest

from valentyn.experiments import canonical, corpus, propose_formula
from valentyn.logic import Formula, evaluate, parse
from valentyn.search import Prover, STRATEGIES


class LogicTests(unittest.TestCase):
    def test_precedence_and_right_associativity(self):
        self.assertEqual(parse("p -> q -> r"), parse("p -> (q -> r)"))
        self.assertEqual(parse("p | q & r"), parse("p | (q & r)"))
        self.assertEqual(parse("!p"), parse("p -> False"))

    def test_reject_lean_injection_and_malformed_inputs(self):
        for source in ("", "p q", "(p", "p)", "p ->", "p; axiom evil : False",
                       "sorry", "p\n#eval 1", "p /- comment -/", "Prop", "h1"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse(source)

    def test_bounded_input(self):
        for source in ("p" * 4097, "(" * 70 + "p" + ")" * 70):
            with self.assertRaises(ValueError):
                parse(source)

    def test_formula_round_trip(self):
        rng = random.Random(42)
        for _ in range(100):
            goal = propose_formula(rng, 3)
            source = goal.lean().replace("∧", "&").replace("∨", "|").replace("→", "->")
            self.assertEqual(parse(source), goal)

    def test_canonical_renaming_and_split_integrity(self):
        self.assertEqual(canonical(parse("p -> q -> p")), canonical(parse("r -> s -> r")))
        self.assertNotEqual(canonical(parse("p -> p")), canonical(parse("p -> q")))
        self.assertEqual(len(corpus()["test"]), 14)

    def test_all_strategies_construct_nontrivial_proofs(self):
        for strategy in STRATEGIES:
            result = Prover().solve(parse("(p -> r) -> (q -> r) -> (p | q) -> r"), strategy)
            self.assertEqual(result.status, "candidate")
            self.assertIn("Or.elim", result.proof)

    def test_counterexample_is_actually_false(self):
        goal = parse("(p -> q) -> (q -> p)")
        result = Prover().solve(goal)
        self.assertEqual(result.status, "counterexample")
        self.assertFalse(evaluate(goal, result.counterexample))

    def test_classical_validity_is_not_a_proof(self):
        result = Prover(max_nodes=1000).solve(parse("p | !p"))
        self.assertEqual(result.status, "exhausted")
        self.assertIsNone(result.proof)

    def test_budget_exhaustion_is_not_disproof(self):
        result = Prover(max_nodes=1).solve(parse("(p & q) -> (q & p)"))
        self.assertEqual(result.status, "exhausted")
        self.assertLessEqual(result.nodes, 1)
        self.assertIsNone(result.counterexample)

    def test_search_is_deterministic_under_node_budget(self):
        goal = parse("(p & (q | r)) -> ((p & q) | (p & r))")
        a, b = Prover().solve(goal), Prover().solve(goal)
        self.assertEqual((a.proof, a.nodes), (b.proof, b.nodes))

    def test_invalid_search_configuration(self):
        for kwargs in ({"max_nodes": 0}, {"max_depth": 65}, {"timeout": 0}):
            with self.assertRaises(ValueError):
                Prover(**kwargs)


if __name__ == "__main__":
    unittest.main()
