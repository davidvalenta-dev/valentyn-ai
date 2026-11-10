from pathlib import Path
import unittest

from valentyn.arithmetic import generate_curriculum, lean_verify_identity, parse_expression, parse_identity


class ArithmeticTests(unittest.TestCase):
    def test_parser_and_evaluator(self):
        expression = parse_expression("(2 + 3) * 4")
        self.assertEqual(expression.evaluate(), 20)
        self.assertEqual(parse_expression("n + 0").evaluate({"n": 9}), 9)

    def test_identity_parser(self):
        left, right = parse_identity("(n + 0) * 1 = n")
        self.assertEqual(left.evaluate({"n": 11}), right.evaluate({"n": 11}))

    def test_curriculum_is_lean_checked(self):
        results = [lean_verify_identity(Path.cwd(), identity) for identity in generate_curriculum()]
        self.assertTrue(all(result["accepted"] for result in results), results)

    def test_rejects_non_identity(self):
        result = lean_verify_identity(Path.cwd(), "2 + 2 = 5")
        self.assertFalse(result["accepted"])
