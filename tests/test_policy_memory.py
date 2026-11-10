from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
import unittest

from valentyn.engine import Engine
from valentyn.logic import parse
from valentyn.memory import Memory
from valentyn.policy import Policy, reward


class PolicyTests(unittest.TestCase):
    def test_learning_changes_strategy_selection(self):
        policy = Policy()
        goal = parse("(p | q) -> (q | p)")
        self.assertEqual(policy.choose(goal), "focused")
        for _ in range(100):
            policy.update(goal, "split", 1.0)
        self.assertEqual(policy.choose(goal), "split")

    def test_round_trip_and_mutation_do_not_modify_parent(self):
        policy = Policy()
        before = policy.digest()
        mutated = policy.mutate(random.Random(3))
        self.assertEqual(before, policy.digest())
        self.assertNotEqual(mutated.digest(), before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            mutated.save(path)
            self.assertEqual(mutated.digest(), Policy.load(path).digest())

    def test_invalid_weights_and_schemas(self):
        for value in (float("nan"), float("inf"), 20, True, "eval('evil')"):
            data = Policy().to_dict()
            data["weights"]["focused"][0] = value
            with self.assertRaises(ValueError):
                Policy.from_dict(data)
        data = Policy().to_dict()
        data["code"] = "print('not executed')"
        with self.assertRaises(ValueError):
            Policy.from_dict(data)

    def test_reward_requires_verification(self):
        self.assertEqual(reward(False, 1, 1000), 0)
        self.assertGreater(reward(True, 1, 1000), reward(True, 900, 1000))
        self.assertGreater(reward(True, 1000, 1000), 0)


class MemoryTests(unittest.TestCase):
    def test_rejected_attempt_does_not_enter_theorem_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(Path(directory) / "archive.sqlite3")
            memory.record({"task_id": "false", "split": "test", "formula": "p",
                           "strategy": "focused", "status": "rejected", "nodes": 2,
                           "verification": {"accepted": False}})
            self.assertFalse(memory.contains("p"))
            self.assertEqual(memory.summary()["attempts"], 1)
            self.assertEqual(memory.history()[0]["task_id"], "false")

    def test_status_must_match_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = Memory(Path(directory) / "archive.sqlite3")
            with self.assertRaises(ValueError):
                memory.record({"status": "verified", "verification": None})


if __name__ == "__main__":
    unittest.main()
