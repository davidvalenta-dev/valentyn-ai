import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from valentyn.configuration import training_arguments
from valentyn.resources import ResourceBudget
from valentyn.neural_evolution import promotion_gate


class ConfigurationTests(unittest.TestCase):
    def test_bundled_configs_are_data_only(self):
        for path in Path('configs').glob('*.yaml'):
            args = training_arguments(path)
            self.assertTrue(args.data.is_file())
            self.assertGreater(args.epochs, 0)

    def test_rejects_unknown_fields_and_nonfinite_learning_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            for extra in ({'execute': 'anything'}, {'learning_rate': float('nan')}, {'epochs': True}):
                path.write_text(json.dumps({'data': 'a', 'output': 'b', **extra}))
                with self.assertRaises(ValueError):
                    training_arguments(path)

    def test_resource_preflight_rejects_oversized_training(self):
        with patch('valentyn.resources.hardware_snapshot', return_value={}):
            with self.assertRaises(ValueError):
                ResourceBudget().preflight(10**9, 32, 256, 96, 2, 4)

    def test_promotion_rejects_test_selection_and_regressions(self):
        base = dict(dataset_digest='d', weights_sha256='w', split='validation', tasks=10,
                    max_generated_tokens=100, inference_seconds=1, solved=1,
                    results=[{'task_id': str(i), 'solved': i == 0} for i in range(10)])
        candidate = {**base, 'solved': 9, 'results': [
            {'task_id': str(i), 'solved': i != 0} for i in range(10)]}
        self.assertFalse(promotion_gate(base, candidate, 1)['promoted'])
        with self.assertRaises(ValueError):
            promotion_gate({**base, 'split': 'test'}, {**candidate, 'split': 'test'}, 1)
        candidate = {**base, 'solved': 10, 'results': [
            {'task_id': str(i), 'solved': True} for i in range(10)]}
        self.assertTrue(promotion_gate(base, candidate, 1)['promoted'])
