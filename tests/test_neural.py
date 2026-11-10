"""Optional neural tests; the dependency-free reasoning installation remains usable."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from valentyn.proof_data import normalize_proof

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ProofNormalizationTests(unittest.TestCase):
    def test_alpha_renaming_preserves_repeated_references(self):
        self.assertEqual(normalize_proof("(fun h99 => (fun h7 => (h99 h7)))"),
                         "(fun h0 => (fun h1 => (h0 h1)))")


@unittest.skipUnless(HAS_TORCH, "Install optional model dependencies")
class NeuralTests(unittest.TestCase):
    def setUp(self):
        import torch
        from valentyn.neural import ModelConfig, ProofModel
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.torch = torch
        self.model = ProofModel(ModelConfig(width=32, layers=1))

    def test_causal_attention_does_not_read_future_proof_tokens(self):
        from valentyn.neural import IDS, prompt
        prefix = prompt("p -> p")
        self.model.eval()
        with self.torch.inference_mode():
            before = self.model(self.torch.tensor([prefix]))
            after = self.model(self.torch.tensor([prefix + [IDS["fun"], IDS["h0"]]]))
        self.assertTrue(self.torch.allclose(before, after[:, :len(prefix)], atol=1e-5))

    def test_prompt_loss_is_masked(self):
        from valentyn.neural import encode_example, prompt
        row = {"formula": "p -> p", "proof": "(fun h0 => h0)"}
        source, targets = encode_example(row, 256)
        boundary = len(prompt(row["formula"])) - 1
        self.assertTrue(all(label == -100 for label in targets[:boundary]))
        self.assertTrue(all(label >= 0 for label in targets[boundary:]))
        self.assertEqual(len(source), len(targets))

    def test_checkpoint_roundtrip_and_tamper_detection(self):
        from valentyn.neural import load_model, save_model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original = save_model(self.model, path, {"test": True})
            recovered, manifest = load_model(path, self.torch.device("cpu"))
            self.assertEqual(original["weights_sha256"], manifest["weights_sha256"])
            for a, b in zip(self.model.parameters(), recovered.parameters()):
                self.assertTrue(self.torch.equal(a, b))
            with (path / "weights.pt").open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaises(ValueError):
                load_model(path, self.torch.device("cpu"))

    def test_generator_never_uses_symbolic_solver(self):
        from valentyn.neural import generate
        with patch("valentyn.search.Prover.solve", side_effect=AssertionError("Solver fallback")):
            result = generate(self.model, "p -> p", beam=2, max_tokens=5)
        self.assertLessEqual(result["inference_calls"], 5)

    def test_model_configuration_is_bounded(self):
        from valentyn.neural import ModelConfig, ProofModel
        for options in ({"kind": "eval"}, {"width": 10000}, {"context": 100000}, {"layers": 0}):
            with self.assertRaises(ValueError):
                ProofModel(ModelConfig(**options))

    def test_small_training_run_reduces_validation_loss(self):
        from valentyn.neural import ModelConfig, train_model
        rows = [{"formula": "p -> p", "proof": "(fun h0 => h0)"},
                {"formula": "q -> q", "proof": "(fun h0 => h0)"}]
        # An intentional same-pattern fixture checks optimization, not generalization.
        data = {"train": rows, "validation": rows, "digest": "unit-test-fixture"}
        with tempfile.TemporaryDirectory() as directory:
            report = train_model(data, Path(directory), config=ModelConfig(width=32, layers=1),
                                 epochs=4, batch_size=2, learning_rate=0.01)
        self.assertLess(report["best_validation_loss"], report["initial_validation_loss"])
        self.assertEqual(report["initialization"]["pretrained_weights"], False)


if __name__ == "__main__":
    unittest.main()
