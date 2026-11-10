import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional model dependencies absent")
class AttentionTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(11)
        self.torch = torch

    def test_shapes_gradients_and_causality_for_all_variants(self):
        from valentyn.neural import ModelConfig, ProofModel
        for kind in ("mha", "gqa", "mla", "hybrid"):
            with self.subTest(kind=kind):
                model = ProofModel(ModelConfig(kind=kind, width=64, latent_rank=16))
                source = self.torch.randint(1, 64, (2, 12))
                logits = model(source)
                self.assertEqual(tuple(logits.shape), (2, 12, 64))
                self.assertTrue(self.torch.isfinite(logits).all())
                logits.square().mean().backward()
                self.assertTrue(all(p.grad is not None and self.torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                model.eval()
                with self.torch.inference_mode():
                    short, long = model(source[:, :7]), model(source)
                self.assertTrue(self.torch.allclose(short, long[:, :7], atol=2e-5))

    def test_incremental_cache_matches_full_prefix(self):
        from valentyn.neural import ModelConfig, ProofModel
        for kind in ("mha", "gqa", "mla", "hybrid"):
            with self.subTest(kind=kind), self.torch.inference_mode():
                model = ProofModel(ModelConfig(kind=kind, width=64, latent_rank=16)).eval()
                source = self.torch.randint(1, 64, (1, 14))
                full = model(source)
                prefix, cache = model.forward_cached(source[:, :8])
                pieces = [prefix]
                for index in range(8, 14):
                    output, cache = model.forward_cached(source[:, index:index + 1], cache)
                    pieces.append(output)
                self.assertEqual(cache["length"], 14)
                self.assertTrue(self.torch.allclose(full, self.torch.cat(pieces, 1), atol=2e-5))

    def test_grouped_and_latent_cache_are_actually_smaller(self):
        from valentyn.attention import cache_bytes
        from valentyn.neural import ModelConfig, ProofModel
        sizes = {}
        for kind in ("mha", "gqa", "mla"):
            with self.torch.inference_mode():
                model = ProofModel(ModelConfig(kind=kind)).eval()
                _, cache = model.forward_cached(self.torch.ones(1, 32, dtype=self.torch.long))
                sizes[kind] = cache_bytes(cache)
        self.assertLess(sizes["gqa"], sizes["mha"])
        self.assertLess(sizes["mla"], sizes["gqa"])

    def test_cached_generation_matches_uncached(self):
        from valentyn.neural import ModelConfig, ProofModel, generate
        for kind in ("mha", "gqa", "mla", "hybrid"):
            with self.subTest(kind=kind):
                model = ProofModel(ModelConfig(kind=kind, width=64, latent_rank=16))
                a = generate(model, "p -> p", max_tokens=8, use_cache=True)
                b = generate(model, "p -> p", max_tokens=8, use_cache=False)
                self.assertEqual(a["proofs"], b["proofs"])
                self.assertTrue(a["cache_used"])


if __name__ == "__main__":
    unittest.main()
