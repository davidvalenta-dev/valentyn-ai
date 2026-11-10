from pathlib import Path
import unittest
from unittest.mock import patch

from valentyn.logic import parse
from valentyn.verifier import Verification, render
from valentyn.verification_cache import VerificationCache


class VerificationCacheTests(unittest.TestCase):
    def test_only_identical_previously_accepted_source_is_reused(self):
        cache = VerificationCache(Path.cwd())
        goal = parse('p -> p')
        proof = '(fun h0 => h0)'
        result = Verification(True, 'verified', 'hash', '4.24.0', .3, 'checked', render(goal, proof))
        with patch.object(cache, '_identity', return_value=('stable',)), \
             patch.object(cache.verifier, 'verify', return_value=result) as backend:
            self.assertEqual(cache.verify(goal, proof).reason, 'verified')
            hit = cache.verify(goal, proof)
            self.assertEqual(hit.reason, 'verified_in_process_cache')
            self.assertEqual(hit.elapsed, 0)
            self.assertEqual(backend.call_count, 1)
            cache.verify(goal, '(fun h1 => h1)')
            self.assertEqual(backend.call_count, 2)
        with patch.object(cache, '_identity', return_value=('changed',)), \
             patch.object(cache.verifier, 'verify', return_value=result) as backend:
            cache.verify(goal, proof)
            self.assertEqual(backend.call_count, 1)

    def test_failed_verification_is_not_cached(self):
        cache = VerificationCache(Path.cwd())
        result = Verification(False, 'lean_rejected', '', '', 0, '', '')
        with patch.object(cache, '_identity', return_value=('stable',)), \
             patch.object(cache.verifier, 'verify', return_value=result) as backend:
            for _ in range(2):
                self.assertFalse(cache.verify(parse('p -> q'), '(fun h0 => h0)').accepted)
            self.assertEqual(backend.call_count, 2)
            self.assertEqual(cache.hits, 0)
