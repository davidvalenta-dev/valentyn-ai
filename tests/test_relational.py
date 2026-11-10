import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from valentyn.logic import parse
from valentyn.proof_states import ACTIONS, Term, available, with_truth


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional model dependencies absent')
class RelationalTests(unittest.TestCase):
    def test_action_features_ignore_context_order_and_atom_names(self):
        from valentyn.relational import action_features, FEATURES
        context = with_truth(((parse('p -> q'), Term('h0')), (parse('p'), Term('h1'))))
        source = parse('p -> q')
        action = ACTIONS.index(f'apply:{next(i for i,(f,_) in enumerate(context) if f == source)}')
        features = action_features(parse('q'), context, action)
        reverse = tuple(reversed(context))
        other = ACTIONS.index(f'apply:{next(i for i,(f,_) in enumerate(reverse) if f == source)}')
        self.assertEqual(features, action_features(parse('q'), reverse, other))
        renamed = with_truth(((parse('s -> t'), Term('h9')), (parse('s'), Term('h2'))))
        self.assertEqual(features, action_features(parse('t'), renamed, action))
        self.assertEqual(len(features), FEATURES)

    def test_oracle_exhaustion_is_unknown(self):
        from valentyn.cost_supervision import CostOracle
        oracle = CostOracle(max_nodes=0)
        self.assertIsNone(oracle.estimate(parse('p -> p'), ()))
        self.assertIsNone(CostOracle().estimate(parse('p'), ()))
        self.assertEqual(CostOracle().estimate(parse('p -> p'), ()), 2)

    def test_candidate_scorer_has_no_teacher_fallback_and_lean_accepts(self):
        import torch
        from valentyn.factorized import generate_proof
        from valentyn.relational import HeuristicPolicy
        from valentyn.verifier import LeanVerifier
        torch.set_num_threads(2)
        with patch('valentyn.search.Prover.solve', side_effect=AssertionError('Teacher used')), \
             patch('valentyn.cost_supervision.CostOracle.actions', side_effect=AssertionError('Oracle used')):
            result = generate_proof(HeuristicPolicy('structural'), '(p & q) -> (q & p)', beam=1)
        self.assertTrue(result['proofs'])
        check = LeanVerifier(Path.cwd()).verify(parse('(p & q) -> (q & p)'), result['proofs'][0])
        self.assertTrue(check.accepted, check.diagnostics)

    def test_checkpoint_tampering_rejected(self):
        from valentyn.relational import RelationalPolicy, save, load
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save(RelationalPolicy(32), path, {})
            load(path)
            with (path/'weights.pt').open('ab') as output:
                output.write(b'tampered')
            with self.assertRaises(ValueError):
                load(path)

    def test_no_test_read_during_supervision_collection(self):
        from valentyn.cost_supervision import collect
        class FitOnly(dict):
            def __getitem__(self, key):
                if key == 'test':
                    raise AssertionError('Test accessed')
                return super().__getitem__(key)
        task = {'id': 'fit', 'formula': 'p -> p', 'proof': '(fun h0 => h0)'}
        rows = collect(FitOnly(digest='d', train=[task], validation=[task]))
        self.assertFalse(rows['test_accessed'])
        self.assertTrue(rows['train'])

    def test_intermediate_conjunction_can_be_derived_and_checked(self):
        from valentyn.factorized import generate_proof
        from valentyn.relational import HeuristicPolicy
        from valentyn.verifier import LeanVerifier
        formula = '(p -> (q & r)) -> p -> r'
        without = generate_proof(HeuristicPolicy('structural'), formula, beam=4)
        self.assertFalse(without['proofs'])
        with patch('valentyn.search.Prover.solve', side_effect=AssertionError('Teacher used')):
            result = generate_proof(HeuristicPolicy('relevance', forward_actions=True), formula, beam=1)
        self.assertTrue(result['proofs'])
        self.assertTrue(LeanVerifier(Path.cwd()).verify(parse(formula), result['proofs'][0]).accepted)

    def test_forward_action_requires_an_existing_premise(self):
        from valentyn.proof_states import available, action_label
        context = with_truth(((parse('p -> (q & r)'), Term('h0')),))
        self.assertFalse(any(action_label(a).startswith('derive:') for a in available(parse('r'),context,forward_actions=True)))
        context = with_truth((*context, (parse('p'),Term('h1'))))
        self.assertTrue(any(action_label(a).startswith('derive:') for a in available(parse('r'),context,forward_actions=True)))
