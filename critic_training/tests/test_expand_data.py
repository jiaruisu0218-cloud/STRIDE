"""Exercise exact edit reconstruction and numerical verification boundaries."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expand_data import reconstruct, verified


class ExpansionTests(unittest.TestCase):
    def test_mixed_insert_replace_delete(self):
        before='def equation(x, params):\n a=params[0]*x\n b=params[1]*x**2\n return a+b'
        after='def equation(x, params):\n a=params[0]*x\n c=params[1]*x**3\n return a+c'
        actions, patches=reconstruct(before,after)
        self.assertTrue(actions['actions'])
        self.assertTrue(patches)

    def test_unchanged_ast_rejected(self):
        with self.assertRaisesRegex(ValueError,'empty_or_large'):
            reconstruct('def equation(x, params):\n return x','def equation(x, params):\n # comment\n return x')

    def test_signature_change_rejected(self):
        with self.assertRaisesRegex(ValueError,'signature_changed'):
            reconstruct('def equation(x, params):\n return x','def equation(y, params):\n return y')

    def test_feedback_must_match(self):
        good=dict(status='ok',score=1.,prediction_nmse=.1,prediction_verified_score=1.)
        self.assertTrue(verified(good))
        self.assertFalse(verified(dict(good,prediction_verified_score=2.)))
        self.assertFalse(verified(dict(good,score=float('nan'))))


if __name__=='__main__':unittest.main()
