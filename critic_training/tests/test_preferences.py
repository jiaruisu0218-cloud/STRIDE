import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_preferences import build
from preference_rules import validate_pair
from prepare_data import canonical_code
from reward import reward


def ok(delta):
    return dict(status='ok',score_delta=delta)


class PreferenceTests(unittest.TestCase):
    def test_all_failure_has_negative_reward(self):
        self.assertEqual(reward([dict(status='invalid')]*8),-2.6)

    def test_valid_pair_and_repeated_failure(self):
        self.assertTrue(validate_pair([ok(.4)]*8,[ok(-.1)]*8,42)['passed'])
        self.assertTrue(validate_pair([ok(.4)]*8,[dict(status='invalid')]*8,42)['passed'])

    def test_single_failure_and_outlier_rejected(self):
        self.assertFalse(validate_pair([ok(.2)]*8,[ok(0)]*7+[dict(status='invalid')],42)['passed'])
        self.assertFalse(validate_pair([ok(.1)]*7+[ok(-3)], [ok(0)]*8,42)['passed'])
        with self.assertRaises(ValueError):validate_pair([ok(.2)]*7,[ok(0)]*7,42)

    def test_exclusions_deduplication_and_context_limit(self):
        code='def equation(x, params):\n return params[0]*x'
        def group(target,delta):
            response=json.dumps(dict(actions=[dict(type='SIMPLIFY',target=target,rationale='Simplify the expression.')],
                                     short_comment='Refit the edited equation.'))
            return dict(response=response,discovery=[ok(delta)]*2,validation=[ok(delta)]*8)
        row=dict(id='example/1',task='example',candidate_code=code,
                 messages=[dict(role='system',content='Propose edits.'),
                           dict(role='user',content=json.dumps(dict(candidate_code=code)))],
                 groups=[group('x',.4),group('params[0]*x',-.1)])
        count=lambda messages:100
        accepted,_=build([row,row],set(),set(),count,4096)
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0][1]['chosen']['content'],row['groups'][0]['response'])
        self.assertFalse(build([row],{canonical_code(code)},set(),count,4096)[0])
        self.assertFalse(build([row],set(),{'example'},count,4096)[0])
        self.assertFalse(build([row],set(),set(),count,50)[0])


if __name__=='__main__':unittest.main()
