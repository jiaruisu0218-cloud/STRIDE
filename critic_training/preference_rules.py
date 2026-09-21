"""Execution-screening rules extracted from the original DPO collection pipeline."""
import numpy as np
from reward import reward, utility

def discovery_ok(a,b):
    return (all(not utility(x)[1] and x['score_delta']>.01 for x in a)
        and all(utility(x)[0]-utility(y)[0]>.05 for x,y in zip(a,b)) and reward(a)>reward(b))

def validate_pair(a,b,seed):
    if len(a) != 8 or len(b) != 8:
        raise ValueError('Preference validation requires eight outcomes per response')
    differences=np.array([utility(x)[0]-utility(y)[0] for x,y in zip(a,b)])
    rng=np.random.default_rng(seed)
    ci=np.quantile(rng.choice(differences,size=(10000,len(a)),replace=True).mean(axis=1),[.025,.975])
    chosen_valid=all(not utility(x)[1] for x in a)
    positives=sum(not utility(x)[1] and x['score_delta']>.01 for x in a)
    margins=int((differences>.05).sum())
    rejected_failures=sum(utility(x)[1] for x in b)
    passed=bool(chosen_valid and positives>=7 and margins>=7 and ci[0]>0 and reward(a)>reward(b)
        and rejected_failures!=1)
    return dict(passed=passed,chosen_all_valid=chosen_valid,positive_seeds=positives,margin_seeds=margins,
        rejected_failures=rejected_failures,mean_margin=float(differences.mean()),ci95=[float(x) for x in ci],
        reward_chosen=reward(a),reward_rejected=reward(b),
        preference_type='repeated_executor_failure' if rejected_failures else 'both_valid')
