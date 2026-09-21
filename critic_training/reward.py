"""Bounded outcome reward with failure at the worst clipped gain."""
import math

LIMIT=5.0
FAILURE_PENALTY=.1


def utility(result):
    value=result.get('score_delta')
    if result.get('status')!='ok' or not isinstance(value,(int,float)) or not math.isfinite(value):
        return -LIMIT,True
    return max(-LIMIT,min(LIMIT,value)),False


def reward(results):
    if not results:raise ValueError('Empty result group')
    scored=[utility(r) for r in results]
    return .5*max(0,max(x for x,_ in scored))+.5*sum(x for x,_ in scored)/len(scored)-FAILURE_PENALTY*sum(f for _,f in scored)/len(scored)


def old_reward(results):
    scored=[utility(r) for r in results]
    values=[0 if failed else value for value,failed in scored]
    return .5*max(0,max(values))+.5*sum(values)/len(values)-FAILURE_PENALTY*sum(f for _,f in scored)/len(scored)
