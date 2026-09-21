"""Build one response-level DPO pair per input from archived candidate groups."""
import argparse
import hashlib
import json
from pathlib import Path

from preference_rules import discovery_ok, validate_pair
from prepare_data import canonical_code


def response_valid(raw):
    try:
        value = json.loads(raw)
        if set(value) != {'actions','short_comment'} or not value['short_comment'].strip():
            return False
        if not isinstance(value['actions'],list) or not value['actions']:
            return False
        for action in value['actions']:
            if action['type'] not in ('REMOVE','SIMPLIFY','ADD_OR_REPLACE'):
                return False
            field = 'proposal' if action['type']=='ADD_OR_REPLACE' else 'target'
            if not all(isinstance(action.get(k),str) and action[k].strip() for k in (field,'rationale')):
                return False
        return True
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def build(records, excluded_hashes, excluded_tasks, token_count, cutoff):
    accepted, audits = {}, []
    for record in records:
        uid, task = record['id'], record['task']
        code_hash = canonical_code(record['candidate_code'])
        if code_hash in excluded_hashes or task in excluded_tasks:
            audits.append(dict(id=uid,reason='excluded_input_or_task'))
            continue
        messages = record['messages']
        if [m['role'] for m in messages] != ['system','user']:
            raise ValueError('Expected a shared system/user prompt')
        prompt = json.loads(messages[1]['content'])
        if canonical_code(prompt['candidate_code']) != code_hash:
            raise ValueError('Candidate code does not match shared critic input')
        groups = record['groups']
        best = None
        for i,a in enumerate(groups):
            for j,b in enumerate(groups):
                if i==j or not response_valid(a['response']) or not response_valid(b['response']):
                    continue
                if json.loads(a['response'])['actions']==json.loads(b['response'])['actions']:
                    continue
                if len(a['discovery'])!=2 or len(b['discovery'])!=2:
                    raise ValueError('Each group needs two discovery outcomes')
                if not discovery_ok(a['discovery'],b['discovery']):
                    continue
                seed = int.from_bytes(hashlib.sha256(f'{uid}:{i}:{j}'.encode()).digest()[:4],'big')
                meta = validate_pair(a['validation'],b['validation'],seed)
                meta.update(id=uid,task=task,code_hash=code_hash,preferred_group=i,rejected_group=j)
                entry = dict(messages=messages,chosen=dict(role='assistant',content=a['response']),
                             rejected=dict(role='assistant',content=b['response']))
                meta['max_tokens'] = max(token_count(messages+[entry[k]]) for k in ('chosen','rejected'))
                meta['passed'] = bool(meta['passed'] and meta['max_tokens']<=cutoff)
                audits.append(meta)
                if meta['passed'] and (best is None or meta['ci95'][0]>best[0]['ci95'][0]):
                    best=(meta,entry)
        if best and (code_hash not in accepted or best[0]['ci95'][0]>accepted[code_hash][0]['ci95'][0]):
            accepted[code_hash]=best
    selected = sorted(accepted.values(),key=lambda p:p[0]['id'])
    if len({x[0]['id'] for x in selected})!=len(selected):
        raise ValueError('An input ID maps to multiple distinct candidate equations')
    return selected,audits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True,help='Normalized group/outcome JSON list; see README')
    parser.add_argument('--model',required=True,help='Local tokenizer directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exclude-manifests',type=Path,nargs='+',required=True)
    parser.add_argument('--exclude-tasks',nargs='+',required=True,metavar='TASK',
                        help='Development and final evaluation task names to exclude from DPO. Development tasks in exclusion manifests are also excluded automatically.')
    parser.add_argument('--cutoff-len',type=int,default=4096)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    excluded=set()
    tasks=set(args.exclude_tasks)
    for path in args.exclude_manifests:
        for r in json.loads(path.read_text(encoding='utf-8')):
            excluded.add(r['code_hash'])
            if r.get('split')=='dev':tasks.add(r['task'])
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    count=lambda messages:len(tokenizer.apply_chat_template(messages,tokenize=True,enable_thinking=False))
    selected,audits=build(json.loads(args.records.read_text(encoding='utf-8')),excluded,tasks,count,args.cutoff_len)
    if not selected:raise ValueError('No preference pairs passed screening')
    args.output.mkdir(parents=True,exist_ok=True)
    outputs={'critic_preferences.json':[entry for _,entry in selected],
             'manifest.json':[meta for meta,_ in selected],'candidate_audit.json':audits,
             'dataset_info.json':{'critic_preferences':dict(file_name='critic_preferences.json',ranking=True,
                formatting='sharegpt',columns=dict(messages='messages',chosen='chosen',rejected='rejected'),
                tags=dict(role_tag='role',content_tag='content',user_tag='user',assistant_tag='assistant',system_tag='system'))}}
    for name,obj in outputs.items():
        (args.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Accepted {len(selected)} distinct inputs; one pair per input')


if __name__=='__main__':
    main()
