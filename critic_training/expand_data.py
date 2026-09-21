"""Add auditable edit targets reconstructed from successful archived revisions."""
import argparse
import ast
from collections import Counter
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil

from prepare_data import SYSTEM, canonical_code, make_input, write_json


def function_statements(code):
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('not_single_function')
    fn = tree.body[0]
    if fn.decorator_list:
        raise ValueError('decorated_function')
    statements = [s for s in fn.body if not (isinstance(s, ast.Expr) and
                  isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
    if any(not isinstance(s, (ast.Assign, ast.AnnAssign, ast.Return)) for s in statements):
        raise ValueError('complex_control_flow')
    return fn, [ast.unparse(s) for s in statements]


def reconstruct(base_code, revised_code):
    original, before = function_statements(base_code)
    revised, after = function_statements(revised_code)
    if original.name != revised.name or ast.dump(original.args) != ast.dump(revised.args):
        raise ValueError('signature_changed')
    opcodes = difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
    patches, actions = [], []
    for tag, i, j, k, l in opcodes:
        if tag == 'equal':
            continue
        old, new = '\n'.join(before[i:j]), '\n'.join(after[k:l])
        patches.append(dict(operation=tag, start=i, end=j, old=before[i:j], new=after[k:l]))
        if tag == 'delete':
            actions.append(dict(type='REMOVE', target=old,
                rationale='Remove this block as part of the coordinated revision; refit the remaining parameters.'))
        else:
            if tag == 'replace':
                proposal=f'Replace this block:\n{old}\nWith:\n{new}'
            elif i < len(before):
                proposal=f'Insert before this statement:\n{before[i]}\nNew block:\n{new}'
            else:
                proposal=f'Append this block to the function body:\n{new}'
            actions.append(dict(type='ADD_OR_REPLACE', proposal=proposal,
                rationale='Apply this exact block edit together with the other listed edits, preserving their order and dependencies.'))
    if not 1 <= len(actions) <= 8:
        raise ValueError('empty_or_large_patch_group')
    # Reconstruct mechanically to ensure every edited statement is accounted for.
    applied = list(before)
    for patch in reversed(patches):
        applied[patch['start']:patch['end']] = patch['new']
    assert applied == after
    return dict(actions=actions, short_comment='Apply the edits jointly, then refit all adjustable parameters on the full training data and compare the original weighted score.'), patches


def verified(feedback):
    if feedback.get('status') != 'ok':
        return False
    values = [feedback.get(k) for k in ('score', 'prediction_nmse', 'prediction_verified_score')]
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values):
        return False
    return values[1] >= 0 and abs(values[0]-values[2]) <= 1e-5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--existing-data', type=Path, required=True)
    parser.add_argument('--model', required=True, help='Local model/tokenizer directory')
    parser.add_argument('--exclude-inputs', type=Path, nargs='*', default=[],
                        help='Evaluation comparison_inputs.json files to reserve')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output must be empty')
    old = args.existing_data
    train = json.loads((old/'critic_train.json').read_text(encoding='utf-8'))
    manifest = json.loads((old/'sample_manifest.json').read_text(encoding='utf-8'))
    train_tasks = {r['task'] for r in manifest if r['split']=='train'}
    seen = {r['code_hash'] for r in manifest}
    heldout = set()
    for excluded_path in args.exclude_inputs:
        for row in json.loads(excluded_path.read_text(encoding='utf-8')):
            heldout.add(canonical_code(row['original_record']['function']))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    candidates, rejected, audits, hashes = [], Counter(), [], {}
    for path in sorted(args.logs.rglob('positive_trajectories.jsonl')):
        task = path.parent.name
        if task not in train_tasks:
            continue
        source = str(path.relative_to(args.logs))
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            row = json.loads(line)
            uid = f'{task}/{row["base_id"]}'
            try:
                base = row['base']
                code_hash = canonical_code(base['function'])
                if code_hash in seen:
                    raise ValueError('existing_train_or_dev_code')
                if code_hash in heldout:
                    raise ValueError('evaluation_code_reserved')
                if row.get('historical_label')!='positive' or row.get('label')!='positive':
                    raise ValueError('label_disagreement')
                refs = row['refinements']
                if not verified(base['backfill']) or not all(verified(r['backfill']) for r in refs):
                    raise ValueError('unverified_feedback')
                gains = [r['backfill']['score']-base['backfill']['score'] for r in refs]
                if max(gains) < .05:
                    raise ValueError('gain_below_005')
                if abs(max(gains)-row['score_gain']) > 1e-6:
                    raise ValueError('gain_mismatch')
                distinct_revisions = len({canonical_code(r['function']) for r in refs})
                tier = 'repeated_success' if distinct_revisions>=2 and all(g>.01 for g in gains) else 'single_or_mixed_success'
                best = max(refs, key=lambda r:r['backfill']['score'])
                if canonical_code(best['function']) in heldout:
                    raise ValueError('evaluation_target_reserved')
                actions, patches = reconstruct(base['function'], best['function'])
                messages = [{'role':'system','content':SYSTEM},
                            {'role':'user','content':json.dumps(make_input(base,base['backfill'],task),ensure_ascii=False,allow_nan=False)},
                            {'role':'assistant','content':json.dumps(actions,ensure_ascii=False,allow_nan=False)}]
                text = ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages)
                tokens = len(tokenizer.encode(text,add_special_tokens=False))
                if tokens > 4096:
                    raise ValueError('over_context_limit')
                candidates.append(dict(id=uid,task=task,code_hash=code_hash,tokens=tokens,source=source,
                    source_line=line_no,split='train',tier=tier,score_gain=max(gains),all_revision_gains=gains,
                    distinct_revision_codes=distinct_revisions,
                    target_origin='deterministic_diff_of_verified_revision',
                    best_revision_id=best['sample_order'],patches=patches,
                    revised_code_hash=canonical_code(best['function']),messages=messages))
            except (ValueError, SyntaxError, KeyError) as exc:
                reason=str(exc)
                rejected[reason]+=1
                audits.append(dict(id=uid,source=source,line=line_no,reason=reason))
    added, reserve = [], []
    for row in sorted(candidates,key=lambda r:(r['tier']!='repeated_success',-r['score_gain'])):
        if row['code_hash'] in seen:
            rejected['duplicate_new_code']+=1
            continue
        seen.add(row['code_hash'])
        (added if row['tier']=='repeated_success' else reserve).append(row)
    assert added, 'No high-confidence additions'
    write_json(out/'critic_train.json',train+[{'messages':r['messages']} for r in added])
    write_json(out/'critic_train_added.json',[{'messages':r['messages']} for r in added])
    write_json(out/'critic_reserve.json',[{'messages':r['messages']} for r in reserve])
    shutil.copyfile(old/'critic_dev.json',out/'critic_dev.json')
    shutil.copyfile(old/'dataset_info.json',out/'dataset_info.json')
    write_json(out/'sample_manifest.json',manifest+[{k:v for k,v in r.items() if k!='messages'} for r in added])
    write_json(out/'reserve_manifest.json',[{k:v for k,v in r.items() if k!='messages'} for r in reserve])
    write_json(out/'rejections.json',audits)
    report=dict(original_train=len(train),added_train=len(added),total_train=len(train)+len(added),
                unchanged_dev=len(json.loads((old/'critic_dev.json').read_text(encoding='utf-8'))),reserve_not_in_training=len(reserve),
                added_by_task=dict(Counter(r['task'] for r in added)),rejections=dict(rejected),
                min_best_gain=.05,min_each_revision_gain=.01,minimum_distinct_successful_revisions=2,
                exact_patch_reconstruction=True,source_sha256=hashes,
                dev_sha256=hashlib.sha256((out/'critic_dev.json').read_bytes()).hexdigest(),
                limitations=['Historical repeated successful revisions are not independent seed trials.',
                    'Reconstructed actions are new targets; their execution by Qwen3-8B is not yet validated.',
                    'New rationale text describes patch application, not inferred scientific causality.',
                    'Existing SFT samples are retained under their original filtering rules.',
                    'Code hash exclusion does not prove semantic novelty.'])
    write_json(out/'data_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
