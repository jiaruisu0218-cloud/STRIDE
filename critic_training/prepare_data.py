"""Build auditable success-edit SFT data from the uploaded offline-refit logs."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

SYSTEM = '''You are a scientific critic for symbolic regression. Given a task, a candidate Python equation, fitted parameters and training-data feedback, propose a coherent group of feasible edits that improve the fit-complexity tradeoff. All numerical fits and scores use the complete training data. Higher score is better. Use the supplied task-specific scoring weights. Input sensitivity and parameter sensitivity are different quantities; parameter sensitivity is diagnostic, not a separate penalty in this score. Do not invent observations or parameter values. Return ONLY a JSON object with exactly "actions" and "short_comment". Each action has "type" (REMOVE, SIMPLIFY, or ADD_OR_REPLACE), a concise "rationale", and "target" for REMOVE/SIMPLIFY or "proposal" for ADD_OR_REPLACE. No markdown, thinking trace, or revised Python function.'''


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def canonical_code(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if hasattr(node, 'body') and isinstance(node.body, list):
            node.body = [x for x in node.body if not (isinstance(x, ast.Expr) and
                         isinstance(x.value, ast.Constant) and isinstance(x.value.value, str))]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def rounded(value):
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Non-finite input value')
        return float(format(value, '.8g'))
    return value


def validate_actions(output, code):
    if not isinstance(output, dict) or output.get('error'):
        return 'critic_error'
    actions = output.get('actions')
    if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
        return 'invalid_actions'
    if not isinstance(output.get('short_comment'), str) or not output['short_comment'].strip():
        return 'invalid_comment'
    for action in actions:
        if not isinstance(action, dict) or action.get('type') not in ('REMOVE', 'SIMPLIFY', 'ADD_OR_REPLACE'):
            return 'invalid_action_type'
        field = 'proposal' if action['type'] == 'ADD_OR_REPLACE' else 'target'
        if any(not isinstance(action.get(k), str) or not action[k].strip() for k in (field, 'rationale')):
            return 'invalid_action_fields'
    # Conservative reconstruction policy: exclude numerical claims in prose,
    # while allowing structural powers and names such as x2 and params[3].
    prose = ' '.join([output['short_comment']] + [a['rationale'] for a in actions])
    prose = re.sub(r'\b(?:params|p)\s*\[\s*\d+\s*\]', 'PARAMETER', prose)
    prose = re.sub(r'(?:\*\*|\^)\s*\d+\b', 'POWER', prose)
    prose = re.sub(r'\b[A-Za-z_][A-Za-z_0-9]*\b', 'WORD', prose)
    if re.search(r'\d', prose):
        return 'numerical_claim_in_prose'
    # Literal decimal/scientific values in edits must already occur in the candidate.
    number = r'(?<![\w])[-+]?(?:\d+\.\d*|\.\d+|\d+[eE][-+]?\d+)(?:[eE][-+]?\d+)?'
    known = {float(v) for v in re.findall(number, code)}
    for action in actions:
        text = action.get('target', action.get('proposal', ''))
        if any(float(v) not in known for v in re.findall(number, text)):
            return 'new_numeric_literal_in_edit'
    return None


def make_input(base, feedback, task):
    code = base['function']
    function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
    weights = feedback.get('score_weights')
    if weights is None:
        if task != 'oscillator2':
            raise ValueError('Missing score weights')
        weights = dict(W_STRUCT=0.7, W_COMP=0.3, K_PARAM=1.0, K_SENS=0.05, K_CURVLOG=0.01)
    features = {k: feedback[k] for k in (
        'prediction_mse', 'prediction_nmse', 'score', 'effective_parameter_count',
        'parameter_vector_length', 'input_sensitivity', 'input_curvature',
        'parameter_complexity', 'sensitivity_complexity', 'curvature_complexity',
        'weighted_fit_reward', 'weighted_complexity_penalty', 'parameter_sensitivity')}
    features['score_formula'] = 'W_STRUCT*(-ln(NMSE+1e-12)) - W_COMP*(K_PARAM*n_eff + K_SENS*input_sensitivity + K_CURVLOG*ln(1+input_curvature))'
    features['score_weights'] = weights
    features['parameter_sensitivity_definition'] = 'One-sided parameter perturbation: step=1e-4*(1+abs(parameter)); RMS output change and RMS derivative on the first 200 training rows; normalized_output_change divides by prediction standard deviation.'
    return rounded({
        'task': {'description': feedback.get('task_description') or ast.get_docstring(function),
                 'input_variables': [a.arg for a in function.args.args if a.arg != 'params'],
                 'fit_split': 'full training data', 'train_rows': feedback['train_rows']},
        'candidate_code': code,
        'fitted_parameters': feedback['fitted_parameters'],
        'evaluation_feedback': features})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--min-score-gain', type=float, default=0.01)
    parser.add_argument('--dev-tasks', nargs='+', required=True, metavar='TASK',
                        help='Task names reserved for SFT development. Other eligible source tasks form the training split. Keep final evaluation tasks out of the input logs.')
    parser.add_argument('--cutoff-len', type=int, default=4096)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rejected, by_task, lengths = Counter(), Counter(), []
    accepted, audit, source_hashes = [], [], {}
    total = 0
    for path in sorted(Path(args.logs).rglob('positive_trajectories.jsonl')):
        source_hashes[str(path.relative_to(args.logs))] = hashlib.sha256(path.read_bytes()).hexdigest()
        task = path.parent.name
        for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            row = json.loads(line)
            total += 1
            reason = None
            base = row['base']
            feedback = base['backfill']
            output = row.get('critic_actions', row.get('critic_output'))
            gain = row.get('score_gain')
            uid = f'{task}/{row["base_id"]}'
            if row.get('label') != 'positive' or row.get('historical_label') != 'positive':
                reason = 'historical_refit_label_disagreement'
            elif not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain < args.min_score_gain:
                reason = 'small_gain'
            elif any(r['backfill'].get('status') != 'ok' for r in [base] + row['refinements']):
                reason = 'unverified_fit'
            else:
                reason = validate_actions(output, base['function'])
            if reason:
                rejected[reason] += 1
                audit.append(dict(id=uid, status='rejected', reason=reason, source=str(path), line=line_number))
                continue
            inputs = make_input(base, feedback, task)
            clean_output = {'actions': [], 'short_comment': output['short_comment']}
            for action in output['actions']:
                field = 'proposal' if action['type'] == 'ADD_OR_REPLACE' else 'target'
                clean_output['actions'].append({k: action[k] for k in ('type', field, 'rationale')})
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': json.dumps(inputs, ensure_ascii=False, allow_nan=False)},
                        {'role': 'assistant', 'content': json.dumps(clean_output, ensure_ascii=False, allow_nan=False)}]
            # Match LF's qwen3_nothink template exactly (no injected think block).
            text = ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages)
            length = len(tokenizer.encode(text, add_special_tokens=False))
            if length > args.cutoff_len:
                rejected['over_context_limit'] += 1
                audit.append(dict(id=uid, status='rejected', reason='over_context_limit', tokens=length))
                continue
            accepted.append(dict(id=uid, task=task, code_hash=canonical_code(base['function']),
                                 messages=messages, tokens=length, score_gain=gain,
                                 source=str(path), source_line=line_number))
    dev_hashes = {r['code_hash'] for r in accepted if r['task'] in args.dev_tasks}
    seen, splits, metadata = set(), {'train': [], 'dev': []}, []
    for row in sorted(accepted, key=lambda r: -r['score_gain']):
        split = 'dev' if row['task'] in args.dev_tasks else 'train'
        reason = None
        if split == 'train' and row['code_hash'] in dev_hashes:
            reason = 'cross_split_duplicate_code'
        elif (split, row['code_hash']) in seen:
            reason = 'duplicate_code'
        if reason:
            rejected[reason] += 1
            audit.append(dict(id=row['id'], status='rejected', reason=reason))
            continue
        seen.add((split, row['code_hash']))
        splits[split].append({'messages': row['messages']})
        metadata.append({k: v for k, v in row.items() if k != 'messages'} | {'split': split})
        by_task[f'{split}/{row["task"]}'] += 1
        lengths.append(row['tokens'])
    assert splits['train'] and splits['dev'], 'Empty split'
    assert not ({r['code_hash'] for r in metadata if r['split'] == 'train'} &
                {r['code_hash'] for r in metadata if r['split'] == 'dev'})
    for name, rows in splits.items():
        write_json(out / f'critic_{name}.json', rows)
    entry = {'formatting': 'sharegpt', 'columns': {'messages': 'messages'},
             'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user',
                      'assistant_tag': 'assistant', 'system_tag': 'system'}}
    write_json(out / 'dataset_info.json', {f'critic_{name}': {'file_name': f'critic_{name}.json', **entry}
                                          for name in splits})
    write_json(out / 'sample_manifest.json', metadata)
    write_json(out / 'rejections.json', audit)
    report = {'raw_positive_trajectories': total, 'train_samples': len(splits['train']),
              'dev_samples': len(splits['dev']), 'min_score_gain': args.min_score_gain,
              'dev_tasks': args.dev_tasks, 'counts_by_task': dict(sorted(by_task.items())),
              'rejections': dict(rejected), 'max_tokens': max(lengths),
              'mean_tokens': sum(lengths)/len(lengths), 'source_sha256': source_hashes,
              'data_sha256': {name: hashlib.sha256((out/f'critic_{name}.json').read_bytes()).hexdigest() for name in splits},
              'limitations': ['Historical actions paired with offline refits; filtering is conservative, not semantic certification.',
                              'Quantitative numbers in rationale/comment are excluded; powers, parameter indices and names are allowed. Some valid suggestions are excluded.',
                              'Development tasks are held out, but related mechanisms can occur in training tasks.',
                              'Success labels reflect joint critic-executor best saved outcomes, not causal critic improvement.']}
    write_json(out / 'data_report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256', 'data_sha256')}, indent=2))


if __name__ == '__main__':
    main()
