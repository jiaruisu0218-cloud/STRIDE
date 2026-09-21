"""Refit one trusted candidate with its archived task specification."""
import argparse
import ast
import contextlib
import inspect
import json
import linecache
import math
import os
from pathlib import Path
import sys
import time
import types

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def worker(job_path, result_path):
    import numpy as np
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    start = time.monotonic()
    try:
        np.random.seed(0)
        tree = ast.parse(Path(job['spec_snapshot']).read_text(encoding='utf-8-sig'))
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'equation')]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = [d for d in node.decorator_list
                    if ast.unparse(d) not in ('evaluate.run', 'equation.evolve')]
        tree.body.extend(ast.parse(job['function']).body)
        ast.fix_missing_locations(tree)
        code = ast.unparse(tree)
        filename = f"<critic_backfill_{job['key']}>"
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        module = types.ModuleType('backfilled_equation')
        sys.modules[module.__name__] = module
        ns = module.__dict__
        ns['__PROGRAM_SOURCE__'] = code
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(code, filename, 'exec'), ns)
            ns['DEBUG_COMPLEXITY'] = False
            captured = {}
            def wrap(name):
                original = ns[name]
                def observed(*args, **kwargs):
                    value = original(*args, **kwargs)
                    captured[name] = {'args': args, 'value': value}
                    return value
                ns[name] = observed
            for name in ['_effective_param_count', '_sensitivity_cost', '_curvature_cost', '_full_complexity']:
                wrap(name)
            data = np.loadtxt(job['data'], delimiter=',', skiprows=1)
            X, y = data[:, :-1], data[:, -1]
            result = ns['evaluate']({'inputs': X, 'outputs': y})
            if len(result) < 4 or '_full_complexity' not in captured:
                raise ValueError('Specification did not produce a valid fitted result')
            score, params, method, nmse = result[:4]
            full = captured['_full_complexity']
            eq, *arguments, fitted = full['args']
            p = np.asarray(params, dtype=float)
            yp = ns['_safe_eval'](eq, *arguments, p)
            if yp is None:
                raise ValueError('Invalid prediction at fitted parameters')
            yp = np.asarray(yp, dtype=float).ravel()
            if yp.shape != y.shape or not np.all(np.isfinite(yp)):
                raise ValueError('Invalid prediction shape or nonfinite values')
            variance = float(np.var(y))
            denominator = variance if math.isfinite(variance) and variance >= 1e-12 else 1.0
            residual = y - yp
            mse = float(np.mean(residual ** 2))
            actual_nmse = mse / denominator
            neff = int(captured['_effective_param_count']['value'])
            sens = float(captured['_sensitivity_cost']['value'])
            curv = float(captured['_curvature_cost']['value'])
            weights = {k: float(ns[k]) for k in ['W_STRUCT', 'W_COMP', 'K_PARAM', 'K_SENS', 'K_CURVLOG']}
            pc = weights['K_PARAM'] * neff
            sc = weights['K_SENS'] * sens
            cc = weights['K_CURVLOG'] * float(np.log1p(curv))
            fit_reward = -weights['W_STRUCT'] * float(np.log(nmse + 1e-12))
            penalty = weights['W_COMP'] * (pc + sc + cc)
            predicted_score = -weights['W_STRUCT'] * float(np.log(actual_nmse + 1e-12)) - penalty
            valid = bool(np.isfinite(score) and abs(score - (fit_reward - penalty)) < 1e-8
                and abs(score - predicted_score) < 1e-6)
            names = list(inspect.signature(eq).parameters)[:-1]
            small_args = [a[:200] for a in arguments]
            base = np.asarray(eq(*small_args, p), dtype=float)
            scale = float(np.std(base))
            if not math.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            parameter_features = []
            for i, value in enumerate(p):
                step = 1e-4 * (1 + abs(value))
                pp = p.copy()
                pp[i] += step
                changed = np.asarray(eq(*small_args, pp), dtype=float)
                rms = float(np.sqrt(np.mean((changed-base) ** 2)))
                parameter_features.append({'index': i, 'value': float(value), 'step': float(step),
                    'rms_output_change': rms, 'rms_derivative': rms/step,
                    'normalized_output_change': rms/scale,
                    'effective': bool(np.isfinite(rms) and rms > max(1e-10, 1e-6*scale))})
            correlations = {}
            for name, values in zip(names, arguments):
                correlations[name] = float(np.corrcoef(residual, values)[0,1]) if np.std(residual)>1e-15 and np.std(values)>1e-15 else None
            source_tree = ast.parse(job['function'])
            fdef = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'equation')
            output = {'status': 'ok' if valid else 'prediction_score_mismatch',
                'provenance': 'offline refit; not the historical fitted parameters or original API prompt',
                'fit_seed': 0, 'n_restarts': int(ns['N_RESTARTS']), 'method': method,
                'fitted_parameters': params, 'specification_nmse': float(nmse), 'prediction_nmse': actual_nmse,
                'prediction_mse': mse, 'score': float(score), 'prediction_verified_score': predicted_score,
                'score_weights': weights, 'effective_parameter_count': neff, 'parameter_vector_length': len(params),
                'input_sensitivity': sens, 'input_curvature': curv, 'parameter_complexity': pc,
                'sensitivity_complexity': sc, 'curvature_complexity': cc,
                'weighted_fit_reward': fit_reward, 'weighted_complexity_penalty': penalty,
                'score_reconstruction_error': float(score - (fit_reward-penalty)),
                'parameter_sensitivity': parameter_features,
                'residual_summary': {'definition': 'y - prediction', 'mean': float(np.mean(residual)),
                    'std': float(np.std(residual)), 'quantiles_0_25_50_75_100': np.quantile(residual,[0,.25,.5,.75,1]).tolist(),
                    'correlation_with_inputs': correlations},
                'input_variables_in_equation_order': names,
                'task_description': ast.get_docstring(fdef), 'train_rows': len(y)}
    except Exception as exc:
        output = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
    output['key'] = job['key']
    output['elapsed_s'] = time.monotonic()-start
    dump(Path(result_path), output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("result")
    args = parser.parse_args()
    worker(args.job, args.result)
