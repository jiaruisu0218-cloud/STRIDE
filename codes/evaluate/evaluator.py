from __future__ import annotations

from abc import ABC, abstractmethod
import ast
import copy
import dataclasses
import json
import logging
import multiprocessing
from codes.evaluate import profile as exp_profile
import random
import re
import time
from typing import Any, Dict, Sequence, Set, Tuple, Type

import numpy as np

from codes.update import buffer
from codes.update import code_manipulation
from codes.evaluate import evaluator_accelerate

# Optional Critic import (must output JSON/dict; str is ignored)
try:
    from codes.refine.critic import PhysicsCritic
except Exception:
    PhysicsCritic = None


# ---------------------------
# Analysis Result
# ---------------------------
@dataclasses.dataclass
class AnalysisResult:
    scores_per_test: dict[str, float]
    reduced_score: float | None
    best_params: list[float] | None
    runs_ok: bool


# ---------------------------
# Metadata helper (never inject into body)
# ---------------------------
def _set_fn_meta(fn: Any, key: str, value: Any) -> None:
    """Attach metadata onto Function object without touching fn.body."""
    try:
        setattr(fn, key, value)
        return
    except Exception:
        pass

    meta = getattr(fn, "metadata", None)
    if meta is None or not isinstance(meta, dict):
        try:
            setattr(fn, "metadata", {})
            meta = getattr(fn, "metadata")
        except Exception:
            return
    meta[key] = value


def _extract_train_nmse_from_eval_tuple(test_output: Any) -> float | None:
    """Last element is train NMSE when it is a non-negative finite float (spec convention)."""
    if not isinstance(test_output, (tuple, list)) or len(test_output) < 3:
        return None
    last = test_output[-1]
    if isinstance(last, (int, float)):
        v = float(last)
        if np.isfinite(v) and v >= 0.0:
            return v
    return None


def _param_opt_method_from_eval_tuple(test_output: Any) -> str | None:
    """Optimization method sits at index 2 only when it is a string (not train_nmse)."""
    if not isinstance(test_output, (tuple, list)) or len(test_output) < 3:
        return None
    mid = test_output[2]
    return mid if isinstance(mid, str) else None


def _snapshot_split_keys_for_early_stop(inputs: Any) -> list[str] | None:
    if isinstance(inputs, dict):
        return [str(k) for k in inputs.keys()]
    return None


def _early_stop_applies_to_split(current_input: Any, split_keys: list[str] | None) -> bool:
    if split_keys is None:
        return True
    if len(split_keys) == 1:
        return True
    return str(current_input).lower() == "train"


def _maybe_request_sampler_stop_early_nmse(config: Any, train_nmse: float) -> None:
    if config is None:
        return
    thr = getattr(config, "early_stop_train_nmse_threshold", None)
    if thr is None:
        return
    try:
        t = float(thr)
    except (TypeError, ValueError):
        return
    if t <= 0 or not np.isfinite(train_nmse):
        return
    if train_nmse < t:
        from codes.sample.sampler import Sampler

        Sampler.request_stop()
        logging.info(
            "Early stop requested: train_nmse=%.6e < threshold=%.6e",
            train_nmse,
            t,
        )


# ---------------------------
# Pruning utilities (safe)
# ---------------------------
def _is_params_index(node: ast.AST, idx: int) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "params"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == idx
    )


def _flatten_add(expr: ast.AST) -> list[ast.AST]:
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _flatten_add(expr.left) + _flatten_add(expr.right)
    return [expr]


def _rebuild_add(terms: list[ast.AST]) -> ast.AST:
    if not terms:
        return ast.Constant(value=0.0)
    out = terms[0]
    for t in terms[1:]:
        out = ast.BinOp(left=out, op=ast.Add(), right=t)
    return out


def _term_has_inactive_coeff(term: ast.AST, inactive: Set[int]) -> bool:
    """
    Only prune if params[i] appears as a direct multiplicative factor
    at top level term (avoid pruning params inside exp/sin/...).
    """
    # term itself is params[i]
    for i in inactive:
        if _is_params_index(term, i):
            return True

    # look for params[i] as a factor in a multiplication tree
    if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult):
        stack = [term]
        while stack:
            n = stack.pop()
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
                stack.append(n.left)
                stack.append(n.right)
            else:
                for i in inactive:
                    if _is_params_index(n, i):
                        return True
    return False


def prune_inactive_add_terms(body: str, inactive: Set[int]) -> Tuple[str, int]:
    """
    Prune top-level additive terms in Return if the term has inactive params[i] as a multiplicative factor.
    Input body must already be indented (Function.body format).
    """
    try:
        wrapper = "def _f_():\n" + body
        tree = ast.parse(wrapper)

        removed = 0

        class Pruner(ast.NodeTransformer):
            def visit_Return(self, node: ast.Return):
                nonlocal removed
                if node.value is None:
                    return node
                terms = _flatten_add(node.value)
                kept: list[ast.AST] = []
                for t in terms:
                    if _term_has_inactive_coeff(t, inactive):
                        removed += 1
                    else:
                        kept.append(t)
                node.value = _rebuild_add(kept)
                return node

        tree = Pruner().visit(tree)
        ast.fix_missing_locations(tree)

        new_src = ast.unparse(tree)
        # drop the wrapper def line and remove its 4-space indentation
        new_lines = new_src.splitlines()[1:]
        new_body = "\n".join(line[4:] if line.startswith("    ") else line for line in new_lines).rstrip() + "\n"
        return new_body, removed
    except Exception:
        return body, 0

def _has_return_stmt(indented_body: str) -> bool:
    """Reject code bodies without any explicit return statement."""
    try:
        wrapper = "def _f_():\n" + indented_body
        tree = ast.parse(wrapper)
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                return True
    except Exception:
        return False
    return False

# ---------------------------
# Code trimming / compilation
# ---------------------------
class _FunctionLineVisitor(ast.NodeVisitor):
    def __init__(self, target_function_name: str) -> None:
        self._target_function_name = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        assert self._function_end_line is not None
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """Extract body lines from model output (as a function body block)."""
    if not generated_code:
        return ""

    code = f"def fake_function_header():\n{generated_code}"
    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            if e.lineno is None:
                return ""
            code = "\n".join(code.splitlines()[: e.lineno - 1])

    if not code:
        return ""

    visitor = _FunctionLineVisitor("fake_function_header")
    visitor.visit(tree)
    body_lines = code.splitlines()[1 : visitor.function_end_line]
    return "\n".join(body_lines) + "\n\n"


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """Compile sample into a Program string and return the evolved Function object + program text."""
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f"{function_to_evolve}_v{version_generated}",
            target_name=function_to_evolve,
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    return evolved_function, str(program)


# ---------------------------
# Sandboxes
# ---------------------------
class Sandbox(ABC):
    @abstractmethod
    def run(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        inputs: Any,
        test_input: str,
        timeout_seconds: int,
        **kwargs,
    ) -> tuple[Any, bool]:
        raise NotImplementedError


class LocalSandbox(Sandbox):
    def __init__(self, verbose: bool = False, numba_accelerate: bool = False):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        inputs: Any,
        test_input: str,
        timeout_seconds: int,
        **kwargs,
    ) -> tuple[Any, bool]:
        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue),
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            results = (None, False)
        else:
            results = self._get_results(result_queue)

        if self._verbose:
            self._print_evaluation_details(program, results, func_to_evolve=function_to_evolve)

        return results

    def _get_results(self, queue):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return (None, False)

    def _print_evaluation_details(self, program, results, **kwargs):
        print("================= Evaluated Program =================")
        func_name = kwargs.get("func_to_evolve", "equation")
        function = code_manipulation.text_to_program(program).get_function(func_name)
        print(f"{str(function).strip()}\n-----------------------------------------------------")
        print(f"Score: {results}\n=====================================================\n\n")

    def _compile_and_run_function(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        dataset: Any,
        numba_accelerate: bool,
        result_queue,
    ):
        try:
            if numba_accelerate:
                try:
                    program_acc = evaluator_accelerate.add_numba_decorator(
                        program=program,
                        function_to_evolve=function_to_evolve,
                    )
                    all_globals_namespace: dict[str, Any] = {}
                    # Inject program source for inspect.getsource() fallback
                    all_globals_namespace['__PROGRAM_SOURCE__'] = program_acc
                    exec(program_acc, all_globals_namespace)
                    fn = all_globals_namespace[function_to_run]
                    results = fn(dataset)
                    if self._validate_results(results):
                        result_queue.put((results, True))
                        return
                except Exception:
                    pass

            all_globals_namespace = {}
            # Inject program source for inspect.getsource() fallback
            all_globals_namespace['__PROGRAM_SOURCE__'] = program
            exec(program, all_globals_namespace)
            fn = all_globals_namespace[function_to_run]
            results = fn(dataset)

            if not self._validate_results(results):
                result_queue.put((None, False))
                return

            result_queue.put((results, True))
        except Exception:
            result_queue.put((None, False))

    def _validate_results(self, results: Any) -> bool:
        if not isinstance(results, (int, float, tuple, list)):
            return False
        if isinstance(results, (tuple, list)) and len(results) == 0:
            return False
        return True


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f"{function_to_evolve}_v"):
            return True
    return False


# ---------------------------
# Evaluator
# ---------------------------
class Evaluator:
    """Analyses generated functions."""

    def __init__(
        self,
        database: buffer.ExperienceBuffer,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        inputs: Any,
        timeout_seconds: int = 30,
        sandbox_class: Type[Sandbox] = LocalSandbox,
        config: Any = None,
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = int(timeout_seconds)
        self._sandbox = sandbox_class()

        self._config = config

        # Extract problem description from template docstring
        self.problem_description = "Mathematical function discovery."
        try:
            target_func = self._template.get_function(self._function_to_evolve)
            if target_func.docstring:
                self.problem_description = target_func.docstring.strip()
        except Exception:
            pass

        # Critic configuration
        self.critic = None
        self.critic_prob = float(getattr(config, "critic_prob", 0.2)) if config else 0.2
        self.critic_score_threshold = float(getattr(config, "critic_score_threshold", -1e-9)) if config else -1e-9

        if PhysicsCritic is not None and config is not None:
            try:
                self.critic = PhysicsCritic(config)
            except Exception:
                self.critic = None

        # fast-eval config (optional)
        self.fast_eval_inputs = int(getattr(config, "fast_eval_inputs", 1)) if config else 1
        self.fast_eval_timeout_seconds = int(getattr(config, "fast_eval_timeout_seconds", 10)) if config else 10

    # ---------- critic normalization ----------
    @staticmethod
    def _normalize_critic_json(x: Any) -> dict | None:
        if x is None:
            return None
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
        return None

    # ---------- main analysis ----------
    def analyse(
        self,
        sample: str,
        island_id: int | None,
        version_generated: int | None,
        *,
        do_register: bool = True,
        fast_eval: bool = False,
        skip_critic: bool = False,
        return_function: bool = False,
        **kwargs,
    ) -> AnalysisResult | tuple[AnalysisResult, code_manipulation.Function] | None:
        """
        Evaluate a sample.

        - do_register=True will register into buffer.
        - fast_eval=True enforces smaller budget via fewer splits and shorter timeout.
        - skip_critic=True disables critic (fast_eval will also force this).
        - return_function=True returns (AnalysisResult, Function) so caller can read metadata.
        """
        # fast_eval must never trigger critic
        if fast_eval:
            skip_critic = True

        new_function, program = _sample_to_program(sample, version_generated, self._template, self._function_to_evolve)

        # ---------------------------
        # EARLY REJECT: must have Return
        # ---------------------------
        if not _has_return_stmt(new_function.body):
            # Invalid body: reject without sandbox.
            result = AnalysisResult(
                scores_per_test={},
                reduced_score=None,
                best_params=None,
                runs_ok=False,
            )
            if return_function:
                return result, new_function
            return result

        scores_per_test: Dict[str, float] = {}
        best_params: list[float] | None = None

        t0 = time.time()
        did_postprocess = False

        # iterator over splits
        if isinstance(self._inputs, dict):
            keys = list(self._inputs.keys())
            input_iterator = keys[: self.fast_eval_inputs] if fast_eval else keys
        else:
            input_iterator = self._inputs[: self.fast_eval_inputs] if fast_eval else self._inputs

        timeout = self.fast_eval_timeout_seconds if fast_eval else self._timeout_seconds

        split_keys = _snapshot_split_keys_for_early_stop(self._inputs)

        for current_input in input_iterator:
            test_output, runs_ok = self._sandbox.run(
                program,
                self._function_to_run,
                self._function_to_evolve,
                self._inputs,
                current_input,
                timeout,
            )

            if not (runs_ok and (not _calls_ancestor(program, self._function_to_evolve)) and test_output is not None):
                continue

            score = None

            # Case A: returns (score, params)
            if isinstance(test_output, (tuple, list)) and len(test_output) >= 1:
                if (not fast_eval) and _early_stop_applies_to_split(current_input, split_keys):
                    tnmse0 = _extract_train_nmse_from_eval_tuple(test_output)
                    if tnmse0 is not None:
                        _maybe_request_sampler_stop_early_nmse(self._config, tnmse0)

                score = test_output[0]

                if len(test_output) >= 2 and not did_postprocess:
                    did_postprocess = True
                    best_params = test_output[1]

                    _pom = _param_opt_method_from_eval_tuple(test_output)
                    if _pom is not None:
                        _set_fn_meta(new_function, "param_optimization_method", _pom)

                    # Parse input variables for critic context
                    raw_args = new_function.args.split(",")
                    input_vars: list[str] = []
                    for arg in raw_args:
                        var_name = arg.split(":")[0].strip()
                        if var_name and ("params" not in var_name) and var_name != "self":
                            input_vars.append(var_name)

                    orig_body = new_function.body

                    # robust param slicing: params[i] / params[:n]
                    pattern_idx = r"(?:params|p|P|w)\[(\d+)\]"
                    param_matches = re.findall(pattern_idx, orig_body)
                    pattern_slice = r"(?:params|p|P|w)\[:(\d+)\]"
                    slice_match = re.search(pattern_slice, orig_body)

                    params_for_critic = best_params
                    if isinstance(best_params, (list, tuple)):
                        if param_matches:
                            max_idx = max(int(m) for m in param_matches)
                            sl = min(max_idx + 1, len(best_params))
                            params_for_critic = list(best_params[:sl])
                        elif slice_match:
                            sl = int(slice_match.group(1))
                            sl = min(sl, len(best_params))
                            params_for_critic = list(best_params[:sl])
                        else:
                            params_for_critic = list(best_params)

                    # -------- critic JSON only --------
                    critic_triggered = False
                    critic_json: dict | None = None

                    if (
                        (not skip_critic)
                        and self.critic is not None
                        and isinstance(score, (int, float))
                        and (float(score) > self.critic_score_threshold)
                        and (random.random() < self.critic_prob)
                    ):
                        critic_triggered = True
                        critic_kwargs = {
                            "code": orig_body,
                            "params": params_for_critic,
                            "input_vars": input_vars,
                            "problem_description": self.problem_description,
                        }

                        critic_out = None
                        try:
                            if hasattr(self.critic, "critique_json"):
                                critic_out = self.critic.critique_json(**critic_kwargs)
                            elif hasattr(self.critic, "propose_actions"):
                                critic_out = self.critic.propose_actions(**critic_kwargs)
                            else:
                                critic_out = self.critic.critique(**critic_kwargs)  # type: ignore[attr-defined]
                        except Exception:
                            critic_out = None

                        critic_json = self._normalize_critic_json(critic_out)

                    _set_fn_meta(new_function, "critic_triggered", bool(critic_triggered))
                    _set_fn_meta(new_function, "critic_json", critic_json)
                    _set_fn_meta(new_function, "params_for_critic", params_for_critic)

                    # -------- pruning (full eval only recommended; keep minimal diff) --------
                    removed_terms = 0
                    if not fast_eval and isinstance(best_params, (list, tuple)):
                        used_indices = {int(m) for m in param_matches} if param_matches else set(range(len(best_params)))
                        eps = 1e-4
                        inactive = {i for i in used_indices if i < len(best_params) and abs(best_params[i]) < eps}

                        if inactive:
                            pruned_body, removed_terms = prune_inactive_add_terms(orig_body, inactive)
                            if removed_terms > 0:
                                new_function.body = pruned_body

                                # re-run once after pruning
                                program_obj = copy.deepcopy(self._template)
                                evolved_func = program_obj.get_function(self._function_to_evolve)
                                evolved_func.body = new_function.body
                                program_rerun = str(program_obj)

                                test_output2, ok2 = self._sandbox.run(
                                    program_rerun,
                                    self._function_to_run,
                                    self._function_to_evolve,
                                    self._inputs,
                                    current_input,
                                    timeout,
                                )

                                if ok2 and isinstance(test_output2, (tuple, list)) and len(test_output2) >= 2:
                                    score = test_output2[0]
                                    best_params = test_output2[1]
                                    _pom2 = _param_opt_method_from_eval_tuple(test_output2)
                                    if _pom2 is not None:
                                        _set_fn_meta(new_function, "param_optimization_method", _pom2)
                                    if (not fast_eval) and _early_stop_applies_to_split(current_input, split_keys):
                                        tnmse1 = _extract_train_nmse_from_eval_tuple(test_output2)
                                        if tnmse1 is not None:
                                            _maybe_request_sampler_stop_early_nmse(self._config, tnmse1)

                    _set_fn_meta(new_function, "removed_terms", int(removed_terms))

            # Case B: returns only score
            elif isinstance(test_output, (int, float)):
                score = test_output

            if score is not None and isinstance(score, (int, float)):
                scores_per_test[str(current_input)] = float(score)

        evaluate_time = time.time() - t0

        reduced = buffer._reduce_score(scores_per_test) if scores_per_test else None
        result = AnalysisResult(
            scores_per_test=scores_per_test,
            reduced_score=reduced,
            best_params=best_params,
            runs_ok=bool(scores_per_test),
        )

        # Always attach basic fields onto function (even without profiler)
        try:
            new_function.score = reduced
        except Exception:
            pass
        try:
            new_function.sample_time = kwargs.get("sample_time", None)
        except Exception:
            pass
        try:
            new_function.evaluate_time = evaluate_time
        except Exception:
            pass
        try:
            new_function.global_sample_nums = kwargs.get("global_sample_nums", None)
        except Exception:
            pass

        # Register into experience buffer ONLY if caller requests
        if do_register and scores_per_test:
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time,
            )
        else:
            # optional: if someone still passes profiler, keep compatibility
            profiler: exp_profile.Profiler = kwargs.get("profiler", None)
            if profiler and scores_per_test:
                sample_order = kwargs.get("sample_order", None)
                global_sample_nums = kwargs.get("global_sample_nums", None)
                try:
                    new_function.global_sample_nums = sample_order if sample_order is not None else global_sample_nums
                except Exception:
                    pass
                profiler.register_function(new_function)

        if return_function:
            return result, new_function
        return result

    # ---------- sync APIs for Sampler refine ----------
    def analyse_sync(
        self,
        sample: str,
        island_id: int | None,
        version_generated: int | None,
        *,
        commit: bool = True,
        fast_eval: bool = False,
        skip_critic: bool = False,
        force_filename: str | None = None,
        meta_extra: dict | None = None,
        stage: str | None = None,
        return_fn: bool = False,
        **kwargs,
    ) -> dict | tuple[dict, code_manipulation.Function]:
        """
        Synchronous evaluation wrapper that returns a dict for Sampler refine loop.
        - commit controls register_program.
        - force_filename/meta_extra/stage are passed through kwargs for your logging pipeline.
        - return_fn=True returns (dict, Function) so Sampler can reuse stage2 outputs to commit without re-evaluating.
        """
        if meta_extra:
            kwargs = dict(kwargs)
            kwargs.update(meta_extra)
        if stage is not None:
            kwargs = dict(kwargs)
            kwargs["stage"] = stage
        if force_filename is not None:
            kwargs = dict(kwargs)
            kwargs["force_filename"] = force_filename

        out = self.analyse(
            sample,
            island_id,
            version_generated,
            do_register=commit,
            fast_eval=fast_eval,
            skip_critic=skip_critic,
            return_function=True,
            **kwargs,
        )

        # If analysis failed
        if out is None:
            fail_dict = {
                "score": -float("inf"),
                "scores_per_test": {},
                "best_params": None,
                "critic_json": None,
                "removed_terms": None,
                "runs_ok": False,
            }
            # NOTE: no Function object on failure; caller must use runs_ok before commit.
            return fail_dict

        result, fn = out
        score = result.reduced_score if result.reduced_score is not None else -float("inf")

        critic_json = None
        if hasattr(fn, "critic_json"):
            critic_json = getattr(fn, "critic_json")
        critic_json = self._normalize_critic_json(critic_json)

        removed_terms = None
        if hasattr(fn, "removed_terms"):
            try:
                removed_terms = int(getattr(fn, "removed_terms"))
            except Exception:
                removed_terms = None

        out_dict = {
            "score": float(score),
            "scores_per_test": dict(result.scores_per_test),
            "best_params": result.best_params,
            "critic_json": critic_json,
            "removed_terms": removed_terms,
            "runs_ok": bool(result.runs_ok),
            "param_optimization_method": getattr(fn, "param_optimization_method", None),
        }

        if return_fn:
            return out_dict, fn
        return out_dict

    def score_sync(
        self,
        sample: str,
        island_id: int | None,
        version_generated: int | None,
        **kwargs,
    ) -> dict:
        """
        Stage-1 fast evaluation:
        - fast_eval=True (fewer splits + shorter timeout)
        - skip_critic=True
        - do_register=False
        """
        res = self.analyse(
            sample,
            island_id,
            version_generated,
            do_register=False,
            fast_eval=True,
            skip_critic=True,
            return_function=False,
            **kwargs,
        )

        if res is None:
            return {"score": -float("inf"), "runs_ok": False, "scores_per_test": {}}

        score = res.reduced_score if res.reduced_score is not None else -float("inf")
        return {
            "score": float(score),
            "runs_ok": bool(res.runs_ok),
            "scores_per_test": dict(res.scores_per_test),
        }

    def commit_pre_evaluated(
        self,
        fn: code_manipulation.Function,
        island_id: int | None,
        scores_per_test: dict[str, float],
        **kwargs,
    ) -> None:
        """
        Commit a previously evaluated Function into experience buffer WITHOUT re-evaluating.
        This is the key to avoid double-scoring on commit.
        """
        if not scores_per_test:
            return
        self._database.register_program(
            fn,
            island_id,
            scores_per_test,
            **kwargs,
        )
