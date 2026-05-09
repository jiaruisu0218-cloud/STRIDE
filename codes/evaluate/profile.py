from __future__ import annotations

import os
import os.path
from typing import Dict, Any
import logging
import json

from codes.update import code_manipulation
from torch.utils.tensorboard import SummaryWriter


def _get_fn_meta(program: Any) -> Dict[str, Any]:
    """Collect metadata from program object without breaking compatibility."""
    meta: Dict[str, Any] = {}

    # 1) preferred: program.metadata dict
    m = getattr(program, "metadata", None)
    if isinstance(m, dict):
        meta.update(m)

    # 2) common attached attributes (safe reads)
    for k in [
        "island_id",
        "cluster_id",        # experience buffer cluster index within island (TF-IDF or score signature)
        "version_generated",
        "sample_time",
        "evaluate_time",
        "global_sample_nums",
        "refine",            # e.g., "base" / "candidate" / "winner"
        "refine_parent",     # e.g., "m"
        "refine_round",      # e.g., 1..K
        "critic_json",       # if you attach it
        "stage",             # "fast" / "full"
        "fast_eval",         # bool
        "param_optimization_method",  # e.g. MIXED / BFGS / TRF from evaluate() 3rd return
    ]:
        if hasattr(program, k):
            v = getattr(program, k)
            # avoid writing huge objects accidentally
            if k == "critic_json" and isinstance(v, str):
                # already stringified JSON; store as-is
                meta[k] = v
            else:
                meta[k] = v

    return meta


def _get_sample_id(program: Any) -> str:
    """
    Determine sample_id for saving:
    - prefer program.sample_id (string like "m_n")
    - else fallback to program.global_sample_nums (int)
    """
    sid = getattr(program, "sample_id", None)
    if sid is not None:
        return str(sid)

    g = getattr(program, "global_sample_nums", None)
    if g is None:
        return "0"
    return str(g)


class Profiler:
    def __init__(
        self,
        log_dir: str | None = None,
        pkl_dir: str | None = None,
        max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file. (unused in current implementation)
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, "samples") if log_dir else None
        if self._json_dir:
            os.makedirs(self._json_dir, exist_ok=True)

        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0

        # key can be str (e.g., "m_n") or int (fallback)
        self._all_sampled_functions: Dict[str, code_manipulation.Function] = {}

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            "Best Score of Function",
            self._cur_best_program_score,
            global_step=self._num_samples,
        )
        self._writer.add_scalars(
            "Legal/Illegal Function",
            {
                "legal function num": self._evaluate_success_program_num,
                "illegal function num": self._evaluate_failed_program_num,
            },
            global_step=self._num_samples,
        )
        self._writer.add_scalars(
            "Total Sample/Evaluate Time",
            {"sample time": self._tot_sample_time, "evaluate time": self._tot_evaluate_time},
            global_step=self._num_samples,
        )
        if self._cur_best_program_str is not None:
            self._writer.add_text(
                "Best Function String",
                self._cur_best_program_str,
                global_step=self._num_samples,
            )

    def _write_json(self, program: code_manipulation.Function):
        if not self._json_dir:
            return

        sample_id = _get_sample_id(program)
        function_str = str(program)
        score = getattr(program, "score", None)

        content = {
            "sample_order": sample_id,   # allow "m_n"
            "function": function_str,
            "score": score,
            "meta": _get_fn_meta(program),
        }

        # file name: samples_{sample_id}.json  (compatible with old int-based scheme)
        path = os.path.join(self._json_dir, f"samples_{sample_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    def register_function(self, program: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_id = _get_sample_id(program)

        # Re-register: refresh JSON only (avoid double TensorBoard / console blocks).
        if sample_id in self._all_sampled_functions:
            self._all_sampled_functions[sample_id] = program
            self._write_json(program)
            return

        self._num_samples += 1
        self._all_sampled_functions[sample_id] = program

        self._record_and_verbose(sample_id)
        self._write_tensorboard()
        self._write_json(program)

    def _record_and_verbose(self, sample_id: str):
        function = self._all_sampled_functions[sample_id]
        function_str = str(function).strip("\n")

        sample_time = getattr(function, "sample_time", None)
        evaluate_time = getattr(function, "evaluate_time", None)
        score = getattr(function, "score", None)
        meta = _get_fn_meta(function)

        print("================= Evaluated Function =================")
        print(f"{function_str}")
        print("------------------------------------------------------")
        print(f"Score        : {score}")
        print(f"Sample time  : {sample_time}")
        print(f"Evaluate time: {evaluate_time}")
        print(f"Sample orders: {sample_id}")
        if meta.get("refine"):
            print(f"Refine tag  : {meta.get('refine')}")
        if meta.get("stage"):
            print(f"Stage       : {meta.get('stage')}")
        if meta.get("param_optimization_method"):
            print(f"Param opt   : {meta.get('param_optimization_method')}")
        print("======================================================\n\n")

        if score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_id
            self._cur_best_program_str = function_str

        # very conservative: treat None as failed; otherwise success
        if score is None:
            self._evaluate_failed_program_num += 1
        else:
            self._evaluate_success_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
