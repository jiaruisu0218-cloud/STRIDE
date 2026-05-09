"""Class for sampling new program skeletons with optional refine loop (API-first).

Policy (your requirement):
- Only log STAGE-2 (full/precise evaluation) items into logs/samples as m_n.
- Stage-1 fast screening produces NO logs, NO sample json files.
- For each global sample m:
    * base full eval is logged as m_0
    * if refine triggers, at most two finalists enter stage2 and are logged as m_1 and m_2
    * finally commit ONLY ONE winner into experience buffer (best among base vs refine winners)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Collection, Sequence, Type

import ast
import json
import os
import time
import random

import numpy as np
import requests

from codes.evaluate import evaluator
from codes.update import buffer
from codes import config as config_lib
from absl import logging


# --------------------------
# LLM interface
# --------------------------
class LLM(ABC):
    """Abstract LLM interface: MUST be implemented by concrete LLM classes."""

    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = int(samples_per_prompt)

    @abstractmethod
    def draw_samples(
        self,
        prompt: str,
        config: config_lib.Config,
        *,
        n: int | None = None,
    ) -> Collection[str]:
        """Return a collection of raw model outputs (strings)."""
        raise NotImplementedError


# --------------------------
# Utilities
# --------------------------
def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract function body from model output.
    Keeps robust AST-based strategy:
    - if markdown fenced code exists, only keep code block
    - try AST parse, locate first FunctionDef body
    - fallback: heuristic lines containing operators/return/np
    """
    lines = sample.splitlines()
    clean_lines: list[str] = []
    in_code_block = False
    has_markdown = any(line.strip().startswith("```") for line in lines)

    if has_markdown:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                clean_lines.append(line)
    else:
        clean_lines = lines

    code_text = "\n".join(clean_lines)

    # AST locate function body
    try:
        tree = ast.parse(code_text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.body:
                start_line = node.body[0].lineno - 1
                end_line = node.end_lineno
                source_lines = code_text.splitlines()
                body_lines = source_lines[start_line:end_line]
                if body_lines:
                    first_line = body_lines[0]
                    indent_len = len(first_line) - len(first_line.lstrip())
                    fixed_lines = []
                    for line in body_lines:
                        if len(line) >= indent_len:
                            fixed_lines.append("    " + line[indent_len:])
                        else:
                            fixed_lines.append(line)
                    return "\n".join(fixed_lines).rstrip() + "\n"
    except Exception:
        pass

    # fallback heuristics
    final_code: list[str] = []
    for line in clean_lines:
        if any(k in line for k in ["=", "return", "np.", "+", "-", "*", "/", "**"]):
            if not line.startswith("    "):
                final_code.append("    " + line)
            else:
                final_code.append(line)
    return "\n".join(final_code).rstrip() + "\n"


def _safe_json_loads(s: str) -> dict | None:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _maybe_normalize_critic_json(x: Any) -> dict | None:
    """Normalize critic output to a dict (from dict or JSON string)."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        return _safe_json_loads(x)
    return None


def _build_refine_prompt(
    base_prompt: str,
    base_body: str,
    critic_json: dict,
) -> str:
    """
    Refine prompt:
    - each completion MUST be ONE variant (so we can request n=K)
    - only use critic JSON (no str)
    """
    critic_blob = json.dumps(critic_json, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "You are generating equation function BODIES for symbolic regression.",
            "",
            "HARD REQUIREMENTS (must satisfy):",
            "1) Output ONLY the function body (no def line), indented with exactly 4 spaces.",
            "2) The body MUST be executable Python and MUST contain an explicit `return` statement.",
            "3) The `return` must return a numpy array .",
            "4) Do NOT output partial code. Do NOT stop after defining variables. Always finish with return.",
            "5) Do NOT include markdown fences, explanations, or multiple variants.",
            "6) You may ONLY use variables that appear in the function signature (input variables + `params`) and `np`.",
            "",
            "=== BASE EQUATION (function body) ===",
            base_body,
            "",
            "=== CRITIC ACTIONS (JSON) ===",
            critic_blob,
        ]
    )


def _drop_profiler(kwargs: dict) -> dict:
    """Remove profiler to guarantee no logging / no json files."""
    k = dict(kwargs)
    k.pop("profiler", None)
    return k


def _kwargs_for_buffer_commit(kwargs: dict) -> dict:
    """
    Pass profiler into register_program on commit so existing samples_*.json can be refreshed
    (see profile.Profiler.register_function).
    """
    k = _drop_profiler(kwargs)
    prof = kwargs.get("profiler")
    if prof is not None:
        k = dict(k)
        k["profiler"] = prof
    return k


class Sampler:
    """Node that samples program skeleton continuations and sends them for analysis."""
    _global_samples_nums: int = 0
    _stop_requested: bool = False

    @classmethod
    def reset_stop(cls) -> None:
        cls._stop_requested = False

    @classmethod
    def request_stop(cls) -> None:
        cls._stop_requested = True

    @classmethod
    def is_stop_requested(cls) -> bool:
        return bool(cls._stop_requested)

    def __init__(
        self,
        database: buffer.ExperienceBuffer,
        evaluators: Sequence[evaluator.Evaluator],
        samples_per_prompt: int,
        config: config_lib.Config,
        max_sample_nums: int | None = None,
        llm_class: Type[LLM] | None = None,
    ):
        self._samples_per_prompt = int(samples_per_prompt)
        self._database = database
        self._evaluators = evaluators
        self._max_sample_nums = max_sample_nums
        self.config = config

        if llm_class is None:
            llm_class = ApiLLM
        self._llm: LLM = llm_class(self._samples_per_prompt)

        # refine hyperparams
        self.refine_enabled = bool(getattr(config, "refine_enabled", True))
        self.refine_prob = float(getattr(config, "refine_prob", 0.5))
        self.refine_score_threshold = float(getattr(config, "refine_score_threshold", -1e9))
        self.refine_k = int(getattr(config, "refine_k", 5))
        self.refine_eps = float(getattr(config, "refine_eps", 0.01))

        self.refine_use_fast_stage1 = bool(getattr(config, "refine_use_fast_stage1", True))

        # logging policy
        self.force_m_n_filenames = True

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def _next_global_sample(self) -> int:
        self.__class__._global_samples_nums += 1
        return self.__class__._global_samples_nums

    def sample(self, **kwargs):
        """
        Logging policy:
        - Base: one full stage-2 eval (writes samples_{m}_0.json).
        - Refine: stage-1 fast eval does not log; finalists get full stage-2 eval and log.
        - Commit: use commit_pre_evaluated(fn, scores_per_test) into the buffer (no re-eval).
        """
        while True:
            if self._max_sample_nums and self._get_global_sample_nums() >= self._max_sample_nums:
                break
            if self.__class__.is_stop_requested():
                logging.info("Sampler stopping: early stop (train NMSE below threshold).")
                break

            prompt = self._database.get_prompt()
            logging.info(
                "Sampler heartbeat: global_samples=%d/%s island=%s",
                self._get_global_sample_nums(),
                self._max_sample_nums if self._max_sample_nums is not None else "inf",
                prompt.island_id,
            )

            t0 = time.time()
            base_samples = self._llm.draw_samples(prompt.code, self.config, n=self._samples_per_prompt)
            sample_time = (time.time() - t0) / max(1, len(base_samples))

            for base_raw in base_samples:
                if self._max_sample_nums and self._get_global_sample_nums() >= self._max_sample_nums:
                    break
                if self.__class__.is_stop_requested():
                    break

                m = self._next_global_sample()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)

                # Fallback if evaluator lacks sync APIs (single-eval guarantee may not hold).
                if not hasattr(chosen_evaluator, "analyse_sync") or not hasattr(chosen_evaluator, "commit_pre_evaluated"):
                    chosen_evaluator.analyse(
                        base_raw,
                        prompt.island_id,
                        prompt.version_generated,
                        **kwargs,
                        global_sample_nums=m,
                        sample_time=sample_time,
                    )
                    continue

                kwargs_stage2 = dict(kwargs)          # keep profiler: stage2 may write sample json
                kwargs_nolog = _drop_profiler(kwargs) # drop profiler: no logging

                # 1) BASE: full stage-2 eval (logged), returns (out_dict, fn)
                base_out, base_fn = chosen_evaluator.analyse_sync(
                    base_raw,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs_stage2,
                    global_sample_nums=m,
                    sample_time=sample_time,
                    stage="stage2",
                    commit=False,
                    return_fn=True,
                    force_filename=(f"samples_{m}_0.json" if self.force_m_n_filenames else None),
                    meta_extra={"sample_order": f"{m}_0", "refine": "base", "stage": "stage2"},
                )

                base_score = float(base_out.get("score", -np.inf))
                base_scores_per_test = dict(base_out.get("scores_per_test", {}) or {})
                critic_json = _maybe_normalize_critic_json(base_out.get("critic_json"))
                base_runs_ok = bool(base_out.get("runs_ok", False))
                logging.info(
                    "Sample %d base_result: runs_ok=%s score=%s tests=%d critic=%s",
                    m,
                    base_runs_ok,
                    f"{base_score:.6g}" if np.isfinite(base_score) else str(base_score),
                    len(base_scores_per_test),
                    "yes" if critic_json is not None else "no",
                )

                if not base_runs_ok or (not base_scores_per_test) or (not np.isfinite(base_score)):
                    reasons = []
                    if not base_runs_ok:
                        reasons.append("runs_ok=False")
                    if not base_scores_per_test:
                        reasons.append("empty_scores_per_test")
                    if not np.isfinite(base_score):
                        reasons.append("non_finite_score")
                    logging.warning("Sample %d dropped at base stage: %s", m, ",".join(reasons))
                    continue

                do_refine = (
                    self.refine_enabled
                    and np.isfinite(base_score)
                    and (base_score > self.refine_score_threshold)
                    and (random.random() < self.refine_prob)
                    and (critic_json is not None)
                )

                if not do_refine:
                    logging.info("Sample %d skip refine: commit base directly", m)
                    chosen_evaluator.commit_pre_evaluated(
                        base_fn,
                        prompt.island_id,
                        base_scores_per_test,
                        **_kwargs_for_buffer_commit(kwargs),
                        global_sample_nums=m,
                        sample_time=sample_time,
                        stage="commit_base",
                        sample_order=f"{m}_0",
                        refine="base",
                    )
                    continue

                # 2) Refine: build prompt and draw K candidates
                base_body = _extract_body(base_raw, self.config)
                refine_prompt = _build_refine_prompt(
                    base_prompt=prompt.code,
                    base_body=base_body,
                    critic_json=critic_json,
                )

                refine_raws = self._llm.draw_samples(refine_prompt, self.config, n=self.refine_k)
                refine_candidates = list(refine_raws)

                # Cheap deduplication by body text
                uniq: list[str] = []
                seen: set[str] = set()
                for cand in refine_candidates:
                    body = _extract_body(cand, self.config).strip()
                    if not body:
                        continue
                    if body in seen:
                        continue
                    seen.add(body)
                    uniq.append(cand)
                refine_candidates = uniq[: self.refine_k]
                logging.info(
                    "Sample %d refine candidates: raw=%d unique_nonempty=%d",
                    m,
                    len(refine_raws),
                    len(refine_candidates),
                )

                if not refine_candidates:
                    logging.info("Sample %d refine empty after filtering: commit base", m)
                    chosen_evaluator.commit_pre_evaluated(
                        base_fn,
                        prompt.island_id,
                        base_scores_per_test,
                        **_kwargs_for_buffer_commit(kwargs),
                        global_sample_nums=m,
                        sample_time=sample_time,
                        stage="commit_base_refine_empty",
                        sample_order=f"{m}_0",
                        refine="base",
                    )
                    continue

                # 3) Stage-1 fast eval (no logging)
                stage1_scored: list[tuple[float, str]] = []
                for cand in refine_candidates:
                    if hasattr(chosen_evaluator, "score_sync"):
                        sres = chosen_evaluator.score_sync(
                            cand,
                            prompt.island_id,
                            prompt.version_generated,
                            **kwargs_nolog,
                            global_sample_nums=m,
                            stage="stage1_fast",
                        )
                    else:
                        # fallback: fast analyse_sync without profiler
                        sres = chosen_evaluator.analyse_sync(
                            cand,
                            prompt.island_id,
                            prompt.version_generated,
                            **kwargs_nolog,
                            global_sample_nums=m,
                            sample_time=sample_time,
                            stage="stage1_fast_fallback",
                            commit=False,
                            fast_eval=True,
                            skip_critic=True,
                            return_fn=False,
                            force_filename=None,
                            meta_extra=None,
                        )
                    s = float(sres.get("score", -np.inf))
                    stage1_scored.append((s, cand))

                stage1_scored.sort(key=lambda x: x[0], reverse=True)
                top1_s, top1_code = stage1_scored[0]
                finalists: list[str] = [top1_code]

                if len(stage1_scored) >= 2:
                    top2_s, top2_code = stage1_scored[1]
                    denom = max(1e-12, abs(top1_s))
                    if denom > 0 and (top1_s - top2_s) / denom < self.refine_eps:
                        finalists.append(top2_code)
                logging.info("Sample %d refine stage1 finalists=%d", m, len(finalists))

                # 4) Finalists: full stage-2 eval (logged)
                best_refine_score = -np.inf
                best_refine_fn = None
                best_refine_scores_per_test: dict[str, float] = {}
                best_refine_n: int | None = None

                for n, code in enumerate(finalists, start=1):
                    fres2, ffn = chosen_evaluator.analyse_sync(
                        code,
                        prompt.island_id,
                        prompt.version_generated,
                        **kwargs_stage2,
                        global_sample_nums=m,
                        sample_time=sample_time,
                        stage="stage2",
                        commit=False,
                        return_fn=True,
                        skip_critic=True,   # no critic on refine finalists at stage-2
                        force_filename=(f"samples_{m}_{n}.json" if self.force_m_n_filenames else None),
                        meta_extra={"sample_order": f"{m}_{n}", "refine": "candidate_stage2", "stage": "stage2"},
                    )


                    s2 = float(fres2.get("score", -np.inf))
                    ok2 = bool(fres2.get("runs_ok", False))
                    scores2 = dict(fres2.get("scores_per_test", {}) or {})
                    if (not ok2) or (not scores2) or (not np.isfinite(s2)):
                        reasons = []
                        if not ok2:
                            reasons.append("runs_ok=False")
                        if not scores2:
                            reasons.append("empty_scores_per_test")
                        if not np.isfinite(s2):
                            reasons.append("non_finite_score")
                        logging.warning("Sample %d refine finalist_%d dropped: %s", m, n, ",".join(reasons))
                        continue

                    if s2 > best_refine_score:
                        best_refine_score = s2
                        best_refine_fn = ffn
                        best_refine_scores_per_test = scores2
                        best_refine_n = n

                # 5) Commit a single winner (no re-eval)
                if best_refine_fn is not None and np.isfinite(best_refine_score) and (best_refine_score > base_score):
                    logging.info(
                        "Sample %d commit winner finalist_%s: refine_score=%.6g > base_score=%.6g",
                        m,
                        str(best_refine_n),
                        best_refine_score,
                        base_score,
                    )
                    chosen_evaluator.commit_pre_evaluated(
                        best_refine_fn,
                        prompt.island_id,
                        best_refine_scores_per_test,
                        **_kwargs_for_buffer_commit(kwargs),
                        global_sample_nums=m,
                        sample_time=sample_time,
                        stage="commit_winner",
                        sample_order=f"{m}_{best_refine_n}",
                        refine="winner",
                    )
                else:
                    logging.info(
                        "Sample %d commit base after refine: best_refine_score=%s base_score=%.6g",
                        m,
                        f"{best_refine_score:.6g}" if np.isfinite(best_refine_score) else str(best_refine_score),
                        base_score,
                    )
                    chosen_evaluator.commit_pre_evaluated(
                        base_fn,
                        prompt.island_id,
                        base_scores_per_test,
                        **_kwargs_for_buffer_commit(kwargs),
                        global_sample_nums=m,
                        sample_time=sample_time,
                        stage="commit_base_after_refine",
                        sample_order=f"{m}_0",
                        refine="base",
                    )


# --------------------------
# API LLM (OpenAI-compatible)
# --------------------------
class ApiLLM(LLM):
    """
    OpenAI-compatible /v1/chat/completions client.
    Designed to be used as class_config.llm_class.
    """

    def __init__(self, samples_per_prompt: int) -> None:
        super().__init__(samples_per_prompt)

        self._session = requests.Session()
        self._session.trust_env = False
        self._prompt_counter = 0

        self._instruction_prompt = (
            "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
            "Complete the 'equation' function body below, considering physical meaning and relationships of inputs. "
            "Be diverse but parsimonious.\n"
        )

    def draw_samples(
        self,
        prompt: str,
        config: config_lib.Config,
        *,
        n: int | None = None,
    ) -> Collection[str]:
        return self._draw_samples_api(prompt, config, n=n)

    def _draw_samples_api(
        self,
        prompt: str,
        config: config_lib.Config,
        *,
        n: int | None = None,
    ) -> Collection[str]:
        self._prompt_counter += 1
        instruction = self._instruction_prompt
        hint_text = getattr(config, "data_hint_text", None)
        hint_enabled = bool(getattr(config, "data_hint_enabled", False))
        hint_every = int(getattr(config, "data_hint_every", 0))
        if hint_enabled and hint_text:
            if hint_every <= 0 or self._prompt_counter == 1 or (self._prompt_counter % hint_every == 0):
                instruction = "\n".join([instruction, hint_text])
        prompt = "\n".join([instruction, prompt])

        api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("No API key found. Set OPENAI_API_KEY or DEEPSEEK_API_KEY.")

        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://4zapi.com/v1"
        ).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        endpoint = f"{base_url}/chat/completions"

        model = getattr(config, "api_model", "gpt-5.1")

        req_n = int(n if n is not None else self._samples_per_prompt)
        req_n = max(1, req_n)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(getattr(config, "api_max_tokens", 4800)),
            "temperature": float(getattr(config, "api_temperature", 0.8)),
            "n": req_n,
        }

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        max_tries = int(getattr(config, "api_max_tries", 8))
        last_err: Exception | None = None

        req_t0 = time.time()
        logging.info(
            "LLM request start: model=%s n=%d prompt_chars=%d",
            model,
            req_n,
            len(prompt),
        )
        for attempt in range(max_tries):
            try:
                resp = self._session.post(endpoint, headers=headers, json=payload, timeout=(10, 120))
                sc = resp.status_code

                # non-retriable client errors
                if sc in (400, 401, 403, 404):
                    raise RuntimeError(f"HTTP {sc} (non-retriable). Body={resp.text[:800]}")

                # rate limit
                if sc == 429:
                    ra = resp.headers.get("Retry-After")
                    sleep_s = float(ra) if ra else min(60.0, (2 ** attempt) * 0.8) + random.random() * 0.4
                    logging.warning(
                        "LLM API 429 rate-limited. attempt=%d sleep=%.2fs body=%s",
                        attempt + 1, sleep_s, resp.text[:200]
                    )
                    time.sleep(sleep_s)
                    continue

                # transient server errors
                if 500 <= sc < 600:
                    sleep_s = min(30.0, (2 ** attempt) * 0.5) + random.random() * 0.3
                    logging.warning(
                        "LLM API %d server error. attempt=%d sleep=%.2fs body=%s",
                        sc, attempt + 1, sleep_s, resp.text[:200]
                    )
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                data = resp.json()

                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"Empty choices. HTTP {sc}. Body={resp.text[:800]}")

                texts = []
                for c in choices:
                    msg = c.get("message", {})
                    if isinstance(msg, dict) and "content" in msg:
                        texts.append(msg["content"])

                if not texts:
                    raise RuntimeError(f"No message content. HTTP {sc}. Body={resp.text[:800]}")

                if "gemini" in str(model).lower():
                    raw_preview = str(texts[0])[:300].replace("\n", "\\n")
                    logging.info("Gemini raw preview (first 300 chars): %s", raw_preview)

                texts = [_extract_body(t, config) for t in texts]
                logging.info(
                    "LLM request done: model=%s choices=%d returned=%d elapsed=%.2fs",
                    model,
                    len(choices),
                    len(texts[:req_n]),
                    time.time() - req_t0,
                )
                return texts[:req_n]

            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                sleep_s = min(20.0, 1.0 + (2 ** attempt) * 0.5) + random.random() * 0.5
                logging.warning("LLM API network error: %r attempt=%d sleep=%.2fs", e, attempt + 1, sleep_s)
                time.sleep(sleep_s)
                continue

            except Exception as e:
                last_err = e
                sleep_s = min(10.0, 1.0 + attempt * 1.2)
                logging.warning("LLM API error: %r attempt=%d sleep=%.2fs", e, attempt + 1, sleep_s)
                time.sleep(sleep_s)
                continue

        raise RuntimeError(f"LLM API failed after {max_tries} retries; last_err={last_err!r}") from last_err
