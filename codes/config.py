"""Configuration of a LLMSR experiments."""
from __future__ import annotations

import dataclasses
from typing import Type

from codes.sample import sampler
from codes.evaluate import evaluator  # for ClassConfig typing


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.

    Args:
        functions_per_prompt (int): Number of previous hypotheses to include in prompts
        num_islands (int): Number of islands in experience buffer for diversity
        reset_period (int): Seconds between weakest island resets
        cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
        cluster_sampling_temperature_period (int): Period for temperature decay
        use_tfidf_for_clustering (bool): If True, cluster by code similarity (TF-IDF); if False, cluster by score only (original LLMSR style).
    """
    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 10_000
    use_tfidf_for_clustering: bool = True


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for LLMSR experiments."""

    # -------------------------
    # Base system settings
    # -------------------------
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 4

    # Stage-2 (full) evaluation
    evaluate_timeout_seconds: int = 120

    # LLM backend
    use_api: bool = True
    api_model: str = "gpt-5.1"

    # -------------------------
    # Optional data-hint injection
    # -------------------------
    data_hint_text: str | None = None
    data_hint_enabled: bool = True
    data_hint_every: int = 50  # <=0 means always; otherwise every N prompts

    # -------------------------
    # Critic / Refine loop knobs
    # -------------------------
    critic_prob: float = 0.4
    critic_score_threshold: float = 0

    refine_enabled: bool = True
    refine_k: int = 4
    refine_eps: float = 0.01  # 1% by default; keep even if you currently select only 1 winner
    refine_diversity_threshold: float = 0.92  # stricter than main TF-IDF threshold (e.g., 0.85)

    # -------------------------
    # Stage-1 fast evaluation knobs
    # -------------------------
    fast_eval_timeout_seconds: int = 40
    fast_eval_num_restarts: int = 1
    fast_eval_num_splits: int = 1  # if your evaluator supports "splits"; otherwise keep but unused

    # -------------------------
    # Early stop (save budget when fit is essentially perfect on train)
    # -------------------------
    # If evaluate() returns a tuple whose last element is train NMSE (float), stop all samplers
    # when train_nmse < threshold, even if max_sample_nums not reached. None disables.
    early_stop_train_nmse_threshold: float | None = 1e-13


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.ApiLLM]
    sandbox_class: Type[evaluator.Sandbox]
