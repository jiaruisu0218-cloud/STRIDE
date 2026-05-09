import os
import json
import time
import random
import requests
from typing import Any


class PhysicsCritic:
    """
    Critic that returns ONLY structured JSON actions for refinement.

    - No str / natural-language critique output is kept.
    - On failure, returns a minimal JSON with empty actions.
    """

    def __init__(self, config: Any):
        # API key: OpenAI-compatible
        self.api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()

        # Base URL
        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.deepseek.com/v1"
        ).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self.base_url = base_url

        # Model
        self.model = getattr(config, "api_model", "gpt-5.1")

        # HTTP session
        self._session = requests.Session()
        self._session.trust_env = False

    # -------------------------
    # Internal helpers
    # -------------------------
    def _call_chat_completions(self, payload: dict) -> str:
        """
        Call OpenAI-compatible /chat/completions and return assistant content.
        Retries on 429 / transient request errors.
        """
        if not self.api_key:
            raise RuntimeError("No API key found. Set OPENAI_API_KEY or DEEPSEEK_API_KEY.")

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err = None
        for attempt in range(5):
            try:
                resp = self._session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(10, 60),
                )

                if resp.status_code == 401:
                    raise RuntimeError("401 Unauthorized: check API key / base_url")

                if resp.status_code == 429:
                    sleep_s = min(30.0, (2 ** attempt) * 0.8) + random.random() * 0.3
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.RequestException as e:
                last_err = e
                time.sleep(1.2 + attempt * 0.6)

        raise RuntimeError(f"Critic API failed after retries: {last_err}")

    @staticmethod
    def _parse_json_loose(text: str) -> dict:
        """
        Parse JSON from model output, stripping common code-fence wrappers.
        """
        s = (text or "").strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

        return json.loads(s)

    @staticmethod
    def _validate_actions(obj: dict) -> dict:
        """
        Enforce:
          - obj is dict
          - obj['actions'] is list of dict
          - each action has: type, rationale
          - REMOVE/SIMPLIFY require target; ADD_OR_REPLACE require proposal
        """
        if not isinstance(obj, dict):
            return {"actions": [], "short_comment": "Invalid critic output (not a JSON object)."}

        actions = obj.get("actions", [])
        if not isinstance(actions, list):
            actions = []

        valid = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            t = a.get("type", "")
            r = a.get("rationale", "")
            if not t or not r:
                continue

            t = str(t).strip().upper()
            if t not in {"REMOVE", "SIMPLIFY", "ADD_OR_REPLACE"}:
                continue

            if t in {"REMOVE", "SIMPLIFY"}:
                if "target" not in a or not str(a.get("target", "")).strip():
                    continue
                valid.append(
                    {
                        "type": t,
                        "target": str(a["target"]).strip(),
                        "rationale": str(r).strip(),
                    }
                )
            else:
                if "proposal" not in a or not str(a.get("proposal", "")).strip():
                    continue
                valid.append(
                    {
                        "type": t,
                        "proposal": str(a["proposal"]).strip(),
                        "rationale": str(r).strip(),
                    }
                )

        short_comment = obj.get("short_comment", "")
        if not isinstance(short_comment, str) or not short_comment.strip():
            short_comment = "No short_comment provided."

        # Keep other fields if you later want to extend schema
        out = dict(obj)
        out["actions"] = valid
        out["short_comment"] = short_comment.strip()
        return out

    # -------------------------
    # Public API
    # -------------------------
    def propose_actions(
        self,
        code: str,
        params: list,
        input_vars: list,
        problem_description: str,
        metrics: dict | None = None,
    ) -> dict:
        """
        Return structured edit actions for refinement (JSON only).
        No natural-language critique fallback.
        """
        # 1) format params (cap to avoid token explosion)
        display_params = (params or [])[:20]
        param_str = "\n".join([f"params[{i}] = {val:.4g}" for i, val in enumerate(display_params)])
        if params and len(params) > 20:
            param_str += "\n... (more parameters hidden) ..."

        # 2) metrics
        metrics_str = ""
        if metrics:
            try:
                metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            except Exception:
                metrics_str = "N/A"

        # 3) prompt (JSON-only)
        prompt = f"""
You are a scientific critic for symbolic regression.
Your ONLY job is to output a JSON object that proposes edit actions to refine the equation.
Do NOT output any markdown, code fences, or explanations outside JSON.

--- SYSTEM DESCRIPTION ---
{problem_description}

--- CANDIDATE CODE ---
{code}

--- OPTIMIZED PARAMETERS (truncated) ---
{param_str}

--- INPUT VARIABLES ---
{input_vars}

--- METRICS ---
{metrics_str if metrics_str else "N/A"}

--- ACTION TYPES ---
- REMOVE: delete redundant term or near-zero-effect component
- SIMPLIFY: algebraic simplification / reduce nesting / lower degree
- ADD_OR_REPLACE: add a missing term or replace with a more plausible mechanism term

--- OUTPUT JSON SCHEMA (exact keys) ---
{{
  "actions": [
    {{"type":"REMOVE","target":"...","rationale":"..."}},
    {{"type":"SIMPLIFY","target":"...","rationale":"..."}},
    {{"type":"ADD_OR_REPLACE","proposal":"...","rationale":"..."}}
  ],
  "short_comment":"one sentence summary"
}}

Rules:
- actions can be empty
- rationale must be specific, feasible, and concise, with fewer than 20 words.
- Output ONLY valid JSON
- Actions and reasons should be concise and to the point, without excessive explanation!
""".strip()

        # 4) payload
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        # Prefer JSON mode if supported; if unsupported, server may ignore it.
        payload["response_format"] = {"type": "json_object"}

        # 5) call + parse
        raw = ""
        try:
            raw = self._call_chat_completions(payload)
            obj = self._parse_json_loose(raw)
            return self._validate_actions(obj)

        except Exception as e:
            # JSON-only failure fallback (no str critique)
            return {
                "actions": [],
                "short_comment": "Critic failed (JSON-only mode).",
                "error": str(e),
                "raw": raw[:2000] if isinstance(raw, str) else "",
            }
