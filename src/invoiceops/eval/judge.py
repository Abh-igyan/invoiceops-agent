"""Layer 5 Calibrated LLM-as-Judge for explanation quality (agreement + Cohen's Kappa)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from invoiceops.config import AppConfig, load_runtime_secrets

JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

JUDGE_SYSTEM_PROMPT = """You are a strict Accounts Payable Audit Explanation Judge.
For each item in the input JSON array, grade `candidate_explanation` as `1` (PASS) or `0` (FAIL):
- Score `1` ONLY IF the explanation states a clear outcome AND cites at least one concrete
  policy clause ID (`POL-...`) or ERP record ID (`PO-...`, `VEN-...`, `GRN-...`, `HIST-...`)
  supporting the decision without vague hand-waving.
- Score `0` if it omits concrete IDs, relies on "common sense", or is empty/generic.

Reply ONLY with a JSON array of objects: `[{"id": "...", "score": 0 or 1}]`.
"""


def compute_cohens_kappa(y_true: list[int], y_pred: list[int]) -> tuple[float, float]:
    """Compute raw agreement rate and binary Cohen's Kappa (`[-1.0, 1.0]`)."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0, 0.0
    n = len(y_true)
    agree = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b)
    p_o = agree / n

    p_true_1 = sum(y_true) / n
    p_pred_1 = sum(y_pred) / n
    p_e = (p_true_1 * p_pred_1) + ((1.0 - p_true_1) * (1.0 - p_pred_1))

    if abs(1.0 - p_e) < 1e-9:
        return round(p_o, 4), 1.0 if p_o == 1.0 else 0.0
    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(p_o, 4), round(kappa, 4)


class ExplanationJudge:
    """Evaluates free-text audit explanations and calibrates against human rubric labels."""

    def __init__(self, config: AppConfig, dotenv_path: Path | None = None) -> None:
        self.config = config
        secrets = load_runtime_secrets(dotenv_path)
        key = secrets.gemini_api_key.get_secret_value() if secrets.gemini_api_key else None
        self._client = genai.Client(api_key=key)
        self._models = [config.models.llm_judge, *config.models.fallback_chain]

    def score_batch(self, items: list[dict[str, str]]) -> dict[str, int]:
        """Grade a list of `{"id": ..., "candidate_explanation": ...}` items in batches of 20."""
        scores: dict[str, int] = {}
        batch_size = 20
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            prompt = json.dumps(chunk, ensure_ascii=True)
            parsed = self._call_judge_json(prompt)
            for entry in parsed:
                item_id = str(entry.get("id", ""))
                val = 1 if int(entry.get("score", 0)) == 1 else 0
                if item_id:
                    scores[item_id] = val
        return scores

    def calibrate(self, calibration_path: Path) -> dict[str, Any]:
        """Run judge on the 40-item hand-labeled set and report agreement & Cohen's Kappa."""
        raw_items: list[dict[str, Any]] = json.loads(calibration_path.read_text(encoding="utf-8"))
        judge_inputs = [
            {
                "id": item["calibration_id"],
                "decision": item["decision"],
                "candidate_explanation": item["candidate_explanation"],
            }
            for item in raw_items
        ]
        pred_map = self.score_batch(judge_inputs)
        y_true = [int(item["human_rubric_pass"]) for item in raw_items]
        y_pred = [int(pred_map.get(item["calibration_id"], 0)) for item in raw_items]
        agreement, kappa = compute_cohens_kappa(y_true, y_pred)
        reliable = kappa >= self.config.evaluation.judge_min_kappa_threshold
        return {
            "calibration_cases": len(raw_items),
            "raw_agreement": agreement,
            "cohens_kappa": kappa,
            "min_kappa_threshold": self.config.evaluation.judge_min_kappa_threshold,
            "reliable_for_reporting": reliable,
        }

    def _call_judge_json(self, user_json: str) -> list[dict[str, Any]]:
        for model_id in self._models:
            try:
                resp = self._client.models.generate_content(
                    model=model_id,
                    contents=user_json,
                    config=types.GenerateContentConfig(
                        system_instruction=JUDGE_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=1024,
                    ),
                )
                txt = resp.text or "[]"
                match = JSON_ARRAY_RE.search(txt)
                payload = json.loads(match.group(0) if match else txt)
                if isinstance(payload, list):
                    return payload
            except Exception:
                continue
        return []
