"""Unit tests for P0 configuration, budget guardrails, secret hygiene, and schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from invoiceops.config import DEFAULT_CONFIG_PATH, AppConfig, load_config, load_runtime_secrets
from invoiceops.schemas import DecisionOutput, EvidenceItem


def test_default_config_loads_and_matches_p0_guardrails() -> None:
    """Verify configs/config.yaml validates cleanly with approved P0 settings."""
    cfg = load_config()
    assert cfg.config_version == "p0.1.0"
    assert cfg.platform.region == "us-central1"
    assert cfg.platform.mode == "local_ai_studio"
    assert cfg.budget_guardrails.p1_p3_budget_usd == 0.0
    assert cfg.budget_guardrails.enforce_free_tier_models_only is True
    assert cfg.budget_guardrails.p4_budget_alert_thresholds_percent == [50, 80, 100]
    assert cfg.evaluation.runs_per_experiment >= 3


def test_free_tier_guardrail_blocks_paid_only_active_model() -> None:
    """Ensure Paid-only models (e.g. gemini-4-carbon) fail fast in zero-cost mode."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["models"]["orchestrator"] = "gemini-4-carbon"
    with pytest.raises(ValidationError, match="Paid-only model"):
        AppConfig.model_validate(raw)


def test_floating_latest_alias_is_rejected() -> None:
    """Ensure floating `-latest` aliases are rejected for reproducibility."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["models"]["extraction"] = "gemini-flash-latest"
    with pytest.raises(ValidationError, match="-latest"):
        AppConfig.model_validate(raw)


def test_deprecated_gemini_25_models_are_rejected() -> None:
    """Ensure deprecated Gemini 2.5 Pro/Flash models cannot be configured."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["models"]["v0_baseline"] = "gemini-2.5-flash"
    with pytest.raises(ValidationError, match=r"Deprecated Gemini 2\.5"):
        AppConfig.model_validate(raw)


def test_shadow_cost_estimation_arithmetic() -> None:
    """Verify deterministic token cost calculation against published pricing."""
    cfg = load_config()
    cost = cfg.estimate_call_cost_usd(
        model_id="gemini-3.8-flash",
        input_tokens=10_000,
        output_tokens=2_000,
    )
    # 10k * $0.75/1M + 2k * $3.75/1M = $0.0075 + $0.0075 = $0.015
    assert cost == pytest.approx(0.015)


def test_runtime_secrets_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure API keys never appear in string representations."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    fake_env = tmp_path / ".env"
    fake_env.write_text("GEMINI_API_KEY=super_secret_test_value_123\n", encoding="utf-8")

    secrets = load_runtime_secrets(dotenv_path=fake_env)
    assert secrets.has_gemini_key() is True
    assert "super_secret_test_value_123" not in repr(secrets)
    assert "super_secret_test_value_123" not in str(secrets)


def test_decision_output_schema_validation() -> None:
    """Verify Section 2 structured output contract validation."""
    decision = DecisionOutput(
        decision="hold",
        reason_codes=["PRICE_VARIANCE_OVER_TOLERANCE"],
        evidence=[
            EvidenceItem(type="policy_clause", id="POL-07#3"),
            EvidenceItem(type="erp_record", id="PO-1042"),
        ],
        confidence=0.92,
    )
    assert decision.decision == "hold"
    assert len(decision.evidence) == 2

    with pytest.raises(ValidationError):
        DecisionOutput(decision="approve", confidence=1.5)
