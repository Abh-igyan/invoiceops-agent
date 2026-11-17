"""Typed configuration loader and budget/model guardrails for invoiceops-agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

ACTIVE_MODEL_FIELDS: tuple[str, ...] = (
    "v0_baseline",
    "orchestrator",
    "extraction",
    "policy_rag",
    "critic",
    "fast_router",
    "llm_judge",
    "embedding",
    "embedding_text_fallback",
)


class PlatformConfig(BaseModel):
    """Execution backend and cloud location settings."""

    mode: Literal["local_ai_studio", "gcp_enterprise"]
    gcp_project_id: str | None = None
    region: str = Field(min_length=2)
    use_enterprise_agent_platform: bool = False


class BudgetGuardrailsConfig(BaseModel):
    """Spend caps and alert thresholds across local and cloud phases."""

    p1_p3_budget_usd: float = Field(ge=0.0)
    enforce_free_tier_models_only: bool = True
    max_single_eval_estimated_usd: float = Field(gt=0.0)
    p4_monthly_budget_usd: float = Field(gt=0.0)
    p4_budget_alert_thresholds_percent: list[int]


class GenerationDefaults(BaseModel):
    """Default generation parameters (temperature=0.0 for deterministic evals)."""

    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(gt=0)


class ModelRate(BaseModel):
    """Published token pricing (USD per 1M tokens) and Free-Tier eligibility."""

    free_tier_eligible: bool
    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)
    cached_input: float = Field(ge=0.0)


class ModelsConfig(BaseModel):
    """Explicit Gemini model IDs, fallback chain, and published pricing table."""

    v0_baseline: str
    orchestrator: str
    extraction: str
    policy_rag: str
    critic: str
    fast_router: str
    llm_judge: str
    embedding: str
    embedding_text_fallback: str
    paid_frontier_reference: str
    fallback_chain: list[str] = Field(default_factory=list)
    generation_defaults: GenerationDefaults
    pricing_usd_per_1m_tokens: dict[str, ModelRate]

    @model_validator(mode="after")
    def reject_latest_aliases_and_deprecated_models(self) -> ModelsConfig:
        """Disallow `-latest` aliases and deprecated Gemini 2.5 model IDs."""
        all_configured = (
            [getattr(self, field) for field in ACTIVE_MODEL_FIELDS]
            + list(self.fallback_chain)
            + [self.paid_frontier_reference]
        )
        for model_id in all_configured:
            if model_id.endswith("-latest"):
                msg = f"Floating '-latest' alias is prohibited: {model_id}"
                raise ValueError(msg)
            if model_id.startswith(("gemini-2.5-flash", "gemini-2.5-pro")) and model_id not in {
                "gemini-2.5-flash-lite"
            }:
                msg = f"Deprecated Gemini 2.5 model is prohibited: {model_id}"
                raise ValueError(msg)
            if model_id not in self.pricing_usd_per_1m_tokens:
                msg = f"Model '{model_id}' is missing from pricing_usd_per_1m_tokens"
                raise ValueError(msg)
        return self


class MatchingAndHitlConfig(BaseModel):
    """Deterministic tolerance rules and human-in-the-loop thresholds."""

    price_variance_tolerance_pct: float = Field(ge=0.0)
    price_reject_threshold_pct: float = Field(default=0.10, ge=0.0)
    quantity_variance_tolerance_pct: float = Field(ge=0.0)
    tax_rate_tolerance_abs: float = Field(ge=0.0)
    late_submission_max_days: int = Field(default=60, gt=0)
    spot_buy_no_po_limit_usd: float = Field(default=500.0, ge=0.0)
    human_approval_amount_threshold_usd: float = Field(gt=0.0)
    human_approval_min_confidence: float = Field(ge=0.0, le=1.0)
    max_extraction_schema_retries: int = Field(ge=0)
    max_critic_sendbacks: int = Field(ge=0)


class StorageConfig(BaseModel):
    """Dual-mode vector and evaluation storage settings."""

    backend: Literal["local_sqlite", "bigquery"]
    local_sqlite_path: str
    local_eval_runs_jsonl: str
    bigquery_dataset: str
    bigquery_policy_table: str
    bigquery_eval_runs_table: str
    top_k_clauses: int = Field(gt=0)
    naive_chunk_chars: int = Field(default=420, gt=50)
    naive_chunk_overlap_chars: int = Field(default=60, ge=0)


class MockErpConfig(BaseModel):
    """Legacy ERP MCP server connection and resilience parameters."""

    topology: Literal["streamable_http_mcp", "stdio_mcp"]
    base_url: str
    require_cloud_run_iam_auth: bool = False
    flaky_mode: bool = False
    flaky_timeout_rate: float = Field(default=0.12, ge=0.0, le=1.0)
    flaky_rate_limit_429_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    flaky_stale_read_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    client_timeout_seconds: float = Field(gt=0.0)
    max_retries: int = Field(ge=0)
    circuit_breaker_failure_threshold: int = Field(gt=0)
    circuit_breaker_recovery_seconds: float = Field(gt=0.0)


class ErrorCostWeights(BaseModel):
    """Asymmetric error penalties for accounts-payable decisions."""

    false_approval: float = Field(gt=0.0)
    false_rejection: float = Field(gt=0.0)
    false_hold: float = Field(gt=0.0)
    false_escalation: float = Field(gt=0.0)


class EvaluationConfig(BaseModel):
    """Dataset split, repetition count, and cost-weighted scoring settings."""

    random_seed: int
    dev_split_ratio: float = Field(gt=0.0, lt=1.0)
    test_split_ratio: float = Field(gt=0.0, lt=1.0)
    runs_per_experiment: int = Field(ge=3)
    default_dev_subset: str = "dev_stratified_24"
    max_concurrency: int = Field(default=2, ge=1, le=16)
    inter_case_delay_seconds: float = Field(default=0.2, ge=0.0)
    judge_min_kappa_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    error_cost_weights: ErrorCostWeights

    @model_validator(mode="after")
    def check_splits_sum_to_one(self) -> EvaluationConfig:
        """Ensure dev and frozen test ratios sum to 1.0."""
        total = round(self.dev_split_ratio + self.test_split_ratio, 6)
        if total != 1.0:
            msg = f"dev_split_ratio + test_split_ratio must equal 1.0, got {total}"
            raise ValueError(msg)
        return self


class AppConfig(BaseModel):
    """Root validated application configuration."""

    config_version: str
    project_name: str
    platform: PlatformConfig
    budget_guardrails: BudgetGuardrailsConfig
    models: ModelsConfig
    matching_and_hitl: MatchingAndHitlConfig
    storage: StorageConfig
    mock_erp: MockErpConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def enforce_zero_cost_guardrails(self) -> AppConfig:
        """Reject non-Free-Tier active or fallback models when zero-cost guardrail is enabled."""
        if self.budget_guardrails.enforce_free_tier_models_only:
            check_models = [(field, getattr(self.models, field)) for field in ACTIVE_MODEL_FIELDS]
            check_models.extend(("fallback_chain", m) for m in self.models.fallback_chain)
            for label, model_id in check_models:
                rate = self.models.pricing_usd_per_1m_tokens[model_id]
                if not rate.free_tier_eligible:
                    msg = (
                        f"Active role '{label}' uses Paid-only model '{model_id}' while "
                        "budget_guardrails.enforce_free_tier_models_only is enabled."
                    )
                    raise ValueError(msg)
        return self

    def estimate_call_cost_usd(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Compute shadow Paid-Tier USD cost for telemetry and experiment comparison."""
        rate = self.models.pricing_usd_per_1m_tokens[model_id]
        uncached_input = max(0, input_tokens - cached_input_tokens)
        return (
            (uncached_input * rate.input)
            + (cached_input_tokens * rate.cached_input)
            + (output_tokens * rate.output)
        ) / 1_000_000.0


class RuntimeSecrets(BaseModel):
    """Redacted container for runtime API credentials loaded from environment."""

    gemini_api_key: SecretStr | None = None

    def has_gemini_key(self) -> bool:
        """Return True when a non-placeholder Gemini API key is configured."""
        if self.gemini_api_key is None:
            return False
        raw = self.gemini_api_key.get_secret_value().strip()
        return bool(raw) and raw != "your_gemini_api_key_here"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate YAML configuration from disk."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle)
    return AppConfig.model_validate(raw_data)


def load_runtime_secrets(dotenv_path: Path | str | None = None) -> RuntimeSecrets:
    """Load credentials from environment / `.env` into redacted `SecretStr` fields."""
    load_dotenv(dotenv_path=dotenv_path, override=False)
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return RuntimeSecrets(gemini_api_key=SecretStr(key) if key else None)
