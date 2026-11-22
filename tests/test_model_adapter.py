"""Unit tests for UnifiedModelClient multi-provider adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

pytest.importorskip("openai")

from invoiceops.models.adapter import UnifiedModelClient


class SampleDecision(BaseModel):
    decision: str
    confidence: float
    reason: str


class TestUnifiedModelClient:
    """Test suite for UnifiedModelClient with mocked provider SDKs."""

    def test_unsupported_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported provider 'anthropic'"):
            UnifiedModelClient(provider="anthropic")

    @patch("google.genai.Client")
    def test_gemini_generate_text_mocked(self, mock_genai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_genai_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = "Approved match"
        mock_resp.usage_metadata.prompt_token_count = 150
        mock_resp.usage_metadata.candidates_token_count = 25
        mock_client.models.generate_content.return_value = mock_resp

        client = UnifiedModelClient(provider="gemini", model="gemini-3.5-flash-lite")
        resp = client.generate_text("Evaluate invoice INV-100")

        assert resp.text == "Approved match"
        assert resp.input_tokens == 150
        assert resp.output_tokens == 25
        assert resp.provider == "gemini"
        assert resp.model == "gemini-3.5-flash-lite"
        assert resp.latency_ms > 0

    @patch("openai.OpenAI")
    def test_openai_generate_text_mocked(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Reject: variance exceeds 2%"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 210
        mock_resp.usage.completion_tokens = 40
        mock_client.chat.completions.create.return_value = mock_resp

        client = UnifiedModelClient(provider="openai", model="gpt-4o-mini")
        resp = client.generate_text("Check PO-500", system_instruction="You are an AP auditor.")

        assert resp.text == "Reject: variance exceeds 2%"
        assert resp.input_tokens == 210
        assert resp.output_tokens == 40
        assert resp.provider == "openai"
        assert resp.model == "gpt-4o-mini"
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_openai_generate_structured_mocked(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_parsed = SampleDecision(
            decision="approve",
            confidence=0.95,
            reason="Within 2% price tolerance",
        )
        mock_choice = MagicMock()
        mock_choice.message.parsed = mock_parsed
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 300
        mock_completion.usage.completion_tokens = 50
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        client = UnifiedModelClient(provider="openai", model="gpt-4o-mini")
        result, meta = client.generate_structured(
            prompt="Evaluate price mismatch",
            response_schema=SampleDecision,
        )

        assert isinstance(result, SampleDecision)
        assert result.decision == "approve"
        assert result.confidence == 0.95
        assert meta.provider == "openai"
        assert meta.input_tokens == 300
