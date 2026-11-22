"""Multi-provider model adapter supporting Google Gemini and direct OpenAI APIs.

Provides a unified interface for text generation and Pydantic structured output,
enabling cross-provider benchmark evaluations across Gemini and OpenAI models.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


@dataclass
class ModelResponse:
    """Standardized response from any model provider."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    provider: str
    raw_response: Any = None


class UnifiedModelClient:
    """Multi-provider LLM client decoupling high-level agent logic from provider SDKs."""

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        dotenv_path: Path | None = None,
    ) -> None:
        self.provider = provider.lower()
        if dotenv_path and dotenv_path.exists():
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        if self.provider == "gemini":
            from google import genai

            api_key = os.getenv("GEMINI_API_KEY")
            self.model = model or "gemini-3.5-flash-lite"
            self._client = genai.Client(api_key=api_key)

        elif self.provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4o-mini"
            self._client = OpenAI(api_key=api_key)

        else:
            msg = f"Unsupported provider '{provider}'. Supported: ['gemini', 'openai']"
            raise ValueError(msg)

    def generate_text(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """Execute text completion and return normalized metrics."""
        t0 = time.perf_counter()

        if self.provider == "gemini":
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            if system_instruction:
                config.system_instruction = system_instruction

            resp = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            latency = (time.perf_counter() - t0) * 1000
            txt = resp.text or ""
            usage = getattr(resp, "usage_metadata", None)
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0

            return ModelResponse(
                text=txt,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=round(latency, 2),
                model=self.model,
                provider="gemini",
                raw_response=resp,
            )

        elif self.provider == "openai":
            messages: list[dict[str, str]] = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = (time.perf_counter() - t0) * 1000
            choice = resp.choices[0]
            txt = choice.message.content or ""
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) or 0
            out_tok = getattr(usage, "completion_tokens", 0) or 0

            return ModelResponse(
                text=txt,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=round(latency, 2),
                model=self.model,
                provider="openai",
                raw_response=resp,
            )

        raise ValueError(f"Unknown provider {self.provider}")

    def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_instruction: str = "",
        temperature: float = 0.0,
    ) -> tuple[T, ModelResponse]:
        """Generate structured JSON adhering to a Pydantic schema."""
        if self.provider == "openai":
            # Direct OpenAI structured output via response_format Pydantic schema
            t0 = time.perf_counter()
            messages: list[dict[str, str]] = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            completion = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_schema,
                temperature=temperature,
            )
            latency = (time.perf_counter() - t0) * 1000
            parsed_obj = completion.choices[0].message.parsed
            usage = getattr(completion, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) or 0
            out_tok = getattr(usage, "completion_tokens", 0) or 0

            resp_meta = ModelResponse(
                text=json.dumps(parsed_obj.model_dump() if parsed_obj else {}),
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=round(latency, 2),
                model=self.model,
                provider="openai",
                raw_response=completion,
            )
            return parsed_obj, resp_meta

        # Fallback to schema-guided JSON prompting for providers or generic models
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        augmented_prompt = (
            f"{prompt}\n\nYou must reply ONLY with a valid JSON object conforming to this schema:\n"
            f"{schema_json}"
        )
        base_resp = self.generate_text(
            prompt=augmented_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        clean_text = base_resp.text.strip()
        match = JSON_BLOCK_RE.search(clean_text)
        payload_str = match.group(1) if match else clean_text
        parsed_data = json.loads(payload_str)
        return response_schema.model_validate(parsed_data), base_resp
