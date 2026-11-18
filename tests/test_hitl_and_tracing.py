"""Unit tests for ADK Tool Confirmation HITL state preservation and OpenTelemetry tracing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from google.adk.events.event_actions import EventActions
from google.adk.tools.tool_confirmation import ToolConfirmation

from invoiceops.config import load_config
from invoiceops.telemetry.tracing import InvoiceOpsTelemetry
from invoiceops.tools.hitl import request_human_approval


class _FakeToolContext:
    """Minimal deterministic double matching ADK 2.14 `ToolContext` confirmation contract."""

    def __init__(
        self,
        state: dict[str, Any],
        tool_confirmation: ToolConfirmation | None = None,
        function_call_id: str = "fc-hitl-001",
    ) -> None:
        self.state = state
        self.tool_confirmation = tool_confirmation
        self.function_call_id = function_call_id
        self._event_actions = EventActions()

    def request_confirmation(self, *, hint: str | None = None, payload: Any | None = None) -> None:
        self._event_actions.requested_tool_confirmations[self.function_call_id] = ToolConfirmation(
            hint=hint or "",
            payload=payload,
        )


def test_adk_hitl_pause_preserves_state_and_resumes_on_confirmation() -> None:
    """Verify HITL tool pauses, persists state, and finalizes approve/reject upon resumption."""
    shared_session_state: dict[str, Any] = {}

    # Turn 1: Unconfirmed high-value invoice -> pauses and records confirmation request
    ctx_pause = _FakeToolContext(state=shared_session_state, tool_confirmation=None)
    pause_res = request_human_approval(
        invoice_id="INV-4350",
        vendor_id="VEN-010",
        total_amount=16500.0,
        currency="USD",
        reason_code="HIGH_VALUE_APPROVAL_REQUIRED",
        confidence=0.94,
        tool_context=ctx_pause,  # type: ignore[arg-type]
    )
    assert pause_res["status"] == "PAUSED_AWAITING_HUMAN_CONFIRMATION"
    assert pause_res["decision"] == "escalate_to_human"
    assert "fc-hitl-001" in ctx_pause._event_actions.requested_tool_confirmations
    assert shared_session_state["pending_hitl_decision"]["total_amount"] == 16500.0

    # Turn 2: Human approves -> state preserved across pause is read and finalized
    ctx_confirm = _FakeToolContext(
        state=shared_session_state,
        tool_confirmation=ToolConfirmation(confirmed=True),
    )
    confirm_res = request_human_approval(
        invoice_id="INV-4350",
        vendor_id="VEN-010",
        total_amount=16500.0,
        currency="USD",
        reason_code="HIGH_VALUE_APPROVAL_REQUIRED",
        confidence=0.94,
        tool_context=ctx_confirm,  # type: ignore[arg-type]
    )
    assert confirm_res["status"] == "CONFIRMED_BY_HUMAN"
    assert confirm_res["decision"] == "approve"
    assert shared_session_state["hitl_resolution"] == "APPROVED_BY_HUMAN"


def test_opentelemetry_parent_child_spans_and_jsonl_export(tmp_path: Path) -> None:
    """Verify OTel root request span, child agent/tool spans, and JSONL attributes."""
    cfg = load_config()
    trace_file = tmp_path / "otel_spans.jsonl"
    telemetry = InvoiceOpsTelemetry(config=cfg, jsonl_trace_path=trace_file, keep_in_memory=True)

    with telemetry.request_span(case_id="INV-4002", vendor_id="VEN-002", version="v1") as req_sp:
        with telemetry.agent_span("extraction_agent", "gemini-3.5-flash-lite") as ag_sp:
            telemetry.record_llm_metrics_on_span(
                span=ag_sp,
                resolved_model="gemini-3.5-flash-lite",
                input_tokens=1200,
                output_tokens=180,
                estimated_cost_usd=0.00081,
                llm_calls=1,
            )
        with telemetry.tool_span("lookup_purchase_order") as tl_sp:
            tl_sp.set_attribute("tool.retries", 1)
        req_sp.set_attribute("invoiceops.outcome", "approve")

    spans = telemetry.jsonl_exporter.spans
    assert len(spans) == 3
    assert trace_file.exists()

    by_name = {s["name"]: s for s in spans}
    root = by_name["invoiceops.resolve_invoice"]
    agent_child = by_name["agent.extraction_agent"]
    tool_child = by_name["tool.lookup_purchase_order"]

    assert root["parent_span_id"] is None
    assert agent_child["parent_span_id"] == root["span_id"]
    assert tool_child["parent_span_id"] == root["span_id"]
    assert agent_child["attributes"]["gen_ai.usage.input_tokens"] == 1200
    assert tool_child["attributes"]["tool.retries"] == 1
    assert root["attributes"]["invoiceops.outcome"] == "approve"
    _ = SimpleNamespace()
