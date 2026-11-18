"""ADK Tool Confirmation / Human-in-the-Loop (HITL) approval gate (Section 3)."""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext


def request_human_approval(
    invoice_id: str,
    vendor_id: str,
    total_amount: float,
    currency: str,
    reason_code: str,
    confidence: float,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Pause high-value, damaged-goods, or low-confidence cases for human sign-off via ADK HITL.

    Persists `pending_hitl_decision` inside ADK `tool_context.state` across the pause and
    invokes `tool_context.request_confirmation(...)` until a human confirms or rejects.
    """
    pending_payload = {
        "invoice_id": invoice_id,
        "vendor_id": vendor_id,
        "total_amount": total_amount,
        "currency": currency,
        "reason_code": reason_code,
        "confidence": confidence,
    }
    tool_context.state["pending_hitl_decision"] = pending_payload

    confirmation = tool_context.tool_confirmation
    if confirmation is None:
        tool_context.request_confirmation(
            hint=(
                f"Human approval required for {invoice_id} ({currency} {total_amount:,.2f}, "
                f"reason={reason_code}, confidence={confidence:.2f})."
            ),
            payload=pending_payload,
        )
        return {
            "status": "PAUSED_AWAITING_HUMAN_CONFIRMATION",
            "decision": "escalate_to_human",
            "preserved_state": pending_payload,
        }

    if confirmation.confirmed:
        tool_context.state["hitl_resolution"] = "APPROVED_BY_HUMAN"
        return {
            "status": "CONFIRMED_BY_HUMAN",
            "decision": "approve",
            "preserved_state": tool_context.state.get("pending_hitl_decision"),
        }

    tool_context.state["hitl_resolution"] = "REJECTED_BY_HUMAN"
    return {
        "status": "REJECTED_BY_HUMAN",
        "decision": "reject",
        "preserved_state": tool_context.state.get("pending_hitl_decision"),
    }
