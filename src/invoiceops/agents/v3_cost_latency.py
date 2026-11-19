"""v3 Cost/Latency Experiment: Tiered Model Routing, Prompt Trimming & Parallel Async ERP Calls."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from google.adk.tools import ToolContext
from google.genai import types

from invoiceops.agents.v0_baseline import _parse_decision_response
from invoiceops.agents.v1_multi_agent import (
    EXTRACTION_INSTRUCTION,
    _parse_extracted_invoice,
    format_objective_measurements,
)
from invoiceops.agents.v2_hybrid_retrieval import (
    V2HybridRetrievalEvaluator,
    build_hybrid_query_expansion,
)
from invoiceops.mcp_erp.client import ErpUnavailableError
from invoiceops.schemas import AgentTraceSummary, DecisionOutput, InvoiceDocument
from invoiceops.tools.hitl import request_human_approval
from invoiceops.tools.matching import ThreeWayMatchReport, compute_three_way_match
from invoiceops.tools.security import scan_untrusted_invoice_fields

V3_COMPACT_RAG_INSTRUCTION = """You are the fast AP Policy Router (`policy_rag_agent`).
Call `search_policy` ONCE with a concise keyword query covering all detected anomalies in the input,
then reply with a short comma-separated list of the active clause IDs returned (e.g. `POL-02-V2#3`).
Keep output under 25 tokens."""

V3_COMPACT_ORCHESTRATOR_INSTRUCTION = """You are the AP Orchestrator (`orchestrator_agent`).
Input JSON has `invoice`, `match` (Python math), and `active_clauses` (active AP policy rules).
Decision rules (apply ONLY active clauses):
- Clean match or price variance <= 2.0% (total <= $10,000) -> `approve`.
- Price variance > 2.0% to 10.0%, qty > GRN accepted, tax diff > 0.005, or days_since_grn > 60
  -> call `place_payment_hold` first, then `hold`.
- Duplicate invoice, missing/invalid PO > $500, currency mismatch, or price variance > 10.0%
  -> `reject` directly (do NOT call hold or HITL).
- Clean total > $10,000, GRN damaged/QA hold, spot-buy <= $500, or confidence < 0.80
  -> call `trigger_hitl_approval` first, then `escalate_to_human`.
- Cite every supporting `policy_clause` ID and inspected `erp_record` ID.
Output ONLY one JSON block:
```json
{
  "decision": "approve|hold|reject|escalate_to_human",
  "reason_codes": ["CODE"],
  "evidence": [
    {"type": "policy_clause", "id": "POL-XX#N"},
    {"type": "erp_record", "id": "PO-XXXX"}
  ],
  "confidence": 0.95,
  "explanation": "1-sentence rationale citing clause and ERP IDs."
}
```"""


def format_compact_invoice_header(invoice: InvoiceDocument) -> dict[str, Any]:
    """Trim verbose line descriptions from Orchestrator payload to save ~35% input tokens."""
    return {
        "invoice_id": invoice.invoice_id,
        "invoice_number": invoice.invoice_number,
        "vendor_id": invoice.vendor_id,
        "po_id": invoice.po_id,
        "invoice_date": invoice.invoice_date,
        "currency": invoice.currency,
        "subtotal": invoice.subtotal,
        "tax_rate": invoice.tax_rate,
        "total_amount": invoice.total_amount,
        "has_notes": bool(invoice.notes and invoice.notes.strip()),
    }


def format_compact_match_facts(report: ThreeWayMatchReport) -> dict[str, Any]:
    """Serialize compact deterministic match signals without pretty-print whitespace."""
    base = format_objective_measurements(report)
    base["anomaly_codes"] = report.anomaly_codes
    return base


class V3CostLatencyEvaluator(V2HybridRetrievalEvaluator):
    """`v3` Pipeline: Tiered Model Routing, Prompt Trimming & Parallel Async ERP Calls."""

    version_tag: str = "v3"
    use_lite_orchestrator: bool = False
    aggressive_top1_truncation: bool = False

    async def _parallel_erp_lookups(
        self,
        invoice_doc: InvoiceDocument,
        tools_called: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        """Execute independent read-only ERP MCP lookups concurrently via `asyncio.gather`."""

        async def _call_vendor() -> dict[str, Any]:
            return await self._traced_erp_call(
                "lookup_vendor",
                tools_called,
                lambda: self.erp_client.lookup_vendor(invoice_doc.vendor_id),
            )

        async def _call_dup() -> dict[str, Any]:
            return await self._traced_erp_call(
                "check_invoice_history",
                tools_called,
                lambda: self.erp_client.check_invoice_history(
                    invoice_doc.vendor_id, invoice_doc.invoice_number
                ),
            )

        if invoice_doc.po_id:
            po_id_val = invoice_doc.po_id

            async def _call_po() -> dict[str, Any]:
                return await self._traced_erp_call(
                    "lookup_purchase_order",
                    tools_called,
                    lambda: self.erp_client.lookup_purchase_order(po_id_val),
                )

            vendor_resp, dup_resp, po_resp = await asyncio.gather(
                _call_vendor(), _call_dup(), _call_po()
            )
            grn_resp: dict[str, Any] | None = None
            if po_resp and po_resp.get("found"):
                grn_resp = await self._traced_erp_call(
                    "lookup_goods_receipts",
                    tools_called,
                    lambda: self.erp_client.lookup_goods_receipts(po_id_val),
                )
            return vendor_resp, dup_resp, po_resp, grn_resp

        vendor_resp, dup_resp = await asyncio.gather(_call_vendor(), _call_dup())
        return vendor_resp, dup_resp, None, None

    async def evaluate_invoice(
        self,
        case_id: str,
        vendor_id: str,
        pdf_path: Path,
    ) -> AgentTraceSummary:
        """Execute `v3` pipeline with model routing, parallel ERP lookups, and prompt trimming."""
        t0 = time.perf_counter()
        pdf_bytes = pdf_path.read_bytes()
        self.erp_client.reset_case_telemetry()

        retrieved_clauses: list[str] = []
        retrieved_ranked_lists: list[list[str]] = []
        tools_called: list[str] = []
        total_in_tok = 0
        total_out_tok = 0
        total_llm_calls = 0
        total_cost_usd = 0.0
        erp_failed = False

        fast_model = self.config.models.fast_router
        orch_model = (
            self.config.models.fast_router
            if self.use_lite_orchestrator
            else self.config.models.orchestrator
        )
        last_resolved_model = orch_model

        with self._maybe_request_span(case_id=case_id, vendor_id=vendor_id) as req_span:
            # Step 1: Extraction routed to fast_router (`gemini-3.1-flash-lite`)
            ext_prompt = (
                f"Extract invoice `{case_id}` submitted by `vendor_id={vendor_id}` from the PDF."
            )
            ext_parts = [
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                types.Part.from_text(text=ext_prompt),
            ]
            ext_res = await self._run_specialist_agent(
                agent_name="extraction_agent",
                preferred_model=fast_model,
                instruction=EXTRACTION_INSTRUCTION,
                parts=ext_parts,
                tools=[],
                json_mode=True,
            )
            if ext_res is None:
                return self._degraded_fallback(
                    t0=t0,
                    reason="LLM_EXTRACTION_UNAVAILABLE",
                    model_id=fast_model,
                    retrieved_clauses=retrieved_clauses,
                    retrieved_ranked_lists=retrieved_ranked_lists,
                    tools_called=tools_called,
                    in_tok=total_in_tok,
                    out_tok=total_out_tok,
                    llm_calls=max(1, total_llm_calls),
                    cost_usd=total_cost_usd,
                )

            ext_text, m_used, in_t, out_t, calls_t, cost_t = ext_res
            last_resolved_model = m_used
            total_in_tok += in_t
            total_out_tok += out_t
            total_llm_calls += calls_t
            total_cost_usd += cost_t

            invoice_doc = _parse_extracted_invoice(
                ext_text, fallback_id=case_id, vendor_id=vendor_id
            )
            if invoice_doc is None:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                out = DecisionOutput(
                    decision="escalate_to_human",
                    reason_codes=["SCHEMA_VALIDATION_FAILED"],
                    evidence=[],
                    confidence=0.0,
                )
                self._finalize_req_span(
                    req_span, out.decision, total_in_tok, total_out_tok, total_cost_usd, 0
                )
                return AgentTraceSummary(
                    output=out,
                    explanation="Extraction agent returned invalid InvoiceDocument JSON.",
                    resolved_model=last_resolved_model,
                    retrieved_clause_ids=[],
                    retrieved_ranked_lists=[],
                    tools_called=tools_called,
                    llm_calls=total_llm_calls,
                    tool_calls=len(tools_called),
                    erp_retries=self.erp_client.case_retries,
                    latency_ms=latency_ms,
                    input_tokens=total_in_tok,
                    output_tokens=total_out_tok,
                    estimated_cost_usd=total_cost_usd,
                    degraded_due_to_erp_error=False,
                )

            # Step 2: Parallel ERP Lookups via asyncio.gather + Deterministic Three-Way Match
            try:
                vendor_resp, dup_resp, po_resp, grn_resp = await self._parallel_erp_lookups(
                    invoice_doc=invoice_doc,
                    tools_called=tools_called,
                )
            except ErpUnavailableError:
                erp_failed = True
                return self._degraded_fallback(
                    t0=t0,
                    reason="ERP_OR_RETRIEVAL_UNAVAILABLE",
                    model_id=last_resolved_model,
                    retrieved_clauses=retrieved_clauses,
                    retrieved_ranked_lists=retrieved_ranked_lists,
                    tools_called=tools_called,
                    in_tok=total_in_tok,
                    out_tok=total_out_tok,
                    llm_calls=total_llm_calls,
                    cost_usd=total_cost_usd,
                    req_span=req_span,
                )

            with self._maybe_tool_span("compute_three_way_match"):
                tools_called.append("compute_three_way_match")
                match_report = compute_three_way_match(
                    invoice=invoice_doc,
                    vendor_resp=vendor_resp,
                    po_resp=po_resp,
                    grn_resp=grn_resp,
                    dup_resp=dup_resp,
                    rules=self.config.matching_and_hitl,
                )
                compact_facts = format_compact_match_facts(match_report)
                inj_scan = scan_untrusted_invoice_fields(invoice_doc)

            # Step 3: Policy RAG Agent routed to fast_router (`gemini-3.1-flash-lite`)
            active_clauses_map: dict[str, str] = {}
            effective_top_k = (
                1 if self.aggressive_top1_truncation else self.config.storage.top_k_clauses
            )

            def search_policy(query: str) -> dict[str, Any]:
                """Search active AP policy clauses using v2 hybrid vector + BM25 + reranker."""
                nonlocal erp_failed
                with self._maybe_tool_span("search_policy"):
                    tools_called.append("search_policy")
                    try:
                        expanded_q = build_hybrid_query_expansion(
                            user_query=query,
                            match_report=match_report,
                            notes_text=invoice_doc.notes,
                        )
                        q_vec = self.embed_query_fn(expanded_q)
                        hits = self.store.search_hybrid_v2(
                            query_vector=q_vec,
                            query_text=expanded_q,
                            top_k=effective_top_k,
                            active_only=True,
                        )
                        call_cids: list[str] = []
                        for h in hits:
                            cid_key = h.clause_ids[0] if h.clause_ids else h.chunk_id
                            clause_body = h.text[:55] if self.aggressive_top1_truncation else h.text
                            active_clauses_map[cid_key] = clause_body
                            for cid in h.clause_ids:
                                if cid not in call_cids:
                                    call_cids.append(cid)
                                if cid not in retrieved_clauses:
                                    retrieved_clauses.append(cid)
                        retrieved_ranked_lists.append(call_cids)
                        return {
                            "clauses": [
                                {
                                    "clause_id": h.clause_ids[0] if h.clause_ids else "",
                                    "score": h.score,
                                    "text": (
                                        h.text[:55] if self.aggressive_top1_truncation else h.text
                                    ),
                                }
                                for h in hits
                            ]
                        }
                    except Exception as exc:
                        erp_failed = True
                        return {"unavailable": True, "error": str(exc)}

            rag_input = json.dumps(
                {
                    "invoice_id": invoice_doc.invoice_id,
                    "total_amount": invoice_doc.total_amount,
                    "anomalies": match_report.anomaly_codes,
                    "untrusted_notes": inj_scan.detected,
                },
                separators=(",", ":"),
            )
            rag_res = await self._run_specialist_agent(
                agent_name="policy_rag_agent",
                preferred_model=fast_model,
                instruction=V3_COMPACT_RAG_INSTRUCTION,
                parts=[types.Part.from_text(text=rag_input)],
                tools=[search_policy],
            )
            if rag_res is None or erp_failed:
                return self._degraded_fallback(
                    t0=t0,
                    reason="ERP_OR_RETRIEVAL_UNAVAILABLE",
                    model_id=last_resolved_model,
                    retrieved_clauses=retrieved_clauses,
                    retrieved_ranked_lists=retrieved_ranked_lists,
                    tools_called=tools_called,
                    in_tok=total_in_tok,
                    out_tok=total_out_tok,
                    llm_calls=total_llm_calls,
                    cost_usd=total_cost_usd,
                    req_span=req_span,
                )

            _rag_text, m_rag, in_r, out_r, calls_r, cost_r = rag_res
            last_resolved_model = m_rag
            total_in_tok += in_r
            total_out_tok += out_r
            total_llm_calls += calls_r
            total_cost_usd += cost_r

            # Step 4: Orchestrator Agent (deduplicated compact payload)
            hitl_triggered = False

            async def place_payment_hold(
                invoice_id: str,
                vendor_id: str,
                reason_code: str,
            ) -> dict[str, Any]:
                """Place an idempotent payment hold in the ERP ledger."""
                nonlocal erp_failed
                try:
                    return await self._traced_erp_call(
                        "place_payment_hold",
                        tools_called,
                        lambda: self.erp_client.place_payment_hold(
                            invoice_id=invoice_id,
                            vendor_id=vendor_id,
                            reason_code=reason_code,
                        ),
                    )
                except ErpUnavailableError as exc:
                    erp_failed = True
                    return {"unavailable": True, "error": str(exc)}

            def trigger_hitl_approval(
                reason_code: str,
                confidence: float,
                tool_context: ToolContext,
            ) -> dict[str, Any]:
                """Pause case for human sign-off using ADK Tool Confirmation state preservation."""
                nonlocal hitl_triggered
                hitl_triggered = True
                with self._maybe_tool_span("request_human_approval"):
                    tools_called.append("request_human_approval")
                    return request_human_approval(
                        invoice_id=invoice_doc.invoice_id,
                        vendor_id=invoice_doc.vendor_id,
                        total_amount=invoice_doc.total_amount,
                        currency=invoice_doc.currency,
                        reason_code=reason_code,
                        confidence=confidence,
                        tool_context=tool_context,
                    )

            orch_payload = json.dumps(
                {
                    "invoice": format_compact_invoice_header(invoice_doc),
                    "match": compact_facts,
                    "active_clauses": active_clauses_map,
                },
                separators=(",", ":"),
            )
            orch_res = await self._run_specialist_agent(
                agent_name="orchestrator_agent",
                preferred_model=orch_model,
                instruction=V3_COMPACT_ORCHESTRATOR_INSTRUCTION,
                parts=[types.Part.from_text(text=orch_payload)],
                tools=[place_payment_hold, trigger_hitl_approval],
            )
            if orch_res is None or erp_failed:
                return self._degraded_fallback(
                    t0=t0,
                    reason="ERP_OR_RETRIEVAL_UNAVAILABLE",
                    model_id=last_resolved_model,
                    retrieved_clauses=retrieved_clauses,
                    retrieved_ranked_lists=retrieved_ranked_lists,
                    tools_called=tools_called,
                    in_tok=total_in_tok,
                    out_tok=total_out_tok,
                    llm_calls=total_llm_calls,
                    cost_usd=total_cost_usd,
                    req_span=req_span,
                )

            orch_text, m_orch, in_o, out_o, calls_o, cost_o = orch_res
            last_resolved_model = m_orch
            total_in_tok += in_o
            total_out_tok += out_o
            total_llm_calls += calls_o
            total_cost_usd += cost_o

            output, explanation = _parse_decision_response(orch_text)
            if hitl_triggered and output.decision not in {"reject", "hold"}:
                output = output.model_copy(update={"decision": "escalate_to_human"})

            if (
                output.decision == "approve"
                and output.confidence < self.config.matching_and_hitl.human_approval_min_confidence
            ):
                output = output.model_copy(
                    update={
                        "decision": "escalate_to_human",
                        "reason_codes": [*output.reason_codes, "LOW_CONFIDENCE_ESCALATION"],
                    }
                )

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self._finalize_req_span(
                req_span,
                output.decision,
                total_in_tok,
                total_out_tok,
                total_cost_usd,
                self.erp_client.case_retries,
            )
            return AgentTraceSummary(
                output=output,
                explanation=explanation,
                resolved_model=last_resolved_model,
                retrieved_clause_ids=list(retrieved_clauses),
                retrieved_ranked_lists=list(retrieved_ranked_lists),
                tools_called=list(tools_called),
                llm_calls=total_llm_calls,
                tool_calls=len(tools_called),
                erp_retries=self.erp_client.case_retries,
                latency_ms=latency_ms,
                input_tokens=total_in_tok,
                output_tokens=total_out_tok,
                estimated_cost_usd=round(total_cost_usd, 8),
                degraded_due_to_erp_error=False,
            )


class V3NegativeAllLiteEvaluator(V3CostLatencyEvaluator):
    """`v3_negative` Ablation (`EXP-004`): Over-aggressive top-1 truncation + all-lite routing.

    Logged negative experiment demonstrating that pruning policy retrieval from `top_k=5` full
    clauses to `top_k=1` 55-char snippets while downgrading the Orchestrator to `3.1-flash-lite`
    degrades multi-clause retrieval recall and decision accuracy.
    """

    version_tag: str = "v3_negative"
    use_lite_orchestrator: bool = True
    aggressive_top1_truncation: bool = True
