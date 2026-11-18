"""v1 Multi-Agent System: Extraction + Deterministic Match + Policy RAG + Orchestrator."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.genai import types

from invoiceops.agents.v0_baseline import _parse_decision_response, _sum_event_tokens
from invoiceops.config import AppConfig
from invoiceops.mcp_erp.client import ErpUnavailableError, ResilientErpMcpClient
from invoiceops.schemas import (
    AgentTraceSummary,
    DecisionOutput,
    EvidenceItem,
    InvoiceDocument,
    InvoiceLineItem,
)
from invoiceops.storage.local_store import LocalSqliteStore
from invoiceops.telemetry.tracing import InvoiceOpsTelemetry
from invoiceops.tools.hitl import request_human_approval
from invoiceops.tools.matching import ThreeWayMatchReport, compute_three_way_match

JSON_OBJ_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

EXTRACTION_INSTRUCTION = """You are the Multimodal Invoice Extraction Agent (`extraction_agent`).
Extract the attached supplier invoice PDF into a single JSON object matching this exact schema:
```json
{
  "invoice_id": "INV-XXXX",
  "invoice_number": "BILL-...",
  "vendor_id": "VEN-XXX",
  "vendor_name": "...",
  "po_id": "PO-XXXX or null if none",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "currency": "USD | EUR | GBP",
  "lines": [
    {"sku": "SKU-...", "description": "...", "quantity": 1, "unit_price": 0.0, "line_total": 0.0}
  ],
  "subtotal": 0.0,
  "tax_rate": 0.08,
  "tax_amount": 0.0,
  "total_amount": 0.0,
  "notes": "verbatim supplier notes if any"
}
```
Note: Convert percentage tax like `8.00%` to decimal `0.08`. Use `null` when no PO is referenced.
Output ONLY the JSON object.
"""

POLICY_RAG_INSTRUCTION = """You are the AP Policy Retrieval Specialist (`policy_rag_agent`).
Given the structured invoice summary and deterministic ERP comparison measurements:
1. Formulate 1 or 2 targeted `search_policy` queries covering every anomaly or threshold relevant
   to this case (e.g., unit price variance tolerance, quantity mismatch, duplicate invoice,
   missing PO, currency mismatch, tax rate variance, submission days after GRN, damaged goods,
   or autonomous approval amount thresholds).
2. Return a concise JSON summary listing the exact active policy clause IDs (`POL-...`) and rules
   that govern this invoice. Prefer `ACTIVE` policy versions over `SUPERSEDED` legacy versions.
"""

ORCHESTRATOR_INSTRUCTION = """You are the AP Exception Orchestrator (`orchestrator_agent`).
You receive:
1. Extracted structured `InvoiceDocument`
2. Exact deterministic arithmetic & ERP three-way match measurements (computed by verified Python
   code — trust these numbers, do NOT recompute math yourself)
3. Verbatim policy chunks retrieved by `policy_rag_agent` plus its clause selection

Rules:
- Read the VERBATIM retrieved policy text carefully and apply ONLY `ACTIVE` policy clauses (ignore
  any clause marked `[SUPERSEDED ...]`).
- Match the exact action specified by the active clause (`approve`, `hold`, `reject`, or
  `escalate_to_human`):
  * Clean match or positive price variance <= active tolerance (and total <= active approval limit)
    -> `approve` (do NOT escalate within-tolerance variances!).
  * Moderate over-tolerance variance, quantity > GRN accepted qty, tax mismatch, or late invoice ->
    call `place_payment_hold` first, then output `hold`.
  * Duplicate invoice, missing/invalid PO above spot-buy limit, currency mismatch, or severe price
    overcharge above active reject threshold -> output `reject` directly (do NOT call hold or HITL).
  * ONLY when the active policy specifies `escalate_to_human` (e.g., `DAMAGED_ON_ARRIVAL` GRN,
    clean invoice exceeding active dollar limit, or spot-buy <= $500) OR confidence < 0.80
    -> call `trigger_hitl_approval` first, then output `escalate_to_human`.
- Cite every supporting active `policy_clause` ID and inspected `erp_record` ID.

Respond with a brief audit rationale followed by a single JSON block:
```json
{
  "decision": "approve | hold | reject | escalate_to_human",
  "reason_codes": ["REASON_CODE"],
  "evidence": [
    {"type": "policy_clause", "id": "POL-XX#N"},
    {"type": "erp_record", "id": "PO-XXXX"}
  ],
  "confidence": 0.92,
  "explanation": "Concise 1-2 sentence audit rationale citing exact clause and ERP IDs."
}
```
"""


def format_objective_measurements(report: ThreeWayMatchReport) -> dict[str, Any]:
    """Serialize objective deterministic math & ERP facts without pre-deciding policy thresholds."""
    return {
        "internal_math_valid": report.internal_math_valid,
        "math_discrepancy_usd": report.math_discrepancy_usd,
        "vendor_found": report.vendor_found,
        "vendor_active": report.vendor_active,
        "is_duplicate_invoice": report.is_duplicate_invoice,
        "duplicate_history_ids": report.duplicate_history_ids,
        "po_referenced": report.po_referenced,
        "po_found_and_open": report.po_found_and_open,
        "currency_matches_po": report.currency_matches_po,
        "tax_rate_diff_abs": report.tax_rate_diff_abs,
        "grn_found": report.grn_found,
        "grn_damaged_or_qa_hold": report.grn_damaged_or_qa_hold,
        "days_since_grn_received": report.days_since_grn_received,
        "max_unit_price_variance_pct": round(report.max_unit_price_variance_pct * 100.0, 3),
        "line_comparisons": [
            {
                "sku": lf.sku,
                "sku_found_on_po": lf.sku_found_on_po,
                "invoiced_qty": lf.invoiced_qty,
                "po_qty": lf.po_qty,
                "grn_accepted_qty": lf.grn_accepted_qty,
                "quantity_exceeds_grn": lf.quantity_exceeds_grn,
                "invoiced_unit_price": lf.invoiced_unit_price,
                "po_unit_price": lf.po_unit_price,
                "unit_price_variance_percent": (
                    round(lf.unit_price_variance_pct * 100.0, 3)
                    if lf.unit_price_variance_pct is not None
                    else None
                ),
            }
            for lf in report.line_findings
        ],
        "inspected_erp_ids": report.inspected_erp_ids,
    }


class V1MultiAgentEvaluator:
    """Three-agent ADK pipeline (`Extraction` -> `Match` -> `Policy RAG` -> `Orchestrator`)."""

    def __init__(
        self,
        config: AppConfig,
        store: LocalSqliteStore,
        embed_query_fn: Callable[[str], list[float]],
        erp_client: ResilientErpMcpClient,
        telemetry: InvoiceOpsTelemetry | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.embed_query_fn = embed_query_fn
        self.erp_client = erp_client
        self.telemetry = telemetry
        self._last_llm_ts: float = 0.0

    async def evaluate_invoice(
        self,
        case_id: str,
        vendor_id: str,
        pdf_path: Path,
    ) -> AgentTraceSummary:
        """Execute `v1` multi-agent pipeline with full OpenTelemetry span instrumentation."""
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
        last_resolved_model = self.config.models.orchestrator

        with self._maybe_request_span(case_id=case_id, vendor_id=vendor_id) as req_span:
            # Step 1: Specialist Extraction Agent (Multimodal PDF -> Validated InvoiceDocument)
            ext_prompt = (
                f"Extract invoice `{case_id}` submitted by `vendor_id={vendor_id}` from the PDF."
            )
            ext_parts = [
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                types.Part.from_text(text=ext_prompt),
            ]
            ext_res = await self._run_specialist_agent(
                agent_name="extraction_agent",
                preferred_model=self.config.models.extraction,
                instruction=EXTRACTION_INSTRUCTION,
                parts=ext_parts,
                tools=[],
                json_mode=True,
            )
            if ext_res is None:
                return self._degraded_fallback(
                    t0=t0,
                    reason="LLM_EXTRACTION_UNAVAILABLE",
                    model_id=self.config.models.extraction,
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

            # Step 2: Deterministic ERP Lookups + Pure-Python Three-Way Match Arithmetic
            try:
                vendor_resp = await self._traced_erp_call(
                    "lookup_vendor",
                    tools_called,
                    lambda: self.erp_client.lookup_vendor(invoice_doc.vendor_id),
                )
                dup_resp = await self._traced_erp_call(
                    "check_invoice_history",
                    tools_called,
                    lambda: self.erp_client.check_invoice_history(
                        invoice_doc.vendor_id, invoice_doc.invoice_number
                    ),
                )
                po_resp: dict[str, Any] | None = None
                grn_resp: dict[str, Any] | None = None
                if invoice_doc.po_id:
                    po_id_val = invoice_doc.po_id
                    po_resp = await self._traced_erp_call(
                        "lookup_purchase_order",
                        tools_called,
                        lambda: self.erp_client.lookup_purchase_order(po_id_val),
                    )
                    if po_resp.get("found"):
                        grn_resp = await self._traced_erp_call(
                            "lookup_goods_receipts",
                            tools_called,
                            lambda: self.erp_client.lookup_goods_receipts(po_id_val),
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
                objective_facts = format_objective_measurements(match_report)

            # Step 3: Specialist Policy RAG Agent (still using naive v0 index to isolate v1!)
            verbatim_retrieved_chunks: dict[str, str] = {}

            def search_policy(query: str) -> dict[str, Any]:
                """Search AP policy chunks using vector similarity."""
                nonlocal erp_failed
                with self._maybe_tool_span("search_policy"):
                    tools_called.append("search_policy")
                    try:
                        q_vec = self.embed_query_fn(query)
                        hits = self.store.search_naive(
                            query_vector=q_vec,
                            top_k=self.config.storage.top_k_clauses,
                        )
                        call_cids: list[str] = []
                        for h in hits:
                            verbatim_retrieved_chunks[h.chunk_id] = h.text
                            for cid in h.clause_ids:
                                if cid not in call_cids:
                                    call_cids.append(cid)
                                if cid not in retrieved_clauses:
                                    retrieved_clauses.append(cid)
                        retrieved_ranked_lists.append(call_cids)
                        return {
                            "chunks": [
                                {"chunk_id": h.chunk_id, "score": h.score, "text": h.text}
                                for h in hits
                            ]
                        }
                    except Exception as exc:
                        erp_failed = True
                        return {"unavailable": True, "error": str(exc)}

            rag_input = json.dumps(
                {
                    "invoice_summary": {
                        "invoice_id": invoice_doc.invoice_id,
                        "vendor_id": invoice_doc.vendor_id,
                        "po_id": invoice_doc.po_id,
                        "currency": invoice_doc.currency,
                        "total_amount": invoice_doc.total_amount,
                    },
                    "deterministic_measurements": objective_facts,
                },
                indent=2,
            )
            rag_res = await self._run_specialist_agent(
                agent_name="policy_rag_agent",
                preferred_model=self.config.models.policy_rag,
                instruction=POLICY_RAG_INSTRUCTION,
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

            rag_text, m_rag, in_r, out_r, calls_r, cost_r = rag_res
            last_resolved_model = m_rag
            total_in_tok += in_r
            total_out_tok += out_r
            total_llm_calls += calls_r
            total_cost_usd += cost_r

            # Step 4: Orchestrator Agent (synthesizes decision, places hold or triggers ADK HITL)
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
                    "extracted_invoice": invoice_doc.model_dump(),
                    "deterministic_three_way_match": objective_facts,
                    "verbatim_retrieved_policy_chunks": verbatim_retrieved_chunks,
                    "retrieved_policy_analysis": rag_text,
                },
                indent=2,
            )
            orch_res = await self._run_specialist_agent(
                agent_name="orchestrator_agent",
                preferred_model=self.config.models.orchestrator,
                instruction=ORCHESTRATOR_INSTRUCTION,
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

            # Enforce hard confidence floor from config
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

    async def _run_specialist_agent(
        self,
        agent_name: str,
        preferred_model: str,
        instruction: str,
        parts: list[types.Part],
        tools: list[Callable[..., Any]],
        json_mode: bool = False,
    ) -> tuple[str, str, int, int, int, float] | None:
        """Run one ADK specialist agent with RPM pacing, fallback, and OTel child span."""
        candidates = [preferred_model, *self.config.models.fallback_chain]
        dedup_models: list[str] = []
        for m in candidates:
            if m not in dedup_models:
                dedup_models.append(m)

        for model_id in dedup_models:
            for attempt in range(2):
                try:
                    # Pace Free-Tier 15 RPM quota smoothly (~4.1s min spacing per runner start)
                    now = time.monotonic()
                    wait_s = 4.1 - (now - self._last_llm_ts)
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    self._last_llm_ts = time.monotonic()

                    with self._maybe_agent_span(agent_name=agent_name, model_id=model_id) as span:
                        cfg_kwargs: dict[str, Any] = {
                            "temperature": self.config.models.generation_defaults.temperature,
                            "max_output_tokens": (
                                self.config.models.generation_defaults.max_output_tokens
                            ),
                        }
                        if json_mode and not tools:
                            cfg_kwargs["response_mime_type"] = "application/json"
                        gen_cfg = types.GenerateContentConfig(**cfg_kwargs)
                        agent = Agent(
                            name=agent_name,
                            model=model_id,
                            instruction=instruction,
                            generate_content_config=gen_cfg,
                            tools=tools,
                        )
                        runner = InMemoryRunner(agent=agent, app_name=f"invoiceops_{agent_name}")
                        session = await runner.session_service.create_session(
                            app_name=f"invoiceops_{agent_name}",
                            user_id="eval_harness",
                        )
                        msg = types.Content(role="user", parts=parts)
                        events = [
                            ev
                            async for ev in runner.run_async(
                                user_id="eval_harness",
                                session_id=session.id,
                                new_message=msg,
                            )
                        ]
                        final_text = ""
                        for ev in reversed(events):
                            content = getattr(ev, "content", None)
                            ev_parts = getattr(content, "parts", None) or []
                            texts = [p.text for p in ev_parts if getattr(p, "text", None)]
                            if texts:
                                final_text = "\n".join(texts)
                                break

                        in_tok, out_tok, calls = _sum_event_tokens(events)
                        cost = self.config.estimate_call_cost_usd(
                            model_id=model_id,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                        )
                        if self.telemetry is not None and span is not None:
                            self.telemetry.record_llm_metrics_on_span(
                                span=span,
                                resolved_model=model_id,
                                input_tokens=in_tok,
                                output_tokens=out_tok,
                                estimated_cost_usd=cost,
                                llm_calls=calls,
                            )
                        return final_text, model_id, in_tok, out_tok, calls, cost
                except Exception as exc:
                    err = str(exc)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err:
                        await asyncio.sleep(2.5 * (attempt + 1))
                        continue
                    break
        return None

    async def _traced_erp_call(
        self,
        tool_name: str,
        tools_called: list[str],
        coro_factory: Callable[[], Any],
    ) -> dict[str, Any]:
        retries_before = self.erp_client.case_retries
        with self._maybe_tool_span(tool_name) as span:
            tools_called.append(tool_name)
            try:
                res: dict[str, Any] = await coro_factory()
                if span is not None:
                    span.set_attribute(
                        "tool.retries", self.erp_client.case_retries - retries_before
                    )
                return res
            except Exception:
                if span is not None:
                    span.set_attribute(
                        "tool.retries", self.erp_client.case_retries - retries_before
                    )
                raise

    def _degraded_fallback(
        self,
        t0: float,
        reason: str,
        model_id: str,
        retrieved_clauses: list[str],
        retrieved_ranked_lists: list[list[str]],
        tools_called: list[str],
        in_tok: int,
        out_tok: int,
        llm_calls: int,
        cost_usd: float,
        req_span: Any = None,
    ) -> AgentTraceSummary:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        out = DecisionOutput(
            decision="escalate_to_human",
            reason_codes=[reason],
            evidence=[EvidenceItem(type="policy_clause", id="POL-12#2")],
            confidence=0.0,
        )
        self._finalize_req_span(
            req_span, out.decision, in_tok, out_tok, cost_usd, self.erp_client.case_retries
        )
        return AgentTraceSummary(
            output=out,
            explanation=f"Graceful degradation to human review ({reason}).",
            resolved_model=model_id,
            retrieved_clause_ids=list(retrieved_clauses),
            retrieved_ranked_lists=list(retrieved_ranked_lists),
            tools_called=list(tools_called),
            llm_calls=llm_calls,
            tool_calls=len(tools_called),
            erp_retries=self.erp_client.case_retries,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost_usd=round(cost_usd, 8),
            degraded_due_to_erp_error=True,
        )

    def _maybe_request_span(self, case_id: str, vendor_id: str) -> Any:
        if self.telemetry is not None:
            return self.telemetry.request_span(case_id=case_id, vendor_id=vendor_id, version="v1")
        return _null_span_ctx()

    def _maybe_agent_span(self, agent_name: str, model_id: str) -> Any:
        if self.telemetry is not None:
            return self.telemetry.agent_span(agent_name=agent_name, model_id=model_id)
        return _null_span_ctx()

    def _maybe_tool_span(self, tool_name: str) -> Any:
        if self.telemetry is not None:
            return self.telemetry.tool_span(tool_name=tool_name)
        return _null_span_ctx()

    @staticmethod
    def _finalize_req_span(
        span: Any,
        outcome: str,
        in_tok: int,
        out_tok: int,
        cost_usd: float,
        retries: int,
    ) -> None:
        if span is not None:
            span.set_attribute("invoiceops.outcome", outcome)
            span.set_attribute("gen_ai.usage.input_tokens", in_tok)
            span.set_attribute("gen_ai.usage.output_tokens", out_tok)
            span.set_attribute("invoiceops.estimated_cost_usd", round(cost_usd, 8))
            span.set_attribute("invoiceops.erp_retries", retries)


class _NullSpanContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _null_span_ctx() -> _NullSpanContext:
    return _NullSpanContext()


def _parse_extracted_invoice(
    raw_text: str,
    fallback_id: str,
    vendor_id: str,
) -> InvoiceDocument | None:
    """Parse `ExtractionAgent` JSON output into a validated `InvoiceDocument`."""
    candidate = ""
    fenced = JSON_OBJ_RE.search(raw_text)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw_text[start : end + 1]

    if not candidate:
        return None

    try:
        data = json.loads(candidate)
        data["invoice_id"] = str(data.get("invoice_id") or fallback_id)
        data["vendor_id"] = str(data.get("vendor_id") or vendor_id)
        po_raw = data.get("po_id")
        if po_raw in {"", "NONE", "None", "null", "NONE (NO PO REFERENCED)"}:
            data["po_id"] = None
        # Normalize percentage tax if LLM emitted 8.0 instead of 0.08
        tax_r = float(data.get("tax_rate", 0.0))
        if tax_r > 1.0:
            data["tax_rate"] = round(tax_r / 100.0, 4)
        raw_lines = data.get("lines") or []
        parsed_lines = [
            InvoiceLineItem(
                sku=str(item.get("sku", "UNKNOWN")),
                description=str(item.get("description", "")),
                quantity=max(1, round(float(item.get("quantity", 1)))),
                unit_price=float(item.get("unit_price", 0.0)),
                line_total=float(item.get("line_total", 0.0)),
            )
            for item in raw_lines
            if isinstance(item, dict)
        ]
        if not parsed_lines:
            return None
        data["lines"] = [ln.model_dump() for ln in parsed_lines]
        data.setdefault("notes", "")
        return InvoiceDocument.model_validate(data)
    except Exception:
        return None
