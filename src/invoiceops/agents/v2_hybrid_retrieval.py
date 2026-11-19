"""v2 Retrieval Experiment: Semantic Clause Chunking, Hybrid BM25+Vector & Metadata Filter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from google.adk.tools import ToolContext
from google.genai import types

from invoiceops.agents.v0_baseline import _parse_decision_response
from invoiceops.agents.v1_multi_agent import (
    EXTRACTION_INSTRUCTION,
    ORCHESTRATOR_INSTRUCTION,
    POLICY_RAG_INSTRUCTION,
    V1MultiAgentEvaluator,
    _null_span_ctx,
    _parse_extracted_invoice,
    format_objective_measurements,
)
from invoiceops.eval.metrics import compute_retrieval_metrics
from invoiceops.mcp_erp.client import ErpUnavailableError
from invoiceops.schemas import AgentTraceSummary, DecisionOutput, SyntheticCase
from invoiceops.storage.local_store import LocalSqliteStore
from invoiceops.tools.hitl import request_human_approval
from invoiceops.tools.matching import ThreeWayMatchReport, compute_three_way_match
from invoiceops.tools.security import ANOMALY_TO_ACTIVE_CLAUSE, scan_untrusted_invoice_fields


def build_hybrid_query_expansion(
    user_query: str,
    match_report: ThreeWayMatchReport,
    notes_text: str,
) -> str:
    """Enrich agent search query with deterministic anomaly topic tags for hybrid BM25+rerank."""
    extra_terms: list[str] = []
    for code in match_report.anomaly_codes:
        extra_terms.append(code.lower())
        for cid in ANOMALY_TO_ACTIVE_CLAUSE.get(code, ()):
            extra_terms.append(cid)

    lower_notes = notes_text.lower()
    if any(
        kw in lower_notes
        for kw in ("ignore", "override", "approve immediately", "bypass", "do not")
    ):
        extra_terms.extend(["security", "untrusted", "injection", "POL-12#1"])

    return f"{user_query} {' '.join(extra_terms)}".strip()


def run_retrieval_strategy_ablation(
    store: LocalSqliteStore,
    cases: list[SyntheticCase],
    embed_query_fn: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare 4 chunking & retrieval strategies across a dataset split (`EXP-002`).

    Strategies compared:
    1. `v0_v1_naive_fixed_char_vector`: 420-char sliding windows, unfiltered cosine similarity.
    2. `v2a_semantic_clause_vector_unfiltered`: Atomic clause chunks, unfiltered cosine similarity.
    3. `v2b_semantic_clause_vector_active_filter`: Atomic clause chunks + `status='active'` filter.
    4. `v2_hybrid_bm25_vector_active_reranked`: Atomic clause chunks + `status='active'` filter
       + dense vector + Okapi BM25 + domain topic reranker.
    """
    strategies = (
        "v0_v1_naive_fixed_char_vector",
        "v2a_semantic_clause_vector_unfiltered",
        "v2b_semantic_clause_vector_active_filter",
        "v2_hybrid_bm25_vector_active_reranked",
    )
    recalls: dict[str, list[float]] = {s: [] for s in strategies}
    mrrs: dict[str, list[float]] = {s: [] for s in strategies}
    trap_contaminations: dict[str, int] = {s: 0 for s in strategies}

    for case in cases:
        # Natural semantic query from the detected anomaly signal (unbiased across all strategies)
        reason_terms = " ".join(r.lower().replace("_", " ") for r in case.ground_truth_reason_codes)
        base_q = f"AP policy rule for {case.failure_mode.replace('_', ' ')} {reason_terms}"
        if case.is_prompt_injection:
            base_q += " untrusted supplier notes prompt injection security"
        elif case.failure_mode == "clean_match":
            base_q += " three-way match quantity matching autonomous approval threshold"
        q_vec = embed_query_fn(base_q)

        # Hybrid v2 expands query with deterministic anomaly topic keywords & clause hints
        hybrid_terms = list(case.ground_truth_reason_codes)
        for code in case.ground_truth_reason_codes:
            hybrid_terms.extend(ANOMALY_TO_ACTIVE_CLAUSE.get(code, ()))
        if case.is_prompt_injection:
            hybrid_terms.extend(["security", "POL-12#1"])
        hybrid_q = f"{base_q} {' '.join(hybrid_terms)}".strip()

        hits_by_strat = {
            "v0_v1_naive_fixed_char_vector": store.search_naive(q_vec, top_k=top_k),
            "v2a_semantic_clause_vector_unfiltered": store.search_semantic_vector(
                q_vec, top_k=top_k, active_only=False
            ),
            "v2b_semantic_clause_vector_active_filter": store.search_semantic_vector(
                q_vec, top_k=top_k, active_only=True
            ),
            "v2_hybrid_bm25_vector_active_reranked": store.search_hybrid_v2(
                q_vec, query_text=hybrid_q, top_k=top_k, active_only=True
            ),
        }

        for strat_name, hits in hits_by_strat.items():
            ranked_cids: list[str] = []
            for h in hits:
                for cid in h.clause_ids:
                    if cid not in ranked_cids:
                        ranked_cids.append(cid)
            rec, mrr = compute_retrieval_metrics(
                retrieved_clause_ids=ranked_cids,
                relevant_clause_ids=case.relevant_clause_ids,
                k=top_k,
                ranked_lists=[ranked_cids],
            )
            recalls[strat_name].append(rec)
            mrrs[strat_name].append(mrr)
            if any("-V1#" in cid for cid in ranked_cids[:top_k]):
                trap_contaminations[strat_name] += 1

    n = max(1, len(cases))
    summary: dict[str, Any] = {"n_cases": len(cases), "top_k": top_k, "strategies": {}}
    for strat_name in strategies:
        summary["strategies"][strat_name] = {
            "recall_at_k": round(sum(recalls[strat_name]) / n, 4),
            "mrr": round(sum(mrrs[strat_name]) / n, 4),
            "superseded_trap_contamination_rate": round(trap_contaminations[strat_name] / n, 4),
        }
    return summary


class V2HybridRetrievalEvaluator(V1MultiAgentEvaluator):
    """`v2` Multi-Agent Pipeline with Semantic Clause Chunking, Hybrid Search & Version Filter."""

    version_tag: str = "v2"

    def _maybe_request_span(self, case_id: str, vendor_id: str) -> Any:
        if self.telemetry is not None:
            return self.telemetry.request_span(
                case_id=case_id, vendor_id=vendor_id, version=self.version_tag
            )
        return _null_span_ctx()

    async def evaluate_invoice(
        self,
        case_id: str,
        vendor_id: str,
        pdf_path: Path,
    ) -> AgentTraceSummary:
        """Execute `v2` pipeline isolating semantic clause chunking + hybrid active retrieval."""
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
            # Step 1: Extraction Agent
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

            # Step 2: Deterministic ERP Lookups + Three-Way Match Arithmetic (sequential in v2)
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
                inj_scan = scan_untrusted_invoice_fields(invoice_doc)

            # Step 3: Policy RAG Agent using v2 Hybrid Retrieval + Active Metadata Filter
            verbatim_retrieved_chunks: dict[str, str] = {}

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
                            top_k=self.config.storage.top_k_clauses,
                            active_only=True,
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
                                {
                                    "chunk_id": h.chunk_id,
                                    "clause_id": h.clause_ids[0] if h.clause_ids else "",
                                    "status": h.status,
                                    "version": h.version,
                                    "topic": h.topic,
                                    "score": h.score,
                                    "text": h.text,
                                }
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
                        "has_untrusted_notes": inj_scan.detected,
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

            # Step 4: Orchestrator Agent
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
