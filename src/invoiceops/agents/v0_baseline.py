"""v0 Baseline Agent: single ADK agent, naive chunking, plain vector search, no math tools."""

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
from google.genai import types

from invoiceops.config import AppConfig
from invoiceops.mcp_erp.client import ErpUnavailableError, ResilientErpMcpClient
from invoiceops.schemas import AgentTraceSummary, DecisionOutput, EvidenceItem
from invoiceops.storage.local_store import LocalSqliteStore

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

V0_SYSTEM_INSTRUCTION = """You are an AP Exception Resolution Agent (`v0` single-agent baseline).
Given an attached supplier invoice PDF and `vendor_id`:
1. Read invoice details from the PDF (invoice #, PO, currency, lines, qty, unit price, tax, dates).
2. Call ERP tools to check vendor status, purchase order, goods receipts, and duplicate history.
3. Call `search_policy` to retrieve AP policy clauses (tolerances, limits, PO/tax/late rules).
4. If the decision is `hold`, call `place_payment_hold` first.
5. If an ERP or policy tool returns `"unavailable": true`, return `"decision": "escalate_to_human"`
   with reason code `"ERP_OR_RETRIEVAL_UNAVAILABLE"`. Never guess an approval.

Respond with a brief rationale followed by one JSON object matching this exact schema:
```json
{
  "decision": "approve | hold | reject | escalate_to_human",
  "reason_codes": ["REASON_CODE"],
  "evidence": [
    {"type": "policy_clause", "id": "POL-XX#N"},
    {"type": "erp_record", "id": "PO-XXXX"}
  ],
  "confidence": 0.90,
  "explanation": "Concise 1-2 sentence audit rationale citing specific clause and ERP IDs."
}
```
"""


class V0BaselineEvaluatorAgent:
    """Executes the single-agent `v0` baseline using Google ADK `Agent` + `InMemoryRunner`."""

    def __init__(
        self,
        config: AppConfig,
        store: LocalSqliteStore,
        embed_query_fn: Callable[[str], list[float]],
        erp_client: ResilientErpMcpClient,
    ) -> None:
        self.config = config
        self.store = store
        self.embed_query_fn = embed_query_fn
        self.erp_client = erp_client

    async def evaluate_invoice(
        self,
        case_id: str,
        vendor_id: str,
        pdf_path: Path,
    ) -> AgentTraceSummary:
        """Run single-agent `v0` workflow on one PDF invoice and return full telemetry."""
        pdf_bytes = pdf_path.read_bytes()
        self.erp_client.reset_case_telemetry()
        retrieved_clauses: list[str] = []
        retrieved_ranked_lists: list[list[str]] = []
        all_tools_called: list[str] = []
        erp_failed = False

        def search_policy(query: str) -> dict[str, Any]:
            """Search Accounts Payable policy chunks using plain vector similarity."""
            nonlocal erp_failed
            all_tools_called.append("search_policy")
            try:
                q_vec = self.embed_query_fn(query)
                hits = self.store.search_naive(
                    query_vector=q_vec,
                    top_k=self.config.storage.top_k_clauses,
                )
                call_clauses: list[str] = []
                for h in hits:
                    for cid in h.clause_ids:
                        if cid not in call_clauses:
                            call_clauses.append(cid)
                        if cid not in retrieved_clauses:
                            retrieved_clauses.append(cid)
                retrieved_ranked_lists.append(call_clauses)
                return {
                    "chunks": [
                        {"chunk_id": h.chunk_id, "score": h.score, "text": h.text} for h in hits
                    ]
                }
            except Exception as exc:
                erp_failed = True
                return {"unavailable": True, "error": str(exc)}

        async def lookup_vendor(vendor_id: str) -> dict[str, Any]:
            """Look up vendor master details in the ERP."""
            nonlocal erp_failed
            all_tools_called.append("lookup_vendor")
            try:
                return await self.erp_client.lookup_vendor(vendor_id)
            except ErpUnavailableError as exc:
                erp_failed = True
                return {"unavailable": True, "error": str(exc)}

        async def lookup_purchase_order(po_id: str) -> dict[str, Any]:
            """Look up a Purchase Order and its line items by `po_id`."""
            nonlocal erp_failed
            all_tools_called.append("lookup_purchase_order")
            try:
                return await self.erp_client.lookup_purchase_order(po_id)
            except ErpUnavailableError as exc:
                erp_failed = True
                return {"unavailable": True, "error": str(exc)}

        async def list_vendor_purchase_orders(vendor_id: str, page: int = 1) -> dict[str, Any]:
            """List paginated Purchase Orders for a supplier `vendor_id`."""
            nonlocal erp_failed
            all_tools_called.append("list_vendor_purchase_orders")
            try:
                return await self.erp_client.list_vendor_purchase_orders(
                    vendor_id=vendor_id, page=page, page_size=5
                )
            except ErpUnavailableError as exc:
                erp_failed = True
                return {"unavailable": True, "error": str(exc)}

        async def lookup_goods_receipts(po_id: str) -> dict[str, Any]:
            """Look up Goods Receipt Notes (GRNs) and QA status for `po_id`."""
            nonlocal erp_failed
            all_tools_called.append("lookup_goods_receipts")
            try:
                return await self.erp_client.lookup_goods_receipts(po_id)
            except ErpUnavailableError as exc:
                erp_failed = True
                return {"unavailable": True, "error": str(exc)}

        async def check_invoice_history(vendor_id: str, invoice_number: str) -> dict[str, Any]:
            """Check ERP ledger for duplicate paid or posted invoices."""
            nonlocal erp_failed
            all_tools_called.append("check_invoice_history")
            try:
                return await self.erp_client.check_invoice_history(vendor_id, invoice_number)
            except ErpUnavailableError as exc:
                erp_failed = True
                return {"unavailable": True, "error": str(exc)}

        async def place_payment_hold(
            invoice_id: str,
            vendor_id: str,
            reason_code: str,
        ) -> dict[str, Any]:
            """Place an idempotent payment hold on `invoice_id` in the ERP."""
            nonlocal erp_failed
            all_tools_called.append("place_payment_hold")
            try:
                return await self.erp_client.place_payment_hold(
                    invoice_id=invoice_id,
                    vendor_id=vendor_id,
                    reason_code=reason_code,
                )
            except ErpUnavailableError as exc:
                erp_failed = True
                return {"unavailable": True, "error": str(exc)}

        tools = [
            search_policy,
            lookup_vendor,
            lookup_purchase_order,
            list_vendor_purchase_orders,
            lookup_goods_receipts,
            check_invoice_history,
            place_payment_hold,
        ]

        candidate_models = [self.config.models.v0_baseline, *self.config.models.fallback_chain]
        dedup_models: list[str] = []
        for m in candidate_models:
            if m not in dedup_models:
                dedup_models.append(m)

        t0 = time.perf_counter()
        last_err: Exception | None = None

        for model_idx, model_id in enumerate(dedup_models):
            # Reset tool telemetry if retrying on fallback model
            if model_idx > 0:
                self.erp_client.reset_case_telemetry()
                retrieved_clauses.clear()
                retrieved_ranked_lists.clear()
                all_tools_called.clear()
                erp_failed = False

            for attempt in range(2):
                try:
                    events, final_text = await self._run_adk_once(
                        model_id=model_id,
                        tools=tools,
                        case_id=case_id,
                        vendor_id=vendor_id,
                        pdf_bytes=pdf_bytes,
                    )
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    in_tok, out_tok, llm_calls = _sum_event_tokens(events)
                    cost_usd = self.config.estimate_call_cost_usd(
                        model_id=model_id,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                    output, explanation = _parse_decision_response(final_text)

                    # Hard safety guardrail: never guess approval when ERP/retrieval failed
                    if erp_failed and output.decision != "escalate_to_human":
                        output = DecisionOutput(
                            decision="escalate_to_human",
                            reason_codes=["ERP_OR_RETRIEVAL_UNAVAILABLE"],
                            evidence=[EvidenceItem(type="policy_clause", id="POL-12#2")],
                            confidence=0.0,
                        )
                        explanation = (
                            "Forced human escalation due to downstream ERP/retrieval outage."
                        )

                    return AgentTraceSummary(
                        output=output,
                        explanation=explanation,
                        resolved_model=model_id,
                        retrieved_clause_ids=list(retrieved_clauses),
                        retrieved_ranked_lists=list(retrieved_ranked_lists),
                        tools_called=list(all_tools_called),
                        llm_calls=llm_calls,
                        tool_calls=len(all_tools_called),
                        erp_retries=self.erp_client.case_retries,
                        latency_ms=latency_ms,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        estimated_cost_usd=cost_usd,
                        degraded_due_to_erp_error=erp_failed,
                    )
                except Exception as exc:
                    last_err = exc
                    err_str = str(exc)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                        await asyncio.sleep(2.5 * (attempt + 1))
                        continue
                    break

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        fallback_output = DecisionOutput(
            decision="escalate_to_human",
            reason_codes=["LLM_OR_SERVICE_UNAVAILABLE"],
            evidence=[EvidenceItem(type="policy_clause", id="POL-12#2")],
            confidence=0.0,
        )
        return AgentTraceSummary(
            output=fallback_output,
            explanation=f"Graceful escalation after model failure: {last_err}",
            resolved_model=dedup_models[0],
            retrieved_clause_ids=list(retrieved_clauses),
            tools_called=list(all_tools_called),
            llm_calls=1,
            tool_calls=len(all_tools_called),
            erp_retries=self.erp_client.case_retries,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            degraded_due_to_erp_error=True,
        )

    async def _run_adk_once(
        self,
        model_id: str,
        tools: list[Callable[..., Any]],
        case_id: str,
        vendor_id: str,
        pdf_bytes: bytes,
    ) -> tuple[list[Any], str]:
        gen_cfg = types.GenerateContentConfig(
            temperature=self.config.models.generation_defaults.temperature,
            max_output_tokens=self.config.models.generation_defaults.max_output_tokens,
        )
        agent = Agent(
            name="invoiceops_v0_baseline",
            model=model_id,
            instruction=V0_SYSTEM_INSTRUCTION,
            generate_content_config=gen_cfg,
            tools=tools,
        )
        runner = InMemoryRunner(agent=agent, app_name="invoiceops_v0")
        session = await runner.session_service.create_session(
            app_name="invoiceops_v0",
            user_id="eval_harness",
        )
        prompt_text = (
            f"Resolve invoice case `{case_id}` submitted by supplier `vendor_id={vendor_id}`. "
            "Inspect the attached invoice PDF, query the ERP and policy knowledge base, "
            "and output your final JSON decision."
        )
        user_msg = types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                types.Part.from_text(text=prompt_text),
            ],
        )
        events = [
            ev
            async for ev in runner.run_async(
                user_id="eval_harness",
                session_id=session.id,
                new_message=user_msg,
            )
        ]
        final_text = ""
        for ev in reversed(events):
            content = getattr(ev, "content", None)
            parts = getattr(content, "parts", None) or []
            texts = [p.text for p in parts if getattr(p, "text", None)]
            if texts:
                final_text = "\n".join(texts)
                break
        return events, final_text


def _sum_event_tokens(events: list[Any]) -> tuple[int, int, int]:
    """Sum input, output+thinking tokens, and LLM call count across ADK turn events."""
    in_tok = 0
    out_tok = 0
    calls = 0
    for ev in events:
        usage = getattr(ev, "usage_metadata", None)
        if usage is not None and getattr(usage, "prompt_token_count", None):
            calls += 1
            in_tok += int(usage.prompt_token_count or 0)
            cand = int(usage.candidates_token_count or 0)
            think = int(getattr(usage, "thoughts_token_count", 0) or 0)
            out_tok += cand + think
    return in_tok, out_tok, max(calls, 1)


def _parse_decision_response(raw_text: str) -> tuple[DecisionOutput, str]:
    """Extract structured `DecisionOutput` and explanation from model text."""
    candidate_json = ""
    fenced = JSON_BLOCK_RE.search(raw_text)
    if fenced:
        candidate_json = fenced.group(1)
    else:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate_json = raw_text[start : end + 1]

    if not candidate_json:
        return (
            DecisionOutput(
                decision="escalate_to_human",
                reason_codes=["UNPARSEABLE_AGENT_OUTPUT"],
                evidence=[],
                confidence=0.0,
            ),
            raw_text.strip()[:400],
        )

    try:
        data = json.loads(candidate_json)
        explanation = str(data.pop("explanation", "")).strip()
        if not explanation:
            prefix = raw_text[: raw_text.find("{")].strip()
            explanation = prefix[:400] if prefix else "Decision emitted without prose summary."
        raw_ev = data.get("evidence") or []
        valid_ev: list[dict[str, str]] = []
        for item in raw_ev:
            if (
                isinstance(item, dict)
                and item.get("type") in {"policy_clause", "erp_record"}
                and item.get("id")
            ):
                valid_ev.append({"type": str(item["type"]), "id": str(item["id"])})
        data["evidence"] = valid_ev
        decision_val = str(data.get("decision", "")).strip().lower()
        if decision_val not in {"approve", "hold", "reject", "escalate_to_human"}:
            decision_val = "escalate_to_human"
        data["decision"] = decision_val
        conf = float(data.get("confidence", 0.5))
        data["confidence"] = min(1.0, max(0.0, conf))
        return DecisionOutput.model_validate(data), explanation
    except Exception:
        return (
            DecisionOutput(
                decision="escalate_to_human",
                reason_codes=["SCHEMA_VALIDATION_FAILED"],
                evidence=[],
                confidence=0.0,
            ),
            raw_text.strip()[:400],
        )
