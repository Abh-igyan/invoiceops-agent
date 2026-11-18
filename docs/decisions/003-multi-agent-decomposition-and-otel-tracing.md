# ADR-003: Multi-Agent Decomposition, Deterministic Arithmetic Boundary, and OpenTelemetry Tracing

- **Status:** Accepted
- **Date:** 2026-11-17
- **Phase:** P2 (`v1` Multi-Agent + OpenTelemetry Tracing)

---

## 1. Context

In Phase P1, the single-agent `v0` baseline (`gemini-3.5-flash-lite` reading the raw PDF, calling ERP/policy tools, and performing tolerance/date arithmetic in prose) achieved `61.1% ± 23.9%` accuracy and `2.694 ± 0.316` cost-weighted error across 3 runs on `golden_12`. Failure taxonomy analysis revealed two structural root causes behind `v0`'s errors:
1. **Conflated Multimodal Extraction, Arithmetic, and Policy Reasoning:** Asking one monolithic prompt to simultaneously parse noisy scanned PDFs, compute percentage price variances, subtract calendar days between `received_date` and `invoice_date`, check QA damage flags, and interpret policy led to missed escalations (`escalation_recall = 0.0%`) and date/currency errors (`late_invoice_vs_terms`, `ambiguous_escalation`, `currency_mismatch`).
2. **Repeated Binary PDF Resending & Opaque Debugging:** Every tool-calling turn in `v0` re-sent the attached binary PDF in conversation history (`~7,043` mean input tokens/case), and without per-agent/per-tool OpenTelemetry spans it was impossible to isolate whether latency or token spend came from extraction, ERP retries, or policy search.

---

## 2. Decisions

### 2.1 Three-Agent `v1` Decomposition (Reserving Slot #4 for `v4` Critic)
To stay strictly within the 4-agent architectural ceiling (Section 3) while keeping each experiment single-variable and reversible (Section 7), `v1` decomposes resolution into three specialized ADK `Agent`s plus a deterministic tool stage:
1. **`extraction_agent` (Multimodal PDF → `InvoiceDocument`):** The *only* agent that ingests the binary PDF (`1` LLM turn), emitting a validated Pydantic `InvoiceDocument` so downstream agents operate purely on compact structured JSON.
2. **Deterministic ERP + Three-Way Match Stage (`compute_three_way_match`):** Queries the MCP ERP (`lookup_vendor`, `check_invoice_history`, `lookup_purchase_order`, `lookup_goods_receipts`) and computes exact objective measurements in Python (`unit_price_variance_percent`, `days_since_grn_received`, `tax_rate_diff_abs`, `quantity_exceeds_grn`, `grn_damaged_or_qa_hold`, `internal_math_valid`). Crucially, `format_objective_measurements` passes objective mathematical facts rather than pre-hardcoding version-sensitive policy thresholds—so policy interpretation still relies on RAG (keeping `v2` retrieval experiments unconfounded).
3. **`policy_rag_agent` (Targeted Policy Search):** Formulates targeted `search_policy` queries from the deterministic anomaly summary (**deliberately retaining `v0`'s naive fixed-character chunking and plain vector index** so `EXP-001` isolates decomposition + deterministic math).
4. **`orchestrator_agent` (Decision Synthesis, ERP Hold & ADK HITL):** Synthesizes the retrieved active clauses with the deterministic measurements, invokes `place_payment_hold` on holds, and invokes `trigger_hitl_approval` (`ToolContext.request_confirmation`) when human sign-off is required.

### 2.2 ADK Tool Confirmation for Human-in-the-Loop (`src/invoiceops/tools/hitl.py`)
High-value (`>$10,000`), damaged-goods (`DAMAGED_ON_ARRIVAL`), spot-buy, or low-confidence (`<0.80`) cases invoke ADK 2.14's native `ToolContext.request_confirmation(hint=..., payload=...)`, persisting `pending_hitl_decision` in `tool_context.state` across the pause and finalizing `approve` or `reject` upon resumption.

### 2.3 Dual-Mode OpenTelemetry Span Hierarchy (`src/invoiceops/telemetry/tracing.py`)
Every request creates one root `invoiceops.resolve_invoice` span with child `agent.<name>` and `tool.<name>` spans carrying `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `invoiceops.estimated_cost_usd`, `tool.retries`, and `invoiceops.outcome`.
- **Phases P2–P3 (`local_ai_studio`, `$0.00` mode):** Exports spans locally to `results/traces/v1_otel_spans.jsonl` (plus committed sample `results/v1_trace_sample.json` and `python scripts/tasks.py traces`), with optional OTLP HTTP export (`INVOICEOPS_ENABLE_OTLP=true`).
- **Phase P4 (`gcp_enterprise`):** Automatically attaches `CloudTraceSpanExporter` (`opentelemetry-exporter-gcp-trace`) so identical span trees appear natively in Google Cloud Trace.

---

## 3. Trade-offs

- **Sequential Pipeline Latency vs. Monolithic Turn Count:** Separating extraction, policy RAG, and orchestration guarantees at least 3 sequential ADK runner invocations (~4–5 LLM calls total vs `v0`'s 2.83 calls), slightly increasing nominal call count on trivial clean matches, but avoids re-sending multimodal PDF bytes on every tool turn and eliminates prose-math hallucinations.
- **Deferring Schema-Retry & Critic Loops to `v4`:** Single-pass extraction in `v1` means severely degraded scans could still fail JSON parsing (`SCHEMA_VALIDATION_FAILED`), cleanly isolating bounded schema retries and Critic self-reflection for `v4`.
