# ADR-001: Architecture, Dual-Mode Execution, and Verified Google Cloud Stack

- **Status:** Accepted
- **Date:** 2026-11-17
- **Phase:** P0 (Foundation & Guardrails)

---

## 1. Context

`invoiceops-agent` resolves accounts-payable three-way match exceptions (invoice PDF/image vs. purchase order vs. goods receipt) against versioned enterprise policies and a flaky legacy ERP. The primary goal is enforcing production engineering standards: integration resilience, strict boundary between deterministic math and probabilistic reasoning, controlled evaluation before and after every architectural change, OpenTelemetry tracing, and hard cost guardrails.

Key constraints established with the project owner prior to implementation:
1. **Zero-Cost Iteration (Phases P1–P3):** Spend must be strictly **$0.00** during dataset construction, baseline evaluation (`v0`), multi-agent decomposition (`v1`), and iterative experiments (`v2`–`v4`), using a Google AI Studio `GEMINI_API_KEY` stored only in a git-ignored `.env`.
2. **Production GCP Cutover (Phase P4):** Live Google Cloud resources (`us-central1`), Secret Manager, BigQuery `VECTOR_SEARCH`, OpenTelemetry export to Cloud Trace/Logging, and Cloud Run deployment connect at Phase P4 behind a Terraform-managed monthly billing budget with alerts at **50%, 80%, and 100%**.
3. **Live SDK & Model Verification (2026-11-17):** Official Google AI documentation and PyPI metadata were audited before selecting model IDs or pinning packages:
   - Vertex AI agentic services are now organized under the **Gemini Enterprise Agent Platform** (`GOOGLE_GENAI_USE_ENTERPRISE=true` in ADK 2.x / `google-genai` 3.x).
   - `gemini-2.5-flash` and `gemini-2.5-pro` are **deprecated** effective December 1, 2026 (auto-rerouting to `gemini-3.5-flash-lite` and `gemini-3.8-flash` respectively) and must not be used.
   - `google-adk` `2.14.1` explicitly constrains `google-genai<3,>=2.30` because `google-genai` `3.4.0` switched default endpoints to `v1` (breaking `-latest` aliases) and HTTP transport to `httpx2`.
   - On the AI Studio Free Tier, `gemini-4-hydrogen`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-embedding-2`, and `gemini-embedding-001` are **free of charge**, whereas `gemini-4-carbon` and `gemini-4-argon` are Paid-Tier only.

---

## 2. Decisions

### 2.1 Agent Framework & Strict 4-Agent Ceiling (ADK Python 2.x)
We use **Google Agent Development Kit (`google-adk[mcp,otel]>=2.14,<3.0`)** with at most four specialized agents:
1. **Orchestrator (`gemini-4-hydrogen`):** Routes work, enforces human-in-the-loop (HITL) pause thresholds via ADK's Tool Confirmation flow, handles graceful degradation when downstream systems fail, and emits the final `DecisionOutput`.
2. **Extraction Agent (`gemini-3.8-flash`):** Multimodal PDF/image extraction into a strict Pydantic `InvoiceDocument` schema with bounded validation retries (`max_extraction_schema_retries: 2`).
3. **Policy RAG Agent (`gemini-3.8-flash` + `gemini-embedding-2`):** Retrieves version-valid policy clauses and returns verbatim clause IDs (e.g., `POL-07#3`) for auditable citation.
4. **Critic / Verifier (`gemini-4-hydrogen`):** Self-reflection gate added in `v4` that verifies citation existence, policy-version validity, and consistency with deterministic tool outputs (max 1 send-back).

**Deterministic Tool Boundary:** All arithmetic—line-item price tolerances, quantity variances, tax calculations, currency checks, and duplicate invoice detection—is implemented in pure, unit-tested Python functions (`src/invoiceops/tools/`). The LLM never performs tolerance arithmetic in prose.

### 2.2 Dual-Mode Storage & Auth Adapter (Option 1 Approved)
To honor the `$0.00` P1–P3 budget without sacrificing P4 production fidelity, storage and LLM authentication sit behind typed configuration and thin repository protocols (`src/invoiceops/storage/`):
- **Phases P1–P3 (`mode: local_ai_studio`):**
  - **LLM & Embeddings:** Gemini Developer API via `GEMINI_API_KEY` using only verified Free-Tier models; shadow production USD cost is still computed on every span using published Paid-Tier rates in `configs/config.yaml`.
  - **Vector & Eval Store:** Local SQLite + NumPy exact cosine search over `gemini-embedding-2` vectors, plus append-only `results/eval_runs.jsonl` + SQLite `eval_runs` mirroring the exact BigQuery table DDL.
- **Phase P4 (`mode: gcp_enterprise`, `region: us-central1`):**
  - **LLM & Embeddings:** Gemini Enterprise Agent Platform via Application Default Credentials / dedicated Cloud Run service account (`GOOGLE_GENAI_USE_ENTERPRISE=true`).
  - **Vector & Eval Store:** BigQuery tables + native `VECTOR_SEARCH` for policy clauses and BigQuery `eval_runs` for cross-version experiment reporting.

### 2.3 Mock Legacy ERP & MCP Topology (Option A Approved)
The legacy ERP (vendor master, PO lookup, goods-receipt lookup, idempotent payment-hold write, plus configurable flaky timeouts/429s/pagination/stale reads) is built as a FastAPI + MCP service (`src/invoiceops/mcp_erp/`):
- **Phases P1–P3:** Runs locally on `127.0.0.1:8001/mcp` over **Streamable HTTP MCP**.
- **Phase P4 (Two Cloud Run Services):**
  1. `invoiceops-api` (Cloud Run, public demo endpoint): Runs the ADK agent server with dedicated least-privilege service account `sa-invoiceops-agent`.
  2. `mock-erp-mcp` (Cloud Run, `--no-allow-unauthenticated` / IAM ingress): Private MCP service; `sa-invoiceops-agent` holds `roles/run.invoker` exclusively on `mock-erp-mcp` and attaches Google OIDC ID tokens on Streamable HTTP MCP calls, wrapped in client-side timeouts, exponential backoff, idempotency keys, and a circuit breaker.

---

## 3. Options Considered & Trade-offs

| Area | Option Chosen | Alternatives Rejected | Trade-off Accepted |
|---|---|---|---|
| **P1–P3 Cloud Footprint** | **Option 1:** Zero-Cost Local AI Studio + SQLite/NumPy adapter, cut over to GCP in P4 | Spin up BigQuery & Vertex AI immediately in P1; or mock LLMs with static stubs | Requires maintaining two thin storage implementations (`SqliteStore` and `BigQueryStore`), mitigated by keeping identical schemas and testing both. Saves 100% of cloud spend during 400-case × multi-run experimentation while testing real Gemini models. |
| **Model Selection** | Explicit IDs (`gemini-4-hydrogen`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-embedding-2`) in `configs/config.yaml` | `gemini-2.5-*` (deprecated 2026-12-01), `-latest` aliases, or Paid-only `gemini-4-carbon` in P1–P3 | AI Studio Free Tier enforces lower RPM/TPM rate limits than Paid Tier; evaluation harness will include configurable concurrency throttling and retry/backoff to avoid 429 noise affecting latency p50/p95 measurements. |
| **P4 ERP Deployment** | **Option A:** Two Cloud Run services with IAM-authenticated private Streamable HTTP MCP | Option B: Single container subprocess (`stdio` MCP); Option C: Unauthenticated public ERP URL | Slightly more Terraform (~25 extra lines for second service + `roles/run.invoker` binding) and network hop latency (~5–15 ms intra-region in `us-central1`), but genuinely demonstrates enterprise zero-trust service-to-service integration rather than in-process toy wiring. |
| **SDK Pinning** | `google-adk[mcp,otel]>=2.14,<3.0` (which pins `google-genai>=2.30,<3.0`) | Forcing `google-genai>=3.4.0` directly | Avoids dependency conflict with ADK 2.14.1 while preserving forward compatibility via explicit model version strings (no `v1beta`-only `-latest` aliases). |

---

## 4. Consequences & Guardrails

1. **Zero Hard-Coded Models or Regions:** Every agent, embedding call, judge, and cost calculator reads exclusively from `configs/config.yaml` via `invoiceops.config.load_config()`.
2. **Programmatic Free-Tier Gate:** While `budget_guardrails.enforce_free_tier_models_only` is `true` (Phases P1–P3), `load_config()` raises a `ValueError` at startup if any active role references a model marked `free_tier_eligible: false` (e.g., `gemini-4-carbon` or `gemini-4-argon`).
3. **Secret Hygiene:** `GEMINI_API_KEY` is read from environment/`.env` only at client initialization and redacted in `repr()`/structured logs.
