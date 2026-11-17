# ADR-002: Six-Layer Evaluation Design, ADK Trajectory Matching, and Free-Tier Model Resilience

- **Status:** Accepted
- **Date:** 2026-11-17
- **Phase:** P1 (Dataset, Mock ERP + MCP, Evaluation Harness, `v0` Baseline)

---

## 1. Context

Phase P1 builds the frozen measurement yardstick before iterating on multi-agent architecture (`v1`), hybrid retrieval (`v2`), model routing (`v3`), and self-reflection/injection hardening (`v4`). During live integration testing against the Google AI Studio Free Tier (`$0.00` budget guardrail), three concrete platform behaviors emerged:
1. **Free-Tier HTTP 503 Load-Shedding & 5-RPM Preview Caps:** `gemini-4-hydrogen` and `gemini-3.8-flash` intermittently return `503 UNAVAILABLE` during peak global traffic on Free Tier, and `gemini-3-flash-preview` enforces a strict `5 RPM` Free-Tier limit (`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`). Stable `gemini-3.5-flash-lite` (`$0.30/$2.50` shadow rate) and `gemini-3.1-flash-lite` (`$0.25/$1.50`) sustain reliable multi-turn agent throughput.
2. **Multimodal Embedding Batching (`gemini-embedding-2`):** Passing a bare `list[str]` to `embed_content(model="gemini-embedding-2", ...)` collapses the batch into **1** multimodal embedding vector rather than returning `N` vectors (unlike `gemini-embedding-001`); wrapping each chunk as `types.Content(parts=[types.Part.from_text(...)])` restores `1:1` batch vector output.
3. **ADK `TrajectoryEvaluator` Exact-Match Brittleness:** ADK's default `ToolTrajectoryCriterion.MatchType.EXACT` scores `0.0` whenever an LLM calls independent verification tools in a different valid order (e.g., `lookup_purchase_order` before `lookup_vendor`) or emits varied search queries.

---

## 2. Decisions

### 2.1 Seeded Dataset & Stratified Frozen Split (`data/splits/frozen_split_v1.json`)
- **Scale & Seed (`seed=42`):** 40 vendors, 600 POs, 600 GRNs, 15 policy Markdown documents (12 active + 3 superseded `v1.0` trap policies), and 400 rendered PDF invoices across 3 visual layouts (`classic_corporate`, `modern_minimal`, `legacy_tabular`), with ~33.5% rasterized as skewed/blurred scans.
- **12 Controlled Scenarios:** `clean_match` (80), `price_variance_within_tolerance` (30), `price_variance_over_tolerance` (40), `quantity_mismatch` (35), `duplicate_invoice` (30), `missing_po` (30), `currency_mismatch` (25), `tax_mismatch` (30), `late_invoice_vs_terms` (25), `stale_policy_version` (25), `ambiguous_escalation` (25), and `prompt_injection` (25).
- **Stratified Split Hierarchy:**
  - `dev` (**240 cases**, 60% stratified across all 12 modes) — open for iterative tuning.
  - `test` (**160 cases**, 40% stratified) — **frozen** and untouched during tuning; reserved strictly for milestone verification.
  - `dev_stratified_36` (**36 cases**, 3 per mode), `dev_stratified_24` (**24 cases**, 2 per mode), and `golden_12` (**12 cases**, 1 per mode) — deterministic subsets recorded inside `frozen_split_v1.json` so 3-repetition Free-Tier evaluations complete within daily AI Studio RPD quotas without sampling bias.

### 2.2 Six-Layer Evaluation Yardstick
1. **Outcome & Asymmetric Cost:** Exact decision accuracy plus configurable cost-weighted error (`false_approval = 10.0`, `false_rejection = 3.0`, `false_hold = 2.0`, `false_escalation = 1.0`) reflecting real AP financial risk.
2. **ADK Trajectory (`golden_12`):** Exports ADK-validated `.evalset.json` (`google.adk.evaluation.eval_set.EvalSet`) and runs `TrajectoryEvaluator` under both `ANY_ORDER` (`ignore_args=True`, primary verification completeness metric) and `EXACT` (logged side-by-side to quantify exact-order brittleness).
3. **Retrieval & Citation Validity:** Multi-query top-`k` `Recall@5` and best-rank `MRR` over retrieved policy clause IDs, plus strict `citation_validity_rate` (every cited `policy_clause` must exist in *active* policy docs—citing superseded `V1` clauses fails validation—and every `erp_record` ID must exist in ERP state).
4. **Safety & Flaky-ERP Resilience:** `injection_resistance_rate` on adversarial invoices, plus a dedicated `flaky_mode=True` robustness slice (`golden_12[:4]`, 4 representative cases) injecting HTTP `429`s, gateway timeouts, and stale replica reads against `ResilientErpMcpClient`.
5. **Calibrated LLM-as-Judge:** Calibrated on 40 reference-labeled explanations (`data/splits/judge_calibration_40.json`) using **Cohen's Kappa ($\kappa$)**; gated at $\kappa \ge 0.65$.
6. **Operational & Failure Taxonomy:** Latency `p50`/`p95`, input/output tokens, shadow Paid-Tier USD/case, and automatic classification of every failed case into 7 failure categories (`injection_success`, `tool_erp_failure`, `policy_version_error`, `retrieval_miss`, `extraction_error`, `wrong_tool`, `reasoning_error`), aggregated as **mean ± 95% Student-t CI** over `K = 3` runs at `temperature = 0.0`.

---

## 3. Trade-offs

- **Primary Model Choice (`gemini-3.5-flash-lite` + `gemini-3.1-flash-lite` / `gemini-3-flash-preview` fallback):** Using `gemini-3.5-flash-lite` as the active Free-Tier `v0` primary avoids both `503` load-shedding on `gemini-4-hydrogen` / `gemini-3.8-flash` and the `5 RPM` preview bottleneck on `gemini-3-flash-preview`, while preserving clean `v3` model-routing experiments (`gemini-3.1-flash-lite` `$0.25/$1.50` for fast extraction/routing vs `gemini-3.5-flash-lite` `$0.30/$2.50` or selective `gemini-3-flash-preview` `$0.50/$3.00` escalation). In Phase P4 on GCP Enterprise tier, `configs/config.yaml` swaps to enterprise endpoints with zero code edits.
- **Stateless Streamable HTTP MCP (`stateless_http=True`):** Slightly higher per-call JSON-RPC handshake overhead on localhost (~3–5 ms), but maps 1-to-1 to scale-to-zero Cloud Run horizontal instances in Phase P4 without sticky-session state loss.
