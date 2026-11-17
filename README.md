# `invoiceops-agent`

A measured, multi-agent accounts-payable exception-resolution system built with Google **Agent Development Kit (ADK)**, **Model Context Protocol (MCP)**, **BigQuery Vector Search**, **OpenTelemetry**, and **Cloud Run**.

It performs three-way matching (invoice PDF/image vs. purchase order vs. goods receipt) against versioned enterprise policies and a flaky legacy ERP—delegating tolerance arithmetic to deterministic tools, enforcing human-in-the-loop (HITL) escalation thresholds, and tracking accuracy, cost-weighted error, citation validity, injection resistance, latency, and cost across controlled experiments (`v0` → `v4`).

---

## Current Phase Status

- **Phase P0 (Complete):** Standalone repository initialized, SDK/model documentation audited (2026-11-17), [ADR-001](docs/decisions/001-architecture-and-stack.md) recorded, `configs/config.yaml` and Free-Tier `$0.00` budget guardrails enforced, Ruff + Pytest CI green.
- **Phase P1 (Complete):** Seeded synthetic dataset (`make data`: 40 vendors, 600 POs/GRNs, 15 versioned policies, 400 noisy/multi-layout PDFs, frozen 60/40 split in `data/splits/frozen_split_v1.json`), FastAPI + Streamable HTTP MCP mock ERP with flaky mode and circuit-breaker client, 6-layer evaluation harness (`make eval`), [ADR-002](docs/decisions/002-evaluation-design-and-free-tier-resilience.md), and 3-run `v0` baseline recorded in `results/v0_summary.json`.
- **Phase P2 (Next):** `v1` multi-agent decomposition + deterministic matching tools + OpenTelemetry tracing.

---

## Controlled Experiment Results (`golden_12`, 3 runs, `temperature=0.0`)

All numbers below come directly from reproducible scripts (`make eval`) backed by committed JSON in [`results/v0_summary.json`](results/v0_summary.json) (mean ± 95% Student-t interval across `K = 3` runs; live AI Studio Free-Tier spend = `$0.00`, shadow USD computed at published Paid-Tier token rates):

| Version | Primary Model | Accuracy | Cost-Weighted Error | Citation Validity | Recall@5 / MRR | Injection Resistance | Latency p50 / p95 (ms) | Shadow Cost / Case (USD) |
|---|---|---|---|---|---|---|---|---|
| **`v0` (baseline)** | `gemini-3.5-flash-lite` | `61.1% ± 23.9%` | `2.694 ± 0.316` | `91.7% ± 20.7%` | `65.3% ± 6.0%` / `0.384 ± 0.109` | `100.0% ± 0.0%` | `22,321` / `44,258` | `$0.00314 ± $0.00037` |
| `v1`..`v4` | *Pending P2–P3* | — | — | — | — | — | — | — |

**`v0` Additional Layer Diagnostics (`results/v0_summary.json`):**
- **Failure Taxonomy (mean failed cases / 12-case run):** `retrieval_miss: 3.0`, `extraction_error: 1.0`, `reasoning_error: 0.67` (driven primarily by `stale_policy_version`, `late_invoice_vs_terms`, and `ambiguous_escalation` lacking version-filtered RAG and deterministic math/date tools).
- **ADK Trajectory (`results/adk_v0_golden_12.evalset.json`):** `ANY_ORDER = 0.9167` vs brittle `EXACT = 0.2500`; `response_match_score = 0.6667`.
- **Flaky-ERP Resilience (`n = 4` slice, `flaky_mode=True`):** `75.0%` task completion across `7` injected timeouts and `5` HTTP `429`s (`20` automatic client retries; **`0` unsafe false approvals**).
- **LLM-as-Judge Calibration (`data/splits/judge_calibration_40.json`):** `40` cases, raw agreement `100.0%`, Cohen's $\kappa = 1.00$ (`agent_explanation_pass_rate = 100.0%`).

---

## Quickstart (Local Zero-Cost Mode — Phases P1–P3)

1. Create and activate a Python 3.12+ virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev]"
   ```
2. Copy `.env.example` to `.env` and set your Google AI Studio `GEMINI_API_KEY` (never committed to Git).
3. Generate the seeded dataset, run lint/tests, and execute evaluation:
   ```powershell
   python scripts/tasks.py data
   python scripts/tasks.py check
   python scripts/tasks.py eval --version v0 --split golden_12 --runs 3
   # Or on Linux / macOS / Git Bash:
   make data && make check && make eval
   ```

---

## Architecture & Decisions

- [ADR-001: Architecture, Dual-Mode Execution, and Verified Google Cloud Stack](docs/decisions/001-architecture-and-stack.md)
- [ADR-002: Six-Layer Evaluation Design, ADK Trajectory Matching, and Free-Tier Model Resilience](docs/decisions/002-evaluation-design-and-free-tier-resilience.md)
- [Friction Log](docs/FRICTION_LOG.md)
- [Backlog (Out-of-Scope Ideas)](docs/BACKLOG.md)
