# `invoiceops-agent`

A measured, multi-agent accounts-payable exception-resolution system built with Google **Agent Development Kit (ADK)**, **Model Context Protocol (MCP)**, **BigQuery Vector Search**, **OpenTelemetry**, and **Cloud Run**.

It performs three-way matching (invoice PDF/image vs. purchase order vs. goods receipt) against versioned enterprise policies and a flaky legacy ERP—delegating all tolerance arithmetic to deterministic tools, enforcing human-in-the-loop (HITL) escalation thresholds, and tracking accuracy, cost-weighted error, citation validity, injection resistance, latency, and cost across controlled experiments (`v0` → `v4`).

---

## Current Phase Status

- **Phase P0 (Complete):** Standalone repository initialized, SDK/model documentation audited (2026-11-17), [ADR-001](docs/decisions/001-architecture-and-stack.md) recorded, `configs/config.yaml` and Free-Tier `$0.00` budget guardrails enforced, Ruff + Pytest CI green.
- **Phase P1 (Next):** Seeded synthetic dataset (`make data`), mock ERP MCP server with flaky mode, evaluation harness (`make eval`), and `v0` baseline measurement.
- **Results Table (`v0`..`v4`):** Populated strictly from reproducible `results/*.json` outputs beginning in Phase P1 (no unmeasured numbers claimed).

---

## Quickstart (Local Zero-Cost Mode — Phases P1–P3)

1. Create and activate a Python 3.12+ virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev]"
   ```
2. Copy `.env.example` to `.env` and set your Google AI Studio `GEMINI_API_KEY` (never committed to Git).
3. Verify lint, formatting, and unit tests:
   ```powershell
   python scripts/tasks.py check
   # Or on Linux / macOS / Git Bash:
   make check
   ```

---

## Architecture & Decisions

- [ADR-001: Architecture, Dual-Mode Execution, and Verified Google Cloud Stack](docs/decisions/001-architecture-and-stack.md)
- [Friction Log](docs/FRICTION_LOG.md)
- [Backlog (Out-of-Scope Ideas)](docs/BACKLOG.md)
