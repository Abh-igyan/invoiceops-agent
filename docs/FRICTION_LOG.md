# Friction Log

Real surprises, SDK/documentation discrepancies, and platform quirks encountered while building `invoiceops-agent`, along with workarounds and concrete product feedback.

---

## FL-001 (2026-11-17, Phase P0): `google-genai` v3 vs `google-adk` 2.14 Dependency Pin & Enterprise Env Rename
- **What happened:** Auditing PyPI and official docs prior to P0 revealed two simultaneous shifts:
  1. `google-genai` `3.4.0` is the latest release on PyPI and changes default Gemini Developer API endpoints from `v1beta` to `v1` (breaking `-latest` model aliases), switches default HTTP transport from `httpx` to `httpx2`, and introduces `GOOGLE_GENAI_USE_ENTERPRISE=true` for the renamed **Gemini Enterprise Agent Platform** (formerly Vertex AI).
  2. However, `google-adk` `2.14.1` currently pins `google-genai<3,>=2.30` in its package metadata, and parts of the ADK quickstart still show `model="gemini-flash-latest"` while warning in a callout that `-latest` fails on regional endpoints like `us-central1`.
- **Workaround:** Let `google-adk[mcp,otel]>=2.14,<3.0` own the `google-genai` version constraint (`2.x`), ban all `-latest` aliases in favor of explicit versioned IDs in `configs/config.yaml`, and document both `GOOGLE_GENAI_USE_ENTERPRISE` and `GOOGLE_GENAI_USE_VERTEXAI` in `.env.example` for smooth P4 cutover.
- **Proposal to Product Team:** Align ADK quickstart snippets to explicit stable model IDs so regional (`us-central1`) and `v1` users do not hit immediate 404s, and publish a compatibility matrix for `google-adk` 2.x migration to `google-genai` 3.x (`httpx2`).

## FL-002 (2026-11-17, Phase P0): Imminent Gemini 2.5 Model Deprecation & Free-Tier Split in Gemini 4
- **What happened:** Older tutorials and memory commonly reference `gemini-2.5-flash` and `gemini-2.5-pro`. Live documentation (`https://aistudio.google.com/docs/models` and `/pricing`) shows both are deprecated and auto-rerouting on **December 1, 2026** (two weeks away), and within the new Gemini 4 family, `gemini-4-carbon` and `gemini-4-argon` are Paid-Tier only while `gemini-4-hydrogen`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`, and `gemini-embedding-2` are Free-Tier eligible.
- **Workaround:** Encoded `free_tier_eligible` metadata directly into `configs/config.yaml` and added a startup guardrail in `invoiceops.config` that rejects non-free-tier models whenever `enforce_free_tier_models_only: true`.
- **Proposal to Product Team:** Expose `free_tier_eligible` and `deprecation_date` fields in the `models.list()` SDK response so agent frameworks can warn or fail fast programmatically before incurring unexpected billing or silent model reroutes.

## FL-003 (2026-11-17, Phase P0): Cross-Platform `make` Ergonomics on Windows Developer Hosts
- **What happened:** The project brief standardizes on `make data`, `make eval`, `make deploy`, and `make destroy`, which run natively in Linux CI and Cloud Shell, but stock Windows PowerShell environments often lack GNU `make`.
- **Workaround:** Provided both the canonical `Makefile` (used by Linux/macOS/GitHub Actions) and a thin zero-dependency `scripts/tasks.py` dispatcher exposing identical subcommands (`python scripts/tasks.py lint|test|data|eval|deploy|destroy`) on Windows.
