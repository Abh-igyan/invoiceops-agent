# Friction Log

Real surprises, SDK/documentation discrepancies, and platform quirks encountered while building `invoiceops-agent`, along with workarounds and concrete product feedback.

---

## FL-001 (2026-11-17, Phase P0): `google-genai` v3 vs `google-adk` 2.14 Dependency Pin & Enterprise Env Rename
- **What happened:** Auditing PyPI and official docs prior to P0 revealed two simultaneous shifts:
  1. `google-genai` `3.4.0` changes default Gemini Developer API endpoints from `v1beta` to `v1` (breaking `-latest` model aliases), switches default HTTP transport from `httpx` to `httpx2`, and introduces `GOOGLE_GENAI_USE_ENTERPRISE=true` for the renamed **Gemini Enterprise Agent Platform**.
  2. However, `google-adk` `2.14.1` currently pins `google-genai<3,>=2.30`, and parts of the ADK quickstart still show `model="gemini-flash-latest"` while warning in a callout that `-latest` fails on regional endpoints like `us-central1`.
- **Workaround:** Let `google-adk[mcp,otel]>=2.14,<3.0` own the `google-genai` constraint (`2.34.0`), ban all `-latest` aliases in `configs/config.yaml`, and document both `GOOGLE_GENAI_USE_ENTERPRISE` and `GOOGLE_GENAI_USE_VERTEXAI` in `.env.example`.
- **Proposal to Product Team:** Align ADK quickstart snippets to explicit stable model IDs and publish a `google-adk` 2.x migration schedule for `google-genai` 3.x (`httpx2`).

## FL-002 (2026-11-17, Phase P0): Imminent Gemini 2.5 Deprecation & Free-Tier Split in Gemini 4
- **What happened:** Live documentation shows `gemini-2.5-flash` and `gemini-2.5-pro` are deprecated effective **December 1, 2026**, and within Gemini 4, `gemini-4-carbon` and `gemini-4-argon` are Paid-Tier only while `gemini-4-hydrogen`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`, and `gemini-embedding-2` are Free-Tier eligible.
- **Workaround:** Added `free_tier_eligible` metadata to `configs/config.yaml` and a startup validator in `invoiceops.config` that blocks Paid-only models whenever `enforce_free_tier_models_only: true`.
- **Proposal to Product Team:** Expose `free_tier_eligible` and `deprecation_date` in `models.list()` metadata so frameworks can validate tier constraints before runtime.

## FL-003 (2026-11-17, Phase P0): Cross-Platform `make` Ergonomics on Windows Hosts
- **What happened:** Stock Windows PowerShell environments often lack GNU `make`.
- **Workaround:** Provided both the canonical `Makefile` and `scripts/tasks.py` exposing identical subcommands (`data`, `eval`, `check`, `deploy`, `destroy`).

## FL-004 (2026-11-17, Phase P1): `gemini-embedding-2` Collapses `list[str]` Batches into a Single Vector
- **What happened:** In `google-genai==2.34.0`, calling `client.models.embed_content(model="gemini-embedding-001", contents=["one", "two", "three"])` returns **3** embeddings (`1:1`), whereas the identical call on multimodal `gemini-embedding-2` treats `list[str]` as a single multi-part document and returns **1** combined vector! Caught immediately by Python's `zip(to_embed, new_vectors, strict=True)`.
- **Workaround:** Wrap each batch element explicitly as `types.Content(parts=[types.Part.from_text(text=t)])`, which consistently returns `N` embeddings for `N` strings across both `gemini-embedding-2` and `gemini-embedding-001`.
- **Proposal to Product Team:** Either warn when `list[str]` is passed to multimodal embedding models or provide an explicit `embed_batch(documents=[...])` helper so migrating RAG pipelines from `gemini-embedding-001` to `gemini-embedding-2` cannot silently collapse chunk batches.

## FL-005 (2026-11-17, Phase P1): Free-Tier 503 Load-Shedding, 5-RPM Preview Quotas, and Minimum Server Deadlines
- **What happened:** Live Free-Tier testing revealed three operational constraints:
  1. Setting `HttpOptions(timeout=8000)` raises `400 INVALID_ARGUMENT: Manually set deadline 8s is too short` (server requires `>=10s`).
  2. `gemini-4-hydrogen` and `gemini-3.8-flash` shed Free-Tier traffic with `503 UNAVAILABLE` during global peak hours, and `gemini-3-flash-preview` enforces a `5 RPM` Free-Tier limit (`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`).
  3. Stable `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite` serve reliably at higher RPM.
- **Workaround:** Configured `gemini-3.5-flash-lite` as the Free-Tier baseline primary with automatic model fallback (`gemini-3.1-flash-lite`, `gemini-3-flash-preview`), inter-case pacing (`2.0s`), and JSONL checkpoint-resume in `harness.py`.

## FL-006 (2026-11-17, Phase P1): FastAPI Sub-Mount Lifespan & MCP Handshake Multi-POST Behavior
- **What happened:** Mounting `MCPServer.streamable_http_app()` onto FastAPI via `app.mount("/mcp", mcp_http_app)` does not trigger the sub-app's Starlette lifespan automatically, and each stateless `ClientSession` tool invocation issues multiple HTTP POSTs (`initialize`, `notifications/initialized`, `tools/call`, `tools/list`).
- **Workaround:** Forwarded FastAPI's `lifespan` into `mcp_http_app.router.lifespan_context(mcp_http_app)` and scoped HTTP flaky fault injection specifically to `tools/call` JSON-RPC requests so fault probabilities map `1:1` to logical tool invocations.

## FL-007 (2026-11-18, Phase P3): Multi-Part ADK `Content.parts` Newline Injection Inside JSON String Literals
- **What happened:** In `v2` Run 1 (`INV-4219`), `orchestrator_agent` produced the exact right decision (`reject`) and citations (`POL-01#2`, `VEN-019`), yet failed evaluation with `SCHEMA_VALIDATION_FAILED` (`extraction_error`). Inspecting `results/eval_runs.jsonl` revealed `"reason_codes": ["MISSING_OR_INVALID\n_PO"]`: when Gemini returns multiple contiguous text `Part`s in a single ADK event (or wraps long explanations mid-string), joining `Part.text` segments with `"\n".join(texts)` injects an unescaped literal `\n` (`0x0A`) inside a JSON string literal, which RFC 8259 / `json.loads` rejects with `Invalid control character`.
- **Workaround:** Joined contiguous `Part.text` segments within an ADK event via `"".join(texts)` in `v4`, added state-machine string sanitizer `sanitize_multiline_json_text()` in `src/invoiceops/tools/security.py` to replace raw in-string `\n`/`\r` characters prior to `json.loads`, and normalized accidental intra-code whitespace in `UPPER_SNAKE_CASE` reason codes.
- **Proposal to Product Team:** Provide an official `event.text` convenience property on ADK `Event` that concatenates contiguous text parts without delimiter injection while ignoring `thought` parts.

## FL-008 (2026-11-20, Phase P4): FastAPI / Starlette `TestClient` Lifespan Execution & Cloud Run Probe Timing
- **What happened:** In Starlette `TestClient(app)`, route requests do not trigger the FastAPI `lifespan` context manager unless the client itself is entered as a Python context manager (`with TestClient(app) as client:`). Accessing `request.app.state.svc` inside `/health` during unit tests raised `AttributeError: 'State' object has no attribute 'svc'`.
- **Workaround:** Made `/health` defensive using `getattr(request.app.state, 'svc', None)` with a fallback config version string, and updated `tests/test_server.py` fixtures to use `with TestClient(srv_mod.app) as c: yield c`.
- **Proposal to Product Team:** Document explicitly in the ADK/FastAPI serving guide that `TestClient` fixtures must wrap the app in a context manager to execute async lifespan startup hooks.


