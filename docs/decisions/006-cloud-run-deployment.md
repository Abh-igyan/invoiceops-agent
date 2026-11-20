# ADR-006: Cloud Run Two-Service Deployment Topology

**Status:** Accepted
**Date:** 2024-11-20
**Deciders:** Abhigyan

## Context

Phase P4 requires deploying the invoiceops-agent to Google Cloud for a live demo URL (Section 8). The system has two runtime components: the multi-agent pipeline (FastAPI + v4 evaluator) and the mock legacy ERP (MCP server). We need to choose a deployment topology, IAM model, and infrastructure-as-code approach.

## Options Considered

1. **Single Cloud Run service** — both agent and ERP in one container.
2. **Two Cloud Run services** — agent (public) and ERP (internal), separate containers.
3. **Agent Engine (managed)** — deploy to Agent Engine instead of Cloud Run.

## Decision

**Option 2: Two-service Cloud Run topology.**

### Rationale

- **Separation of concerns:** The agent service handles public requests and LLM orchestration; the ERP service simulates the legacy system the agent integrates with. Keeping them separate mirrors real production topology where the ERP is a separate backend.
- **Independent scaling:** The agent service needs more CPU/memory (LLM calls, PDF processing) while the ERP is lightweight. Independent scaling avoids over-provisioning.
- **Security boundary:** The ERP service is internal-only (no `allUsers` IAM binding), accessible only by the agent's service account. This demonstrates least-privilege IAM — a key production engineering skill.
- **FastAPI over ADK's server:** We use a FastAPI wrapper around the v4 evaluator rather than ADK's built-in server for three reasons: (1) explicit control over request IDs, structured logging, and health checks; (2) multipart PDF upload support; (3) configurable timeouts via environment variable.

### IAM Roles (Least-Privilege)

| Role | Justification |
|---|---|
| `bigquery.dataEditor` | Write eval rows to `eval_runs` table |
| `bigquery.jobUser` | Execute BigQuery queries for RAG vector search |
| `secretmanager.secretAccessor` | Read `GEMINI_API_KEY` at container startup |
| `cloudtrace.agent` | Export OpenTelemetry spans to Cloud Trace |
| `logging.logWriter` | Structured JSON log export to Cloud Logging |

### Infrastructure

- **Terraform** manages all resources: service account, IAM bindings, Secret Manager secret, BigQuery dataset + 2 tables, 2 Cloud Run v2 services, billing budget.
- **Billing budget:** $10/month with alerts at 50%, 80%, and 100%.
- **`make destroy`** (terraform destroy) tears down everything to prevent accidental spend.

## Trade-offs

- Two services means two container images to build and push, adding CI complexity.
- The ERP runs as a Cloud Run service rather than a sidecar container — slightly higher latency for tool calls vs. localhost, but more realistic for demonstrating production integration patterns.
- Agent Engine was rejected because it is a managed service that would hide the deployment engineering decisions this portfolio project is designed to showcase.

## Consequences

- CI pipeline gains a `terraform-validate` job and an `eval.yml` workflow dispatch.
- A load test script validates concurrency behavior against the Cloud Run URL.
- The ERP service URL must be configured in the agent's environment (or config) when deployed.
