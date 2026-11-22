"""FastAPI production server wrapping the v4 agent pipeline for Cloud Run (Section 8)."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.security import APIKeyHeader

from invoiceops.config import AppConfig, load_config, load_runtime_secrets
from invoiceops.mcp_erp.client import ResilientErpMcpClient
from invoiceops.storage.local_store import LocalSqliteStore
from invoiceops.telemetry.tracing import InvoiceOpsTelemetry

logger = logging.getLogger("invoiceops.server")

REPO_ROOT = Path(__file__).resolve().parents[2]
_TIMEOUT_SECONDS = float(os.getenv("INVOICEOPS_TIMEOUT_SECONDS", "120"))


def _setup_logging() -> None:
    """Configure structured JSON logging for Cloud Run."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class ServerState:
    """Shared application state initialized during lifespan."""

    config: AppConfig
    erp_client: ResilientErpMcpClient
    store: LocalSqliteStore
    telemetry: InvoiceOpsTelemetry
    embed_fn: Any


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load config, init ERP client, embedder, store, and telemetry on startup."""
    _setup_logging()
    logger.info("Starting invoiceops-agent server")

    state = ServerState()
    state.config = load_config()
    secrets = load_runtime_secrets()

    # Telemetry
    trace_path = REPO_ROOT / "results" / "server_traces.jsonl"
    state.telemetry = InvoiceOpsTelemetry(
        config=state.config,
        jsonl_trace_path=trace_path,
        keep_in_memory=False,
    )

    # Policy store
    db_path = REPO_ROOT / state.config.storage.local_sqlite_path
    state.store = LocalSqliteStore(db_path=db_path)

    # Embedding function
    if secrets.has_gemini_key():
        from invoiceops.storage.local_store import GeminiEmbedder

        embedder = GeminiEmbedder(config=state.config)

        def embed_query(text: str) -> list[float]:
            return embedder.embed_texts([text])[0]

        state.embed_fn = embed_query
    else:

        def fallback_embed(text: str) -> list[float]:
            return [0.0] * 768

        state.embed_fn = fallback_embed

    # ERP client (connects to the sidecar MCP service)
    mcp_url = os.getenv("MOCK_ERP_BASE_URL") or state.config.mock_erp.base_url
    state.erp_client = ResilientErpMcpClient(
        mcp_url=mcp_url,
        timeout_seconds=state.config.mock_erp.client_timeout_seconds,
        max_retries=state.config.mock_erp.max_retries,
    )

    app.state.svc = state
    logger.info(
        "Server ready config_version=%s mode=%s",
        state.config.config_version,
        state.config.platform.mode,
    )
    yield

    state.telemetry.provider.shutdown()
    logger.info("Server shutdown complete")


DEFAULT_DEMO_API_KEY = "invoiceops-demo-2026"
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API Key for protected endpoints (Demo key: `invoiceops-demo-2026`)",
)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Validate X-API-Key header against server secret or demo key using constant time compare."""
    expected_key = os.getenv("INVOICEOPS_API_KEY", DEFAULT_DEMO_API_KEY)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing X-API-Key header. For demo evaluation, "
                f"use demo key: '{DEFAULT_DEMO_API_KEY}'"
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )
    key_bytes = api_key.strip().encode("utf-8")
    expected_bytes = expected_key.strip().encode("utf-8")
    if not secrets.compare_digest(key_bytes, expected_bytes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid X-API-Key header. Access denied.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


app = FastAPI(
    title="InvoiceOps Agent",
    version="0.1.0",
    description=(
        "Production multi-agent accounts-payable exception resolution on Google Cloud Run.\n\n"
        "**Authentication:** Protected evaluation endpoints require an `X-API-Key` header. "
        f"For testing, use demo key: `{DEFAULT_DEMO_API_KEY}`."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    """Inject X-Request-ID header (use incoming value or generate UUID)."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness and readiness probe for Cloud Run."""
    svc: ServerState | None = getattr(request.app.state, "svc", None)
    config_ver = svc.config.config_version if svc and hasattr(svc, "config") else "0.1.0"
    return {
        "status": "ok",
        "version": "0.1.0",
        "config_version": config_ver,
    }


@app.post("/v1/invoices/evaluate", dependencies=[Depends(verify_api_key)])
async def evaluate_invoice(
    pdf: UploadFile = File(..., description="Invoice PDF document"),  # noqa: B008
    case_id: str = Form(..., description="Unique case identifier"),
    vendor_id: str = Form(..., description="Vendor identifier from ERP"),
) -> dict[str, Any]:
    """Evaluate an invoice PDF through the v4 multi-agent pipeline."""
    request_id = str(uuid.uuid4())
    svc: ServerState = app.state.svc
    t0 = time.perf_counter()

    logger.info(
        "Evaluating invoice request_id=%s case_id=%s vendor_id=%s",
        request_id,
        case_id,
        vendor_id,
    )

    # Save uploaded PDF to temp location
    pdf_bytes = await pdf.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF file")

    tmp_dir = REPO_ROOT / "tmp" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{request_id}.pdf"
    tmp_path.write_bytes(pdf_bytes)

    try:
        from invoiceops.agents.v4_critic_reliability import V4CriticReliabilityEvaluator

        evaluator = V4CriticReliabilityEvaluator(
            config=svc.config,
            erp_client=svc.erp_client,
            store=svc.store,
            embed_query_fn=svc.embed_fn,
            telemetry=svc.telemetry,
        )

        result = await asyncio.wait_for(
            evaluator.evaluate_invoice(
                case_id=case_id,
                vendor_id=vendor_id,
                pdf_path=tmp_path,
            ),
            timeout=_TIMEOUT_SECONDS,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "Invoice evaluated request_id=%s decision=%s latency_ms=%.2f",
            request_id,
            result.output.decision,
            elapsed_ms,
        )

        return {
            "request_id": request_id,
            "decision": result.output.model_dump(),
            "explanation": result.explanation,
            "telemetry": {
                "latency_ms": elapsed_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
                "llm_calls": result.llm_calls,
                "tool_calls": result.tool_calls,
            },
        }

    except TimeoutError as exc:
        logger.warning("Request timed out request_id=%s", request_id)
        raise HTTPException(
            status_code=504,
            detail=f"Evaluation timed out after {_TIMEOUT_SECONDS}s",
        ) from exc
    except Exception as exc:
        logger.exception("Evaluation failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Internal evaluation error") from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> None:
    """Entry point for `python -m invoiceops.server` or Docker CMD."""
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "invoiceops.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
