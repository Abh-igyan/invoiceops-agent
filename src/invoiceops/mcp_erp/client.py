"""Hardened Streamable HTTP MCP client (timeouts, retries, idempotency, circuit breaker)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

import httpx2
import uvicorn
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from invoiceops.mcp_erp.flaky import FlakyInjector
from invoiceops.mcp_erp.server import ErpDatabase, create_erp_app


class ErpUnavailableError(RuntimeError):
    """Raised when an MCP ERP call fails after exhausting retries."""


class ErpCircuitOpenError(ErpUnavailableError):
    """Raised immediately when the ERP circuit breaker is OPEN."""


class CircuitState(StrEnum):
    """States of the client-side ERP circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Simple three-state circuit breaker guarding flaky ERP calls."""

    failure_threshold: int = 5
    recovery_seconds: float = 30.0
    time_fn: Callable[[], float] = time.monotonic
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0

    def before_call(self) -> None:
        """Verify whether calls are permitted or transition OPEN -> HALF_OPEN."""
        if self.state == CircuitState.OPEN:
            elapsed = self.time_fn() - self.opened_at
            if elapsed >= self.recovery_seconds:
                self.state = CircuitState.HALF_OPEN
            else:
                msg = (
                    f"ERP circuit breaker is OPEN ({self.consecutive_failures} consecutive "
                    f"failures; {self.recovery_seconds - elapsed:.1f}s remaining)."
                )
                raise ErpCircuitOpenError(msg)

    def record_success(self) -> None:
        """Reset failure count and close the circuit."""
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Increment failure count and trip OPEN when threshold is reached."""
        self.consecutive_failures += 1
        if (
            self.state == CircuitState.HALF_OPEN
            or self.consecutive_failures >= self.failure_threshold
        ):
            self.state = CircuitState.OPEN
            self.opened_at = self.time_fn()


def make_hold_idempotency_key(invoice_id: str, vendor_id: str, reason_code: str) -> str:
    """Generate a deterministic idempotency key for ERP payment-hold writes."""
    digest = hashlib.sha256(f"{invoice_id}:{vendor_id}:{reason_code}".encode()).hexdigest()
    return f"idem-{digest[:20]}"


@dataclass
class ResilientErpMcpClient:
    """MCP client wrapper adding timeout, exponential backoff, stale-read retry, and CB."""

    mcp_url: str
    timeout_seconds: float = 5.0
    max_retries: int = 3
    base_backoff_seconds: float = 0.12
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    asgi_app: FastAPI | None = None
    auth_headers: dict[str, str] = field(default_factory=dict)

    case_tool_calls: int = 0
    case_retries: int = 0
    tools_invoked: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize MCP endpoint URL to ensure trailing slash and prevent 307 redirects."""
        if self.mcp_url and not self.mcp_url.endswith("/"):
            self.mcp_url = f"{self.mcp_url}/"

    def reset_case_telemetry(self) -> None:
        """Clear per-case counters prior to evaluating a new invoice."""
        self.case_tool_calls = 0
        self.case_retries = 0
        self.tools_invoked.clear()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke an MCP tool with timeout, backoff retries, stale-read detection, and CB."""
        self.circuit_breaker.before_call()
        self.case_tool_calls += 1
        self.tools_invoked.append(tool_name)

        last_exc: Exception | None = None
        total_attempts = 1 + self.max_retries

        for attempt in range(total_attempts):
            if attempt > 0:
                self.case_retries += 1
                backoff = self.base_backoff_seconds * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

            try:
                payload = await asyncio.wait_for(
                    self._invoke_once(tool_name=tool_name, arguments=arguments),
                    timeout=self.timeout_seconds,
                )
                if payload.get("stale_replica") is True and attempt + 1 < total_attempts:
                    last_exc = ErpUnavailableError("Stale ERP replica read detected")
                    continue

                self.circuit_breaker.record_success()
                return payload
            except Exception as exc:
                last_exc = exc
                continue

        self.circuit_breaker.record_failure()
        msg = f"MCP tool '{tool_name}' failed after {total_attempts} attempts: {last_exc}"
        raise ErpUnavailableError(msg) from last_exc

    def _build_request_headers(self) -> dict[str, str]:
        """Build HTTP headers, fetching a Cloud Run OIDC ID token for HTTPS endpoints if needed."""
        headers = dict(self.auth_headers)
        has_auth = any(k.lower() == "authorization" for k in headers)
        if self.mcp_url.startswith("https://") and not has_auth and self.asgi_app is None:
            try:
                from google.auth.transport.requests import Request
                from google.oauth2 import id_token

                parsed = urlsplit(self.mcp_url)
                audience = f"{parsed.scheme}://{parsed.netloc}"
                token = id_token.fetch_id_token(Request(), audience)
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Cloud Run OIDC token fetch skipped/fallback for %s: %s",
                    self.mcp_url,
                    exc,
                )
        return headers

    async def _invoke_once(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Open a stateless Streamable HTTP MCP exchange and parse structured JSON output."""
        transport = httpx2.ASGITransport(app=self.asgi_app) if self.asgi_app is not None else None
        req_headers = await asyncio.to_thread(self._build_request_headers)
        async with (
            httpx2.AsyncClient(
                transport=transport,
                headers=req_headers or None,
                follow_redirects=True,
                timeout=httpx2.Timeout(self.timeout_seconds),
            ) as http_client,
            streamable_http_client(
                url=self.mcp_url,
                http_client=http_client,
                terminate_on_close=False,
            ) as streams,
        ):
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)

        if getattr(result, "isError", False):
            text_err = _extract_text(result)
            raise ErpUnavailableError(f"MCP tool returned error: {text_err}")

        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            # MCP 2.x wraps dict return values directly or under 'result'
            if "result" in structured and isinstance(structured["result"], dict):
                return dict(structured["result"])
            return dict(structured)

        raw_txt = _extract_text(result)
        parsed = json.loads(raw_txt)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    async def lookup_vendor(self, vendor_id: str) -> dict[str, Any]:
        return await self.call_tool("lookup_vendor", {"vendor_id": vendor_id})

    async def lookup_purchase_order(self, po_id: str) -> dict[str, Any]:
        return await self.call_tool("lookup_purchase_order", {"po_id": po_id})

    async def list_vendor_purchase_orders(
        self,
        vendor_id: str,
        page: int = 1,
        page_size: int = 5,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "list_vendor_purchase_orders",
            {"vendor_id": vendor_id, "page": page, "page_size": page_size},
        )

    async def lookup_goods_receipts(self, po_id: str) -> dict[str, Any]:
        return await self.call_tool("lookup_goods_receipts", {"po_id": po_id})

    async def check_invoice_history(
        self,
        vendor_id: str,
        invoice_number: str,
    ) -> dict[str, Any]:
        return await self.call_tool(
            "check_invoice_history",
            {"vendor_id": vendor_id, "invoice_number": invoice_number},
        )

    async def place_payment_hold(
        self,
        invoice_id: str,
        vendor_id: str,
        reason_code: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        idem = idempotency_key or make_hold_idempotency_key(invoice_id, vendor_id, reason_code)
        return await self.call_tool(
            "place_payment_hold",
            {
                "invoice_id": invoice_id,
                "vendor_id": vendor_id,
                "reason_code": reason_code,
                "idempotency_key": idem,
            },
        )


def _extract_text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    parts = [getattr(item, "text", "") for item in content if getattr(item, "text", None)]
    return "\n".join(parts) if parts else "{}"


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@asynccontextmanager
async def running_local_erp_server(
    db: ErpDatabase,
    flaky: FlakyInjector | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> AsyncIterator[str]:
    """Start a real background Uvicorn HTTP MCP server and yield its `/mcp/` URL."""
    chosen_port = port or _pick_free_port(host)
    app = create_erp_app(db=db, flaky=flaky)
    uvi_cfg = uvicorn.Config(app=app, host=host, port=chosen_port, log_level="error")
    server = uvicorn.Server(uvi_cfg)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.03)
        yield f"http://{host}:{chosen_port}/mcp/"
    finally:
        server.should_exit = True
        await task
