"""Unit tests for Mock Legacy ERP MCP server, flaky faults, idempotency, and circuit breaker."""

from __future__ import annotations

import pytest

from invoiceops.mcp_erp.client import (
    CircuitBreaker,
    CircuitState,
    ErpCircuitOpenError,
    ErpUnavailableError,
    ResilientErpMcpClient,
    running_local_erp_server,
)
from invoiceops.mcp_erp.flaky import FlakyInjector, FlakySettings
from invoiceops.mcp_erp.server import ErpDatabase


def _sample_erp_db() -> ErpDatabase:
    vendors = [
        {
            "vendor_id": "VEN-001",
            "name": "Apex Systems Ltd",
            "country": "US",
            "default_currency": "USD",
            "payment_terms_days": 30,
            "expected_tax_rate": 0.08,
            "status": "ACTIVE",
        }
    ]
    pos = [
        {
            "po_id": f"PO-100{i}",
            "vendor_id": "VEN-001",
            "issue_date": "2026-06-01",
            "currency": "USD",
            "status": "OPEN",
            "lines": [],
            "subtotal": 500.0,
            "total_amount": 540.0,
        }
        for i in range(1, 8)
    ]
    grns = [
        {
            "grn_id": "GRN-2001",
            "po_id": "PO-1001",
            "vendor_id": "VEN-001",
            "received_date": "2026-06-05",
            "qa_status": "ACCEPTED",
            "lines": [{"sku": "SKU-101", "received_quantity": 5, "accepted_quantity": 5}],
        }
    ]
    history = [
        {
            "history_id": "HIST-3001",
            "vendor_id": "VEN-001",
            "invoice_number": "DUP-001",
            "po_id": "PO-1001",
            "total_amount": 540.0,
            "currency": "USD",
            "status": "PAID",
        }
    ]
    return ErpDatabase(
        vendors=vendors,
        purchase_orders=pos,
        goods_receipts=grns,
        invoice_history=history,
    )


async def test_mcp_pagination_duplicate_lookup_and_idempotent_hold() -> None:
    """Verify pagination, duplicate check, and idempotent payment-hold deduplication."""
    db = _sample_erp_db()
    async with running_local_erp_server(db=db) as mcp_url:
        client = ResilientErpMcpClient(mcp_url=mcp_url, timeout_seconds=3.0, max_retries=1)

        page1 = await client.list_vendor_purchase_orders("VEN-001", page=1, page_size=5)
        page2 = await client.list_vendor_purchase_orders("VEN-001", page=2, page_size=5)
        assert page1["total_items"] == 7
        assert page1["has_next"] is True and len(page1["items"]) == 5
        assert page2["has_next"] is False and len(page2["items"]) == 2

        dup = await client.check_invoice_history("VEN-001", "DUP-001")
        assert dup["is_duplicate"] is True

        h1 = await client.place_payment_hold("INV-4001", "VEN-001", "QUANTITY_MISMATCH")
        h2 = await client.place_payment_hold("INV-4001", "VEN-001", "QUANTITY_MISMATCH")
        assert h1["hold_id"] == h2["hold_id"]
        assert h1["deduplicated"] is False
        assert h2["deduplicated"] is True
        assert len(db.holds_by_idempotency_key) == 1


async def test_mcp_client_retries_429_timeout_and_stale_reads() -> None:
    """Verify resilient MCP client recovers from HTTP 429, timeout, and stale replica reads."""
    db = _sample_erp_db()
    flaky = FlakyInjector(FlakySettings(enabled=False, timeout_sleep_seconds=0.55))
    # 1st call hits 429 -> retry ok; next hits timeout -> retry stale_read -> retry ok
    flaky.enqueue_scripted("rate_limit_429", "ok", "timeout", "ok", "stale_read", "ok")

    async with running_local_erp_server(db=db, flaky=flaky) as mcp_url:
        client = ResilientErpMcpClient(
            mcp_url=mcp_url,
            timeout_seconds=0.35,
            max_retries=3,
            base_backoff_seconds=0.03,
        )
        vendor = await client.lookup_vendor("VEN-001")
        assert vendor["found"] is True

        grn = await client.lookup_goods_receipts("PO-1001")
        assert grn["stale_replica"] is False
        assert len(grn["goods_receipts"]) == 1
        assert client.case_retries >= 3


async def test_circuit_breaker_opens_and_recovers_half_open() -> None:
    """Verify circuit breaker trips OPEN on repeated outages and recovers after cool-off."""
    clock = [100.0]
    cb = CircuitBreaker(failure_threshold=2, recovery_seconds=10.0, time_fn=lambda: clock[0])

    db = _sample_erp_db()
    flaky = FlakyInjector(FlakySettings(enabled=False))
    flaky.enqueue_scripted("rate_limit_429", "rate_limit_429", "ok")

    async with running_local_erp_server(db=db, flaky=flaky) as mcp_url:
        client = ResilientErpMcpClient(
            mcp_url=mcp_url,
            timeout_seconds=1.5,
            max_retries=0,
            circuit_breaker=cb,
        )
        with pytest.raises(ErpUnavailableError):
            await client.lookup_vendor("VEN-001")
        assert cb.state == CircuitState.CLOSED

        with pytest.raises(ErpUnavailableError):
            await client.lookup_vendor("VEN-001")
        assert cb.state == CircuitState.OPEN

        # Immediate third call fails fast with ErpCircuitOpenError without hitting network
        with pytest.raises(ErpCircuitOpenError):
            await client.lookup_vendor("VEN-001")

        # Advance monotonic clock past recovery_seconds -> HALF_OPEN -> probe succeeds -> CLOSED
        clock[0] += 12.0
        recovered = await client.lookup_vendor("VEN-001")
        assert recovered["found"] is True
        assert cb.state == CircuitState.CLOSED


def test_oidc_token_header_fetching_and_graceful_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HTTPS MCP URLs fetch OIDC tokens for root origin and degrade cleanly on error."""
    import google.oauth2.id_token as id_token_mod

    captured_audiences: list[str] = []

    def fake_fetch_id_token(_req: object, audience: str) -> str:
        captured_audiences.append(audience)
        return "mock-oidc-jwt"

    monkeypatch.setattr(id_token_mod, "fetch_id_token", fake_fetch_id_token)

    https_client = ResilientErpMcpClient(
        mcp_url="https://invoiceops-erp-ljzj6brkyq-uc.a.run.app/mcp/"
    )
    headers = https_client._build_request_headers()
    assert headers["Authorization"] == "Bearer mock-oidc-jwt"
    assert captured_audiences == ["https://invoiceops-erp-ljzj6brkyq-uc.a.run.app"]

    # Existing Authorization header should not trigger OIDC fetch
    preset_client = ResilientErpMcpClient(
        mcp_url="https://invoiceops-erp-ljzj6brkyq-uc.a.run.app/mcp/",
        auth_headers={"Authorization": "Bearer custom-token"},
    )
    assert preset_client._build_request_headers()["Authorization"] == "Bearer custom-token"
    assert len(captured_audiences) == 1

    # Auth exception falls back cleanly without raising
    def failing_fetch(_req: object, _audience: str) -> str:
        raise RuntimeError("No metadata server in local dev")

    monkeypatch.setattr(id_token_mod, "fetch_id_token", failing_fetch)
    fallback_headers = https_client._build_request_headers()
    assert "Authorization" not in fallback_headers
