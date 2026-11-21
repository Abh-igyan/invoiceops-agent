"""FastAPI + Streamable HTTP MCP server simulating a legacy Accounts Payable ERP."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

from invoiceops.mcp_erp.flaky import FlakyInjector, FlakySettings


class ErpDatabase:
    """In-memory relational ledger loaded from `erp_seed.json`."""

    def __init__(
        self,
        vendors: list[dict[str, Any]],
        purchase_orders: list[dict[str, Any]],
        goods_receipts: list[dict[str, Any]],
        invoice_history: list[dict[str, Any]],
    ) -> None:
        self.vendors = {v["vendor_id"]: v for v in vendors}
        self.purchase_orders = {p["po_id"]: p for p in purchase_orders}
        self.pos_by_vendor: dict[str, list[dict[str, Any]]] = {}
        for po in purchase_orders:
            self.pos_by_vendor.setdefault(po["vendor_id"], []).append(po)
        self.grns_by_po: dict[str, list[dict[str, Any]]] = {}
        for grn in goods_receipts:
            self.grns_by_po.setdefault(grn["po_id"], []).append(grn)
        self.history_by_vendor: dict[str, list[dict[str, Any]]] = {}
        for item in invoice_history:
            self.history_by_vendor.setdefault(item["vendor_id"], []).append(item)
        self.holds_by_idempotency_key: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_seed_file(cls, seed_path: Path) -> ErpDatabase:
        """Load ERP state from a generated `erp_seed.json` file."""
        raw = json.loads(seed_path.read_text(encoding="utf-8"))
        return cls(
            vendors=raw["vendors"],
            purchase_orders=raw["purchase_orders"],
            goods_receipts=raw["goods_receipts"],
            invoice_history=raw["invoice_history"],
        )

    def lookup_vendor(self, vendor_id: str) -> dict[str, Any]:
        vendor = self.vendors.get(vendor_id)
        if not vendor:
            return {"found": False, "vendor_id": vendor_id, "error": "VENDOR_NOT_FOUND"}
        return {"found": True, "vendor": vendor}

    def lookup_purchase_order(self, po_id: str) -> dict[str, Any]:
        po = self.purchase_orders.get(po_id)
        if not po:
            return {"found": False, "po_id": po_id, "error": "PO_NOT_FOUND"}
        return {"found": True, "purchase_order": po}

    def list_vendor_purchase_orders(
        self,
        vendor_id: str,
        page: int = 1,
        page_size: int = 5,
    ) -> dict[str, Any]:
        all_pos = sorted(self.pos_by_vendor.get(vendor_id, []), key=lambda x: x["po_id"])
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 25))
        start = (safe_page - 1) * safe_size
        end = start + safe_size
        slice_items = all_pos[start:end]
        return {
            "vendor_id": vendor_id,
            "page": safe_page,
            "page_size": safe_size,
            "total_items": len(all_pos),
            "has_next": end < len(all_pos),
            "items": [
                {
                    "po_id": p["po_id"],
                    "currency": p["currency"],
                    "total_amount": p["total_amount"],
                    "status": p["status"],
                }
                for p in slice_items
            ],
        }

    def lookup_goods_receipts(self, po_id: str, stale: bool = False) -> dict[str, Any]:
        if stale:
            return {
                "po_id": po_id,
                "stale_replica": True,
                "goods_receipts": [],
                "warning": "REPLICA_LAG_DETECTED_RETRY_RECOMMENDED",
            }
        grns = self.grns_by_po.get(po_id, [])
        return {
            "po_id": po_id,
            "stale_replica": False,
            "goods_receipts": grns,
        }

    def check_invoice_history(
        self,
        vendor_id: str,
        invoice_number: str,
    ) -> dict[str, Any]:
        records = self.history_by_vendor.get(vendor_id, [])
        matches = [r for r in records if r["invoice_number"] == invoice_number]
        return {
            "vendor_id": vendor_id,
            "invoice_number": invoice_number,
            "is_duplicate": len(matches) > 0,
            "matches": matches,
        }

    def place_payment_hold(
        self,
        invoice_id: str,
        vendor_id: str,
        reason_code: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        existing = self.holds_by_idempotency_key.get(idempotency_key)
        if existing is not None:
            return {**existing, "deduplicated": True}

        hold_id = f"HOLD-{5000 + len(self.holds_by_idempotency_key) + 1}"
        record = {
            "hold_id": hold_id,
            "invoice_id": invoice_id,
            "vendor_id": vendor_id,
            "reason_code": reason_code,
            "idempotency_key": idempotency_key,
            "created_at": datetime.now(UTC).isoformat(),
            "deduplicated": False,
        }
        self.holds_by_idempotency_key[idempotency_key] = record
        return record


class FlakyTogglePayload(BaseModel):
    """Admin request body for toggling ERP flaky mode at runtime."""

    enabled: bool
    timeout_rate: float | None = None
    rate_limit_429_rate: float | None = None
    stale_read_rate: float | None = None


def build_mcp_server(db: ErpDatabase, flaky: FlakyInjector) -> MCPServer:
    """Create the `MCPServer` exposing legacy ERP read/write tools."""
    mcp = MCPServer(name="legacy-erp-mcp", version="0.1.0")

    @mcp.tool()
    def lookup_vendor(vendor_id: str) -> dict[str, Any]:
        """Look up supplier master record (status, currency, payment terms, expected tax rate)."""
        return db.lookup_vendor(vendor_id)

    @mcp.tool()
    def lookup_purchase_order(po_id: str) -> dict[str, Any]:
        """Look up a Purchase Order header and contracted line items by `po_id`."""
        return db.lookup_purchase_order(po_id)

    @mcp.tool()
    def list_vendor_purchase_orders(
        vendor_id: str,
        page: int = 1,
        page_size: int = 5,
    ) -> dict[str, Any]:
        """List Purchase Orders for `vendor_id` with pagination (`page`, `page_size`)."""
        return db.list_vendor_purchase_orders(vendor_id=vendor_id, page=page, page_size=page_size)

    @mcp.tool()
    def lookup_goods_receipts(po_id: str) -> dict[str, Any]:
        """Look up warehouse Goods Receipt Notes (GRNs) and QA status for `po_id`."""
        stale = flaky.should_return_stale_read()
        return db.lookup_goods_receipts(po_id=po_id, stale=stale)

    @mcp.tool()
    def check_invoice_history(vendor_id: str, invoice_number: str) -> dict[str, Any]:
        """Check whether `invoice_number` for `vendor_id` has already been paid or posted."""
        return db.check_invoice_history(vendor_id=vendor_id, invoice_number=invoice_number)

    @mcp.tool()
    def place_payment_hold(
        invoice_id: str,
        vendor_id: str,
        reason_code: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Record an idempotent payment hold in the ERP ledger using `idempotency_key`."""
        return db.place_payment_hold(
            invoice_id=invoice_id,
            vendor_id=vendor_id,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
        )

    return mcp


def create_erp_app(
    db: ErpDatabase,
    flaky: FlakyInjector | None = None,
) -> FastAPI:
    """Build combined FastAPI REST + Streamable HTTP MCP application."""
    injector = flaky or FlakyInjector(FlakySettings(enabled=False))
    mcp_server = build_mcp_server(db=db, flaky=injector)
    mcp_http_app = mcp_server.streamable_http_app(
        streamable_http_path="/",
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with mcp_http_app.router.lifespan_context(mcp_http_app):
            yield

    app = FastAPI(title="Mock Legacy ERP + MCP Service", version="0.1.0", lifespan=lifespan)
    app.state.db = db
    app.state.flaky = injector

    @app.middleware("http")
    async def flaky_transport_middleware(request: Request, call_next: Any) -> Response:
        should_sample = request.url.path.startswith("/api/")
        if request.url.path.startswith("/mcp") and request.method == "POST":
            body_bytes = await request.body()
            if b"tools/call" in body_bytes:
                should_sample = True
        if should_sample:
            fault = injector.next_http_fault()
            if fault == "rate_limit_429":
                return JSONResponse(
                    status_code=429,
                    content={"error": "ERP_RATE_LIMITED", "retry_after_ms": 120},
                    headers={"Retry-After": "1"},
                )
            if fault == "timeout":
                await asyncio.sleep(injector.settings.timeout_sleep_seconds)
                return JSONResponse(
                    status_code=504,
                    content={"error": "ERP_GATEWAY_TIMEOUT"},
                )
        resp: Response = await call_next(request)
        return resp

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "flaky": injector.stats()}

    @app.post("/admin/flaky")
    async def configure_flaky(payload: FlakyTogglePayload) -> dict[str, Any]:
        injector.settings.enabled = payload.enabled
        if payload.timeout_rate is not None:
            injector.settings.timeout_rate = payload.timeout_rate
        if payload.rate_limit_429_rate is not None:
            injector.settings.rate_limit_429_rate = payload.rate_limit_429_rate
        if payload.stale_read_rate is not None:
            injector.settings.stale_read_rate = payload.stale_read_rate
        return injector.stats()

    @app.get("/api/vendors/{vendor_id}")
    async def rest_vendor(vendor_id: str) -> dict[str, Any]:
        res = db.lookup_vendor(vendor_id)
        if not res["found"]:
            raise HTTPException(status_code=404, detail="Vendor not found")
        return res

    @app.get("/api/pos/{po_id}")
    async def rest_po(po_id: str) -> dict[str, Any]:
        res = db.lookup_purchase_order(po_id)
        if not res["found"]:
            raise HTTPException(status_code=404, detail="PO not found")
        return res

    @app.get("/api/holds")
    async def rest_holds() -> dict[str, Any]:
        return {"holds": list(db.holds_by_idempotency_key.values())}

    app.mount("/mcp", mcp_http_app)
    return app
