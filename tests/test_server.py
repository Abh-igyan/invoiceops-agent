"""Unit tests for the FastAPI production server (Section 8 / Phase P4)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def _mock_svc(monkeypatch: pytest.MonkeyPatch):
    """Patch lifespan to avoid loading full agent pipeline in tests."""
    from contextlib import asynccontextmanager

    from invoiceops.config import load_config

    @asynccontextmanager
    async def mock_lifespan(app):
        class MockState:
            config = load_config()

        app.state.svc = MockState()
        yield

    import invoiceops.server as srv_mod

    monkeypatch.setattr(srv_mod, "lifespan", mock_lifespan)


@pytest.fixture()
def client(_mock_svc) -> Iterator[TestClient]:
    """Create a test client with mocked lifespan."""
    # Re-import to pick up the patched lifespan
    from importlib import reload

    import invoiceops.server as srv_mod

    reload(srv_mod)
    with TestClient(srv_mod.app) as c:
        yield c


class TestHealthEndpoint:
    """Health check endpoint returns expected structure."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "0.1.0"
        assert "config_version" in body

    def test_health_has_request_id(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

    def test_custom_request_id_echoed(self, client: TestClient) -> None:
        resp = client.get("/health", headers={"X-Request-ID": "test-123"})
        assert resp.headers["X-Request-ID"] == "test-123"


class TestEvaluateEndpointAuthentication:
    """Security tests for the /v1/invoices/evaluate endpoint."""

    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/invoices/evaluate",
            data={"case_id": "CASE-001", "vendor_id": "VND-001"},
        )
        assert resp.status_code == 401
        assert "Missing X-API-Key" in resp.json()["detail"]

    def test_invalid_api_key_returns_403(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/invoices/evaluate",
            headers={"X-API-Key": "wrong-key"},
            data={"case_id": "CASE-001", "vendor_id": "VND-001"},
        )
        assert resp.status_code == 403
        assert "Invalid X-API-Key" in resp.json()["detail"]


class TestEvaluateEndpointValidation:
    """Input validation for the /v1/invoices/evaluate endpoint."""

    def test_missing_pdf_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/invoices/evaluate",
            headers={"X-API-Key": "invoiceops-demo-2026"},
            data={"case_id": "CASE-001", "vendor_id": "VND-001"},
        )
        assert resp.status_code == 422

    def test_missing_case_id_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/invoices/evaluate",
            headers={"X-API-Key": "invoiceops-demo-2026"},
            files={"pdf": ("test.pdf", b"%PDF-1.4 test", "application/pdf")},
            data={"vendor_id": "VND-001"},
        )
        assert resp.status_code == 422
