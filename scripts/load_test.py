"""Simple async concurrency load test for the InvoiceOps Cloud Run service (Section 8)."""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    pdf_path: Path,
    case_id: str,
    vendor_id: str,
    results: list[dict],
) -> None:
    """Send a single invoice evaluation request and record latency / status."""
    t0 = time.perf_counter()
    api_key = os.getenv("INVOICEOPS_API_KEY", "invoiceops-demo-2026")
    try:
        with pdf_path.open("rb") as f:
            resp = await client.post(
                url,
                headers={"X-API-Key": api_key},
                files={"pdf": (pdf_path.name, f, "application/pdf")},
                data={"case_id": case_id, "vendor_id": vendor_id},
                timeout=180.0,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        payload = {}
        is_ok = 200 <= resp.status_code < 300
        decision_val = None
        reason_codes = []
        if is_ok:
            try:
                payload = resp.json()
                dec = payload.get("decision", {})
                decision_val = dec.get("decision")
                reason_codes = dec.get("reason_codes", [])
                if "ERP_OR_RETRIEVAL_UNAVAILABLE" in reason_codes:
                    is_ok = False
            except Exception:
                pass
        results.append(
            {
                "case_id": case_id,
                "vendor_id": vendor_id,
                "status": resp.status_code,
                "latency_ms": elapsed_ms,
                "ok": is_ok,
                "decision": decision_val,
                "reason_codes": reason_codes,
            }
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        results.append(
            {
                "case_id": case_id,
                "vendor_id": vendor_id,
                "status": 0,
                "latency_ms": elapsed_ms,
                "ok": False,
                "error": str(exc),
            }
        )


async def run_load_test(
    base_url: str,
    concurrency: int = 5,
    total_requests: int = 20,
) -> dict:
    """Execute a simple concurrency load test against the /v1/invoices/evaluate endpoint."""
    import json

    url = f"{base_url.rstrip('/')}/v1/invoices/evaluate"

    # Find sample cases from manifest or data/invoices
    manifest_path = REPO_ROOT / "data" / "generated" / "cases_manifest.json"
    tasks_list: list[tuple[Path, str, str]] = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        for c in manifest[:total_requests]:
            pdf = REPO_ROOT / c.get("pdf_path", "")
            cid = c.get("case_id", "")
            vid = c.get("invoice", {}).get("vendor_id", "VEN-001")
            if pdf.exists():
                tasks_list.append((pdf, cid, vid))

    if not tasks_list:
        invoices_dir = REPO_ROOT / "data" / "invoices"
        pdf_files = sorted(invoices_dir.glob("*.pdf"))[:total_requests]
        if not pdf_files:
            print("ERROR: No PDF files found in data/invoices/. Run tasks.py data.")
            sys.exit(1)
        for i in range(total_requests):
            pdf = pdf_files[i % len(pdf_files)]
            case_id = f"load-test-{i:04d}"
            vendor_id = f"VEN-{(i % 40) + 1:03d}"
            tasks_list.append((pdf, case_id, vendor_id))

    results: list[dict] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(pdf: Path, cid: str, vid: str) -> None:
        async with semaphore:
            await send_request(client, url, pdf, cid, vid, results)

    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        coros = [bounded_request(pdf, cid, vid) for pdf, cid, vid in tasks_list]
        await asyncio.gather(*coros)
    total_elapsed_s = time.perf_counter() - t0

    # Compute statistics
    latencies = [r["latency_ms"] for r in results]
    ok_count = sum(1 for r in results if r["ok"])
    error_count = len(results) - ok_count

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
    p95_idx = int(len(sorted_lat) * 0.95)
    p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0.0

    report = {
        "total_requests": len(results),
        "concurrency": concurrency,
        "ok_count": ok_count,
        "error_count": error_count,
        "error_rate_pct": round(error_count / max(len(results), 1) * 100, 2),
        "total_elapsed_seconds": round(total_elapsed_s, 2),
        "throughput_rps": round(len(results) / max(total_elapsed_s, 0.001), 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
    }

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cloud_run_load_test.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"report": report, "raw_results": results}, f, indent=2)

    return report


def main() -> int:
    """CLI entry point for the load test script."""
    parser = argparse.ArgumentParser(description="InvoiceOps Cloud Run load test")
    parser.add_argument(
        "--url",
        default=os.getenv("INVOICEOPS_URL", "http://localhost:8080"),
        help="Base URL of the InvoiceOps service (default: $INVOICEOPS_URL or localhost:8080)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Total number of requests to send (default: 20)",
    )
    args = parser.parse_args()

    print(f"Load test: {args.url} | concurrency={args.concurrency} | requests={args.requests}")
    report = asyncio.run(
        run_load_test(
            base_url=args.url,
            concurrency=args.concurrency,
            total_requests=args.requests,
        )
    )

    print("\n--- Load Test Report ---")
    for key, value in report.items():
        print(f"  {key}: {value}")

    # Fail if error rate > 20%
    if report["error_rate_pct"] > 20.0:
        print(f"\nFAIL: Error rate {report['error_rate_pct']}% exceeds 20% threshold.")
        return 1

    print("\nPASS: Load test within acceptable thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
