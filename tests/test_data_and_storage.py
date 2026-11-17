"""Unit tests for policy corpus, PDF rendering, naive chunking, and SQLite vector search."""

from __future__ import annotations

from pathlib import Path

from invoiceops.data_gen.pdf_renderer import render_invoice_pdf
from invoiceops.data_gen.policies import build_policy_corpus, write_policy_files
from invoiceops.schemas import CaseEvalRecord, InvoiceDocument, InvoiceLineItem
from invoiceops.storage.local_store import LocalSqliteStore, build_naive_chunks


def test_policy_corpus_has_15_docs_and_superseded_versions() -> None:
    """Verify policy corpus contains 15 documents including active and superseded versions."""
    clauses, docs_md = build_policy_corpus()
    assert len(docs_md) == 15
    statuses = {c.status for c in clauses}
    assert statuses == {"active", "superseded"}
    assert any(c.clause_id == "POL-02-V1#1" and c.status == "superseded" for c in clauses)
    assert any(c.clause_id == "POL-02-V2#1" and c.status == "active" for c in clauses)


def test_digital_and_noisy_scan_pdf_rendering(tmp_path: Path) -> None:
    """Verify both crisp digital PDFs and skewed/blurred scan PDFs render cleanly."""
    invoice = InvoiceDocument(
        invoice_id="INV-9901",
        invoice_number="BILL-9901",
        vendor_id="VEN-001",
        vendor_name="Apex Systems Ltd",
        po_id="PO-1001",
        invoice_date="2026-06-15",
        due_date="2026-07-15",
        currency="USD",
        lines=[
            InvoiceLineItem(
                sku="SKU-101",
                description="Rack PDU",
                quantity=2,
                unit_price=400.0,
                line_total=800.0,
            )
        ],
        subtotal=800.0,
        tax_rate=0.08,
        tax_amount=64.0,
        total_amount=864.0,
        notes="Test remittance note.",
    )
    digital_pdf = tmp_path / "digital.pdf"
    noisy_pdf = tmp_path / "noisy.pdf"

    render_invoice_pdf(invoice, digital_pdf, layout_style="classic_corporate", is_noisy_scan=False)
    render_invoice_pdf(
        invoice,
        noisy_pdf,
        layout_style="legacy_tabular",
        is_noisy_scan=True,
        skew_degrees=-2.2,
        blur_radius=0.6,
        rng_seed=42,
    )

    assert digital_pdf.exists() and digital_pdf.read_bytes().startswith(b"%PDF")
    assert noisy_pdf.exists() and noisy_pdf.read_bytes().startswith(b"%PDF")
    assert noisy_pdf.stat().st_size > 2_000


def test_local_sqlite_naive_vector_search_and_caching(tmp_path: Path) -> None:
    """Verify SHA-256 embedding cache reuse, cosine search, and `eval_runs` persistence."""
    pol_dir = tmp_path / "policies"
    write_policy_files(pol_dir)
    chunks = build_naive_chunks(pol_dir, chunk_chars=380, overlap_chars=50)
    assert len(chunks) >= 15

    embed_calls = 0

    def fake_embed(texts: list[str]) -> list[list[float]]:
        nonlocal embed_calls
        embed_calls += len(texts)
        vecs: list[list[float]] = []
        for t in texts:
            v = [
                1.0 if "duplicate" in t.lower() else 0.1,
                1.0 if "variance" in t.lower() else 0.1,
                float((len(t) % 10) + 1),
            ]
            vecs.append(v)
        return vecs

    db_path = tmp_path / "store.sqlite3"
    jsonl_path = tmp_path / "eval_runs.jsonl"
    store = LocalSqliteStore(db_path=db_path, jsonl_path=jsonl_path)

    first_embedded = store.index_naive_chunks(chunks, embed_fn=fake_embed)
    assert first_embedded == len(chunks)
    calls_after_first = embed_calls

    # Re-indexing identical text must trigger zero new embedding calls
    second_embedded = store.index_naive_chunks(chunks, embed_fn=fake_embed)
    assert second_embedded == 0
    assert embed_calls == calls_after_first

    hits = store.search_naive([10.0, 0.0, 0.1], top_k=3)
    assert len(hits) == 3
    assert any("POL-04#1" in h.clause_ids for h in hits)

    rec = CaseEvalRecord(
        run_id="r1",
        experiment_version="v0",
        config_version="p1.0.0",
        git_sha="abc1234",
        split_name="golden_12",
        rep_index=1,
        case_id="INV-4001",
        failure_mode="clean_match",
        is_prompt_injection=False,
        ground_truth_decision="approve",
        predicted_decision="approve",
        predicted_reason_codes=["THREE_WAY_MATCH_PASSED"],
        cited_ids=["POL-05-V2#2", "PO-1001"],
        retrieved_clause_ids=["POL-05-V2#2"],
        is_correct=True,
        weighted_error_cost=0.0,
        citation_valid=True,
        retrieval_recall_at_k=1.0,
        retrieval_mrr=1.0,
        injection_resisted=None,
        failure_category="none",
        latency_ms=1250.0,
        input_tokens=900,
        output_tokens=80,
        estimated_cost_usd=0.00069,
        llm_calls=2,
        tool_calls=4,
        erp_retries=0,
        resolved_model="gemini-3-flash-preview",
        explanation="All three-way match checks passed.",
        timestamp_utc="2026-11-17T13:30:00Z",
    )
    store.record_eval_case(rec)
    assert jsonl_path.exists()
    assert "INV-4001" in jsonl_path.read_text(encoding="utf-8")
