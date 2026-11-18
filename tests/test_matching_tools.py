"""Unit tests for deterministic three-way match, tolerance, tax, and timeliness tools."""

from __future__ import annotations

import pytest

from invoiceops.config import load_config
from invoiceops.schemas import InvoiceDocument, InvoiceLineItem
from invoiceops.tools.matching import compute_three_way_match, verify_invoice_internal_math


def _base_invoice(
    unit_price: float = 400.0,
    qty: int = 2,
    tax_rate: float = 0.08,
    currency: str = "USD",
    po_id: str | None = "PO-1001",
    invoice_date: str = "2026-06-12",
) -> InvoiceDocument:
    subtotal = round(qty * unit_price, 2)
    tax_amt = round(subtotal * tax_rate, 2)
    return InvoiceDocument(
        invoice_id="INV-4001",
        invoice_number="BILL-4001",
        vendor_id="VEN-001",
        vendor_name="Apex Systems",
        po_id=po_id,
        invoice_date=invoice_date,
        due_date="2026-07-12",
        currency=currency,
        lines=[
            InvoiceLineItem(
                sku="SKU-101",
                description="PDU",
                quantity=qty,
                unit_price=unit_price,
                line_total=subtotal,
            )
        ],
        subtotal=subtotal,
        tax_rate=tax_rate,
        tax_amount=tax_amt,
        total_amount=round(subtotal + tax_amt, 2),
    )


def _base_erp_fixtures() -> tuple[dict, dict, dict, dict]:
    vendor = {
        "found": True,
        "vendor": {
            "vendor_id": "VEN-001",
            "status": "ACTIVE",
            "expected_tax_rate": 0.08,
            "default_currency": "USD",
        },
    }
    po = {
        "found": True,
        "purchase_order": {
            "po_id": "PO-1001",
            "status": "OPEN",
            "currency": "USD",
            "lines": [{"sku": "SKU-101", "quantity": 2, "unit_price": 400.0, "line_total": 800.0}],
        },
    }
    grn = {
        "goods_receipts": [
            {
                "grn_id": "GRN-2001",
                "received_date": "2026-06-05",
                "qa_status": "ACCEPTED",
                "lines": [
                    {
                        "sku": "SKU-101",
                        "received_quantity": 2,
                        "accepted_quantity": 2,
                        "condition_note": "GOOD",
                    }
                ],
            }
        ]
    }
    dup = {"is_duplicate": False, "matches": []}
    return vendor, po, grn, dup


def test_clean_match_and_within_tolerance_price_variance() -> None:
    """Verify exact match and +1.5% unit price variance within 2% tolerance."""
    rules = load_config().matching_and_hitl
    vendor, po, grn, dup = _base_erp_fixtures()

    clean_rep = compute_three_way_match(_base_invoice(), vendor, po, grn, dup, rules)
    assert clean_rep.recommended_deterministic_action == "approve"
    assert clean_rep.anomaly_codes == ["THREE_WAY_MATCH_PASSED"]

    # +1.5% ($406 vs $400) -> within 2.0% tolerance
    tol_rep = compute_three_way_match(_base_invoice(unit_price=406.0), vendor, po, grn, dup, rules)
    assert tol_rep.max_unit_price_variance_pct == pytest.approx(0.015)
    assert tol_rep.recommended_deterministic_action == "approve"
    assert "PRICE_VARIANCE_WITHIN_TOLERANCE" in tol_rep.anomaly_codes


def test_over_tolerance_hold_and_severe_reject() -> None:
    """Verify +5% variance triggers HOLD while +15% triggers REJECT."""
    rules = load_config().matching_and_hitl
    vendor, po, grn, dup = _base_erp_fixtures()

    hold_rep = compute_three_way_match(_base_invoice(unit_price=420.0), vendor, po, grn, dup, rules)
    assert hold_rep.recommended_deterministic_action == "hold"
    assert "PRICE_VARIANCE_OVER_TOLERANCE" in hold_rep.anomaly_codes

    rej_rep = compute_three_way_match(_base_invoice(unit_price=460.0), vendor, po, grn, dup, rules)
    assert rej_rep.recommended_deterministic_action == "reject"
    assert "SEVERE_PRICE_OVERCHARGE" in rej_rep.anomaly_codes


def test_quantity_tax_late_duplicate_and_damaged_escalation() -> None:
    """Verify deterministic detection of qty, tax, late submission, duplicate, and QA damage."""
    rules = load_config().matching_and_hitl
    vendor, po, grn, dup = _base_erp_fixtures()

    qty_rep = compute_three_way_match(_base_invoice(qty=4), vendor, po, grn, dup, rules)
    assert "QUANTITY_MISMATCH" in qty_rep.anomaly_codes
    assert qty_rep.recommended_deterministic_action == "hold"

    tax_rep = compute_three_way_match(_base_invoice(tax_rate=0.12), vendor, po, grn, dup, rules)
    assert "TAX_RATE_MISMATCH" in tax_rep.anomaly_codes
    assert tax_rep.recommended_deterministic_action == "hold"

    late_rep = compute_three_way_match(
        _base_invoice(invoice_date="2026-09-15"), vendor, po, grn, dup, rules
    )
    assert late_rep.days_since_grn_received == 102
    assert late_rep.late_submission_exceeded is True
    assert late_rep.recommended_deterministic_action == "hold"

    dup_hit = {"is_duplicate": True, "matches": [{"history_id": "HIST-3001"}]}
    dup_rep = compute_three_way_match(_base_invoice(), vendor, po, grn, dup_hit, rules)
    assert dup_rep.recommended_deterministic_action == "reject"
    assert "HIST-3001" in dup_rep.inspected_erp_ids

    damaged_grn = {
        "goods_receipts": [
            {
                "grn_id": "GRN-2001",
                "received_date": "2026-06-05",
                "qa_status": "DAMAGED_ON_ARRIVAL",
                "lines": [{"sku": "SKU-101", "received_quantity": 2, "accepted_quantity": 2}],
            }
        ]
    }
    dmg_rep = compute_three_way_match(_base_invoice(), vendor, po, damaged_grn, dup, rules)
    assert dmg_rep.grn_damaged_or_qa_hold is True
    assert dmg_rep.recommended_deterministic_action == "escalate_to_human"


def test_internal_invoice_math_validation() -> None:
    """Verify tampered subtotal/total arithmetic fails deterministic validation."""
    inv = _base_invoice()
    bad_inv = inv.model_copy(update={"total_amount": inv.total_amount + 25.0})
    ok, diff = verify_invoice_internal_math(bad_inv)
    assert ok is False and diff == pytest.approx(25.0)
