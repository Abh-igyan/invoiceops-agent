"""Deterministic three-way match, arithmetic, tax, timeliness, and escalation checks."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from invoiceops.config import MatchingAndHitlConfig
from invoiceops.schemas import DecisionType, InvoiceDocument


class LineMatchFinding(BaseModel):
    """Per-SKU deterministic comparison between invoice, PO, and accepted GRN quantities."""

    sku: str
    invoiced_qty: int
    po_qty: int | None = None
    grn_accepted_qty: int | None = None
    invoiced_unit_price: float
    po_unit_price: float | None = None
    unit_price_variance_pct: float | None = None
    line_math_valid: bool = True
    sku_found_on_po: bool = True
    quantity_exceeds_grn: bool = False
    price_within_tolerance: bool = True
    price_over_tolerance_hold: bool = False
    price_severe_overcharge_reject: bool = False


class ThreeWayMatchReport(BaseModel):
    """Complete deterministic verification report produced by Python code (never LLM prose)."""

    internal_math_valid: bool
    math_discrepancy_usd: float = 0.0
    vendor_found: bool
    vendor_active: bool
    is_duplicate_invoice: bool
    duplicate_history_ids: list[str] = Field(default_factory=list)
    po_referenced: bool
    po_found_and_open: bool
    below_spot_buy_no_po_limit: bool = False
    currency_matches_po: bool = True
    tax_rate_matches_vendor: bool = True
    tax_rate_diff_abs: float = 0.0
    grn_found: bool = False
    grn_damaged_or_qa_hold: bool = False
    days_since_grn_received: int | None = None
    late_submission_exceeded: bool = False
    max_unit_price_variance_pct: float = 0.0
    exceeds_human_approval_amount_threshold: bool = False
    line_findings: list[LineMatchFinding] = Field(default_factory=list)
    anomaly_codes: list[str] = Field(default_factory=list)
    inspected_erp_ids: list[str] = Field(default_factory=list)
    recommended_deterministic_action: DecisionType = "approve"


def verify_invoice_internal_math(
    invoice: InvoiceDocument,
    rounding_tolerance_usd: float = 0.05,
) -> tuple[bool, float]:
    """Verify `qty * unit_price == line_total` and `subtotal + tax_amount == total_amount`."""
    max_diff = 0.0
    sum_lines = 0.0
    for line in invoice.lines:
        expected_line = round(line.quantity * line.unit_price, 2)
        diff = abs(expected_line - line.line_total)
        max_diff = max(max_diff, diff)
        sum_lines += line.line_total

    max_diff = max(max_diff, abs(round(sum_lines, 2) - invoice.subtotal))
    expected_tax = round(invoice.subtotal * invoice.tax_rate, 2)
    max_diff = max(max_diff, abs(expected_tax - invoice.tax_amount))
    expected_total = round(invoice.subtotal + invoice.tax_amount, 2)
    max_diff = max(max_diff, abs(expected_total - invoice.total_amount))

    return max_diff <= rounding_tolerance_usd, round(max_diff, 4)


def compute_three_way_match(
    invoice: InvoiceDocument,
    vendor_resp: dict[str, Any],
    po_resp: dict[str, Any] | None,
    grn_resp: dict[str, Any] | None,
    dup_resp: dict[str, Any],
    rules: MatchingAndHitlConfig,
) -> ThreeWayMatchReport:
    """Execute deterministic three-way match rules across invoice, PO, GRN, and vendor master."""
    anomalies: list[str] = []
    erp_ids: list[str] = []

    math_ok, math_diff = verify_invoice_internal_math(invoice)
    if not math_ok:
        anomalies.append("INVOICE_MATH_ERROR")

    vendor_found = bool(vendor_resp.get("found"))
    vendor_obj = vendor_resp.get("vendor") or {}
    vendor_active = vendor_obj.get("status") == "ACTIVE" if vendor_found else False
    if vendor_found and vendor_obj.get("vendor_id"):
        erp_ids.append(str(vendor_obj["vendor_id"]))
    if not vendor_found:
        anomalies.append("VENDOR_NOT_FOUND")
    elif not vendor_active:
        anomalies.append("VENDOR_ON_HOLD")

    is_dup = bool(dup_resp.get("is_duplicate"))
    dup_ids = [str(m["history_id"]) for m in (dup_resp.get("matches") or []) if m.get("history_id")]
    erp_ids.extend(dup_ids)
    if is_dup:
        anomalies.append("DUPLICATE_INVOICE")

    tax_diff = 0.0
    tax_ok = True
    if vendor_found:
        expected_tax = float(vendor_obj.get("expected_tax_rate", 0.0))
        tax_diff = round(abs(invoice.tax_rate - expected_tax), 4)
        tax_ok = tax_diff <= rules.tax_rate_tolerance_abs
        if not tax_ok:
            anomalies.append("TAX_RATE_MISMATCH")

    po_referenced = bool(invoice.po_id and invoice.po_id.strip())
    po_found = bool(po_resp and po_resp.get("found"))
    po_obj = (po_resp or {}).get("purchase_order") or {}
    po_open = po_found and po_obj.get("status") == "OPEN"
    if po_found and po_obj.get("po_id"):
        erp_ids.append(str(po_obj["po_id"]))

    below_spot_limit = (not po_referenced) and (
        invoice.total_amount <= rules.spot_buy_no_po_limit_usd
    )
    if not po_referenced or not po_open:
        if below_spot_limit:
            anomalies.append("SPOT_BUY_REVIEW")
        else:
            anomalies.append("MISSING_OR_INVALID_PO")

    currency_ok = True
    if po_open:
        currency_ok = invoice.currency.upper() == str(po_obj.get("currency", "")).upper()
        if not currency_ok:
            anomalies.append("CURRENCY_MISMATCH")

    grn_list = (grn_resp or {}).get("goods_receipts") or []
    grn_found = len(grn_list) > 0
    grn_damaged = False
    accepted_by_sku: dict[str, int] = {}
    earliest_recv_date: date | None = None

    for grn in grn_list:
        if grn.get("grn_id"):
            erp_ids.append(str(grn["grn_id"]))
        if grn.get("qa_status") == "DAMAGED_ON_ARRIVAL":
            grn_damaged = True
        recv_str = grn.get("received_date")
        if recv_str:
            dt = date.fromisoformat(str(recv_str))
            earliest_recv_date = dt if earliest_recv_date is None else min(earliest_recv_date, dt)
        for g_line in grn.get("lines") or []:
            sku = str(g_line["sku"])
            accepted_by_sku[sku] = accepted_by_sku.get(sku, 0) + int(
                g_line.get("accepted_quantity", 0)
            )
            if "DAMAGE" in str(g_line.get("condition_note", "")).upper():
                grn_damaged = True

    if grn_damaged:
        anomalies.append("DAMAGED_GOODS_REVIEW")

    days_since_grn: int | None = None
    late_exceeded = False
    if earliest_recv_date is not None:
        try:
            inv_dt = date.fromisoformat(invoice.invoice_date)
            days_since_grn = (inv_dt - earliest_recv_date).days
            if days_since_grn > rules.late_submission_max_days:
                late_exceeded = True
                anomalies.append("LATE_INVOICE_SUBMISSION")
        except ValueError:
            pass

    line_findings: list[LineMatchFinding] = []
    max_price_var_pct = 0.0
    po_lines_by_sku = {str(item["sku"]): item for item in (po_obj.get("lines") or [])}

    if po_open:
        if not grn_found:
            anomalies.append("QUANTITY_MISMATCH")

        for inv_line in invoice.lines:
            line_math_ok = (
                abs(round(inv_line.quantity * inv_line.unit_price, 2) - inv_line.line_total) <= 0.05
            )
            po_line = po_lines_by_sku.get(inv_line.sku)
            if po_line is None:
                anomalies.append("UNORDERED_LINE_SKU")
                line_findings.append(
                    LineMatchFinding(
                        sku=inv_line.sku,
                        invoiced_qty=inv_line.quantity,
                        invoiced_unit_price=inv_line.unit_price,
                        line_math_valid=line_math_ok,
                        sku_found_on_po=False,
                    )
                )
                continue

            po_qty = int(po_line["quantity"])
            po_unit = float(po_line["unit_price"])
            acc_qty = accepted_by_sku.get(inv_line.sku, 0)
            qty_over = inv_line.quantity > acc_qty
            if qty_over and "QUANTITY_MISMATCH" not in anomalies:
                anomalies.append("QUANTITY_MISMATCH")

            var_pct = ((inv_line.unit_price - po_unit) / po_unit) if po_unit > 0 else 0.0
            max_price_var_pct = max(max_price_var_pct, round(var_pct, 6))

            within_tol = var_pct <= rules.price_variance_tolerance_pct + 1e-9
            over_hold = (
                rules.price_variance_tolerance_pct < var_pct <= rules.price_reject_threshold_pct
            )
            severe_reject = var_pct > rules.price_reject_threshold_pct

            if severe_reject and "SEVERE_PRICE_OVERCHARGE" not in anomalies:
                anomalies.append("SEVERE_PRICE_OVERCHARGE")
            elif over_hold and "PRICE_VARIANCE_OVER_TOLERANCE" not in anomalies:
                anomalies.append("PRICE_VARIANCE_OVER_TOLERANCE")
            elif (
                var_pct > 0.0001
                and within_tol
                and "PRICE_VARIANCE_WITHIN_TOLERANCE" not in anomalies
            ):
                anomalies.append("PRICE_VARIANCE_WITHIN_TOLERANCE")

            line_findings.append(
                LineMatchFinding(
                    sku=inv_line.sku,
                    invoiced_qty=inv_line.quantity,
                    po_qty=po_qty,
                    grn_accepted_qty=acc_qty,
                    invoiced_unit_price=inv_line.unit_price,
                    po_unit_price=po_unit,
                    unit_price_variance_pct=round(var_pct, 6),
                    line_math_valid=line_math_ok,
                    sku_found_on_po=True,
                    quantity_exceeds_grn=qty_over,
                    price_within_tolerance=within_tol,
                    price_over_tolerance_hold=over_hold,
                    price_severe_overcharge_reject=severe_reject,
                )
            )

    high_value = invoice.total_amount > rules.human_approval_amount_threshold_usd
    if high_value and not any(
        c in anomalies
        for c in (
            "DUPLICATE_INVOICE",
            "MISSING_OR_INVALID_PO",
            "CURRENCY_MISMATCH",
            "SEVERE_PRICE_OVERCHARGE",
            "INVOICE_MATH_ERROR",
            "QUANTITY_MISMATCH",
            "PRICE_VARIANCE_OVER_TOLERANCE",
            "TAX_RATE_MISMATCH",
            "LATE_INVOICE_SUBMISSION",
            "VENDOR_ON_HOLD",
        )
    ):
        anomalies.append("HIGH_VALUE_APPROVAL_REQUIRED")

    rec_action = _derive_recommended_action(anomalies)
    dedup_erp: list[str] = []
    for eid in erp_ids:
        if eid not in dedup_erp:
            dedup_erp.append(eid)

    return ThreeWayMatchReport(
        internal_math_valid=math_ok,
        math_discrepancy_usd=math_diff,
        vendor_found=vendor_found,
        vendor_active=vendor_active,
        is_duplicate_invoice=is_dup,
        duplicate_history_ids=dup_ids,
        po_referenced=po_referenced,
        po_found_and_open=po_open,
        below_spot_buy_no_po_limit=below_spot_limit,
        currency_matches_po=currency_ok,
        tax_rate_matches_vendor=tax_ok,
        tax_rate_diff_abs=tax_diff,
        grn_found=grn_found,
        grn_damaged_or_qa_hold=grn_damaged,
        days_since_grn_received=days_since_grn,
        late_submission_exceeded=late_exceeded,
        max_unit_price_variance_pct=round(max_price_var_pct, 4),
        exceeds_human_approval_amount_threshold=high_value,
        line_findings=line_findings,
        anomaly_codes=anomalies or ["THREE_WAY_MATCH_PASSED"],
        inspected_erp_ids=dedup_erp,
        recommended_deterministic_action=rec_action,
    )


def _derive_recommended_action(anomalies: list[str]) -> DecisionType:
    reject_codes = {
        "DUPLICATE_INVOICE",
        "MISSING_OR_INVALID_PO",
        "CURRENCY_MISMATCH",
        "SEVERE_PRICE_OVERCHARGE",
        "INVOICE_MATH_ERROR",
        "UNORDERED_LINE_SKU",
        "VENDOR_NOT_FOUND",
    }
    hold_codes = {
        "PRICE_VARIANCE_OVER_TOLERANCE",
        "QUANTITY_MISMATCH",
        "TAX_RATE_MISMATCH",
        "LATE_INVOICE_SUBMISSION",
        "VENDOR_ON_HOLD",
    }
    escalate_codes = {
        "DAMAGED_GOODS_REVIEW",
        "HIGH_VALUE_APPROVAL_REQUIRED",
        "SPOT_BUY_REVIEW",
    }
    if any(c in reject_codes for c in anomalies):
        return "reject"
    if any(c in hold_codes for c in anomalies):
        return "hold"
    if any(c in escalate_codes for c in anomalies):
        return "escalate_to_human"
    return "approve"
