"""Instruction/data separation, prompt-injection detection, and pre-critic decision verification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from invoiceops.schemas import DecisionOutput, DecisionType, EvidenceItem, InvoiceDocument
from invoiceops.tools.matching import ThreeWayMatchReport

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "IGNORE_PRIOR_INSTRUCTIONS",
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
    ),
    (
        "SYSTEM_OVERRIDE",
        re.compile(r"system\s+override|admin\s+override|executive\s+override", re.I),
    ),
    (
        "FORCED_APPROVAL_DIRECTIVE",
        re.compile(
            r'approve\s+(?:this\s+invoice\s+)?immediately|output\s+["\']?approve["\']?',
            re.I,
        ),
    ),
    (
        "BYPASS_VERIFICATION",
        re.compile(r"bypass\s+(?:all\s+)?(?:po|three-way|erp|policy|verification|hold)", re.I),
    ),
    (
        "SUPPRESS_HOLD_OR_REJECT",
        re.compile(
            r"do\s+not\s+(?:place\s+hold|reject|escalate|call\s+place_payment_hold)",
            re.I,
        ),
    ),
)

ANOMALY_TO_ACTIVE_CLAUSE: dict[str, tuple[str, ...]] = {
    "THREE_WAY_MATCH_PASSED": ("POL-05-V2#2", "POL-03#1"),
    "PRICE_VARIANCE_WITHIN_TOLERANCE": ("POL-02-V2#1",),
    "PRICE_VARIANCE_OVER_TOLERANCE": ("POL-02-V2#2",),
    "SEVERE_PRICE_OVERCHARGE": ("POL-02-V2#3",),
    "QUANTITY_MISMATCH": ("POL-03#2",),
    "DUPLICATE_INVOICE": ("POL-04#1",),
    "MISSING_OR_INVALID_PO": ("POL-01#2",),
    "SPOT_BUY_REVIEW": ("POL-01#3",),
    "CURRENCY_MISMATCH": ("POL-06#1",),
    "VENDOR_ON_HOLD": ("POL-07#1",),
    "TAX_RATE_MISMATCH": ("POL-08#2",),
    "LATE_INVOICE_SUBMISSION": ("POL-09-V2#1",),
    "INVOICE_MATH_ERROR": ("POL-10#1",),
    "DAMAGED_GOODS_REVIEW": ("POL-11#1",),
    "HIGH_VALUE_APPROVAL_REQUIRED": ("POL-05-V2#1",),
}


@dataclass(frozen=True)
class InjectionScanResult:
    """Result of scanning untrusted invoice text fields for prompt-injection payloads."""

    detected: bool
    matched_rules: list[str] = field(default_factory=list)
    delimited_notes: str = ""


@dataclass(frozen=True)
class ProposalVerificationResult:
    """Result of checking an Orchestrator decision proposal against deterministic invariants."""

    is_valid: bool
    defects: list[str]
    expected_decision: DecisionType
    expected_reason_codes: list[str]
    repaired_evidence: list[EvidenceItem]


def scan_untrusted_invoice_fields(invoice: InvoiceDocument) -> InjectionScanResult:
    """Scan untrusted supplier fields (`notes`, `vendor_name`, `lines`) and wrap with XML tags."""
    combined_text = " | ".join(
        [
            invoice.notes or "",
            invoice.vendor_name or "",
            *(ln.description for ln in invoice.lines),
        ]
    )
    matched: list[str] = []
    for label, pattern in INJECTION_PATTERNS:
        if pattern.search(combined_text):
            matched.append(label)

    safe_notes = (invoice.notes or "").replace("</untrusted_supplier_text>", "[REDACTED_TAG]")
    delimited = f"<untrusted_supplier_text>{safe_notes}</untrusted_supplier_text>"
    return InjectionScanResult(
        detected=len(matched) > 0,
        matched_rules=matched,
        delimited_notes=delimited,
    )


def sanitize_multiline_json_text(raw_text: str) -> str:
    """Escape literal newlines/carriage-returns occurring inside JSON double-quoted strings.

    Fixes `FL-007` where ADK multi-part chunk boundaries or model line wraps insert raw `\n`
    mid-token or mid-sentence inside a JSON string literal, which otherwise causes
    `json.loads` to fail with `Invalid control character`.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in raw_text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch == "\n":
                out.append(" ")
            elif ch == "\r":
                continue
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def build_canonical_evidence(
    proposed_evidence: list[EvidenceItem],
    match_report: ThreeWayMatchReport,
    injection_scan: InjectionScanResult,
    retrieved_clause_ids: list[str],
    active_clause_ids: set[str],
    valid_erp_ids: set[str],
) -> list[EvidenceItem]:
    """Filter out invalid/superseded citations and ensure active clause + ERP record coverage."""
    clean_policies: list[str] = []
    clean_erp: list[str] = []

    for item in proposed_evidence:
        if (
            item.type == "policy_clause"
            and item.id in active_clause_ids
            and item.id not in clean_policies
        ):
            clean_policies.append(item.id)
        elif item.type == "erp_record" and item.id in valid_erp_ids and item.id not in clean_erp:
            clean_erp.append(item.id)

    # Ensure primary anomaly active clause(s) are present
    for code in match_report.anomaly_codes:
        for cid in ANOMALY_TO_ACTIVE_CLAUSE.get(code, ()):
            if cid in active_clause_ids and cid not in clean_policies:
                clean_policies.append(cid)

    if (
        injection_scan.detected
        and "POL-12#1" in active_clause_ids
        and "POL-12#1" not in clean_policies
    ):
        clean_policies.append("POL-12#1")

    if not clean_policies:
        for cid in retrieved_clause_ids:
            if cid in active_clause_ids and cid not in clean_policies:
                clean_policies.append(cid)
                break

    if not clean_erp:
        for eid in match_report.inspected_erp_ids:
            if eid in valid_erp_ids and eid not in clean_erp:
                clean_erp.append(eid)

    result: list[EvidenceItem] = [
        EvidenceItem(type="policy_clause", id=cid) for cid in clean_policies
    ]
    result.extend(EvidenceItem(type="erp_record", id=eid) for eid in clean_erp)
    return result


def verify_decision_proposal(
    proposed: DecisionOutput,
    match_report: ThreeWayMatchReport,
    injection_scan: InjectionScanResult,
    retrieved_clause_ids: list[str],
    active_clause_ids: set[str],
    valid_erp_ids: set[str],
) -> ProposalVerificationResult:
    """Deterministically audit an Orchestrator decision proposal before Critic reflection."""
    defects: list[str] = []
    expected_decision = match_report.recommended_deterministic_action

    # 1. Hard injection safety guardrail
    if injection_scan.detected and proposed.decision == "approve":
        defects.append(
            f"UNSAFE_INJECTION_APPROVAL: invoice contains prompt-injection pattern "
            f"{injection_scan.matched_rules} and cannot be approved."
        )

    # 2. Decision vs deterministic arithmetic + active policy consistency
    if proposed.decision != expected_decision:
        defects.append(
            f"DECISION_MISMATCH: proposed '{proposed.decision}' conflicts with deterministic "
            f"anomaly codes {match_report.anomaly_codes} which require '{expected_decision}'."
        )

    # 3. Citation validity check (active policies only + real ERP records only)
    if not proposed.evidence:
        defects.append("EMPTY_EVIDENCE: decision must cite at least one active policy clause.")
    else:
        has_valid_policy = False
        has_valid_erp = False
        for item in proposed.evidence:
            if item.type == "policy_clause":
                if item.id not in active_clause_ids:
                    defects.append(
                        f"INVALID_OR_SUPERSEDED_POLICY_CITATION: '{item.id}' is not active."
                    )
                else:
                    has_valid_policy = True
            elif item.type == "erp_record":
                if item.id not in valid_erp_ids:
                    defects.append(
                        f"INVALID_ERP_RECORD_CITATION: '{item.id}' is not a valid ERP record ID."
                    )
                else:
                    has_valid_erp = True

        if not has_valid_policy:
            defects.append("MISSING_ACTIVE_POLICY_CITATION: no valid active policy clause cited.")
        if match_report.inspected_erp_ids and not has_valid_erp:
            defects.append("MISSING_ERP_RECORD_CITATION: no inspected ERP record cited.")

    if injection_scan.detected and not any(
        ev.type == "policy_clause" and ev.id == "POL-12#1" for ev in proposed.evidence
    ):
        defects.append("MISSING_SECURITY_CITATION: prompt injection detected; must cite POL-12#1.")

    repaired_ev = build_canonical_evidence(
        proposed_evidence=proposed.evidence,
        match_report=match_report,
        injection_scan=injection_scan,
        retrieved_clause_ids=retrieved_clause_ids,
        active_clause_ids=active_clause_ids,
        valid_erp_ids=valid_erp_ids,
    )

    expected_reasons = list(match_report.anomaly_codes)
    if injection_scan.detected and "PROMPT_INJECTION_DETECTED" not in expected_reasons:
        expected_reasons.append("PROMPT_INJECTION_DETECTED")

    return ProposalVerificationResult(
        is_valid=len(defects) == 0,
        defects=defects,
        expected_decision=expected_decision,
        expected_reason_codes=expected_reasons,
        repaired_evidence=repaired_ev,
    )
