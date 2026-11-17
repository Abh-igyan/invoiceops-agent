"""Core Pydantic domain contracts for invoice exception resolution, ERP data, and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DecisionType = Literal["approve", "hold", "reject", "escalate_to_human"]
EvidenceType = Literal["policy_clause", "erp_record"]
FailureModeType = Literal[
    "clean_match",
    "price_variance_within_tolerance",
    "price_variance_over_tolerance",
    "quantity_mismatch",
    "duplicate_invoice",
    "missing_po",
    "currency_mismatch",
    "tax_mismatch",
    "late_invoice_vs_terms",
    "stale_policy_version",
    "ambiguous_escalation",
    "prompt_injection",
]
FailureCategory = Literal[
    "none",
    "retrieval_miss",
    "extraction_error",
    "wrong_tool",
    "reasoning_error",
    "tool_erp_failure",
    "policy_version_error",
    "injection_success",
]


class EvidenceItem(BaseModel):
    """Auditable citation pointing to a retrieved policy clause or ERP entity."""

    type: EvidenceType
    id: str = Field(min_length=1, description="Exact clause ID (e.g. POL-02-V2#1) or ERP record ID")


class DecisionOutput(BaseModel):
    """Final structured decision contract required by Section 2 of the project brief."""

    decision: DecisionType
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class InvoiceLineItem(BaseModel):
    """Single line item on an invoice or purchase order."""

    sku: str
    description: str
    quantity: int = Field(gt=0)
    unit_price: float = Field(ge=0.0)
    line_total: float = Field(ge=0.0)


class InvoiceDocument(BaseModel):
    """Structured multimodal extraction schema for an invoice PDF/image."""

    invoice_id: str
    invoice_number: str
    vendor_id: str
    vendor_name: str
    po_id: str | None = None
    invoice_date: str
    due_date: str
    currency: str
    lines: list[InvoiceLineItem]
    subtotal: float
    tax_rate: float
    tax_amount: float
    total_amount: float
    notes: str = ""


class VendorRecord(BaseModel):
    """Vendor master record in the mock legacy ERP."""

    vendor_id: str
    name: str
    country: str
    default_currency: str
    payment_terms_days: int
    expected_tax_rate: float
    status: Literal["ACTIVE", "ON_HOLD"] = "ACTIVE"


class PurchaseOrder(BaseModel):
    """Purchase order header and line items in the mock legacy ERP."""

    po_id: str
    vendor_id: str
    issue_date: str
    currency: str
    status: Literal["OPEN", "CLOSED", "CANCELLED"] = "OPEN"
    lines: list[InvoiceLineItem]
    subtotal: float
    total_amount: float


class GoodsReceiptLine(BaseModel):
    """Received and accepted quantity for a PO line."""

    sku: str
    received_quantity: int = Field(ge=0)
    accepted_quantity: int = Field(ge=0)
    condition_note: str = "GOOD"


class GoodsReceipt(BaseModel):
    """Goods receipt note (GRN) linked to a purchase order."""

    grn_id: str
    po_id: str
    vendor_id: str
    received_date: str
    qa_status: Literal["ACCEPTED", "PARTIAL", "DAMAGED_ON_ARRIVAL"] = "ACCEPTED"
    lines: list[GoodsReceiptLine]


class HistoricalInvoice(BaseModel):
    """Previously processed invoice stored in ERP ledger for duplicate checks."""

    history_id: str
    vendor_id: str
    invoice_number: str
    po_id: str
    total_amount: float
    currency: str
    status: Literal["PAID", "POSTED"] = "PAID"


class PolicyClause(BaseModel):
    """Atomic clause inside a versioned AP policy document."""

    clause_id: str
    doc_id: str
    doc_title: str
    version: str
    effective_from: str
    status: Literal["active", "superseded"]
    topic: str
    text: str


class SyntheticCase(BaseModel):
    """Complete synthetic evaluation case pairing rendered PDF with ground truth."""

    case_id: str
    pdf_path: str
    layout_style: Literal["classic_corporate", "modern_minimal", "legacy_tabular"]
    is_noisy_scan: bool
    skew_degrees: float = 0.0
    blur_radius: float = 0.0
    failure_mode: FailureModeType
    is_prompt_injection: bool = False
    invoice: InvoiceDocument
    ground_truth_decision: DecisionType
    ground_truth_reason_codes: list[str]
    relevant_clause_ids: list[str]
    expected_erp_ids: list[str]
    superseded_trap_clause_ids: list[str] = Field(default_factory=list)


class AgentTraceSummary(BaseModel):
    """Execution telemetry and intermediate artifacts captured from an agent run."""

    output: DecisionOutput
    explanation: str = ""
    resolved_model: str
    retrieved_clause_ids: list[str] = Field(default_factory=list)
    retrieved_ranked_lists: list[list[str]] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    erp_retries: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    degraded_due_to_erp_error: bool = False


class CaseEvalRecord(BaseModel):
    """Single evaluated case row mirroring the BigQuery `eval_runs` table DDL."""

    run_id: str
    experiment_version: str
    config_version: str
    git_sha: str
    split_name: str
    rep_index: int
    case_id: str
    failure_mode: FailureModeType
    is_prompt_injection: bool
    ground_truth_decision: DecisionType
    predicted_decision: DecisionType
    predicted_reason_codes: list[str]
    cited_ids: list[str]
    retrieved_clause_ids: list[str]
    tools_called: list[str] = Field(default_factory=list)
    is_correct: bool
    weighted_error_cost: float
    citation_valid: bool
    retrieval_recall_at_k: float
    retrieval_mrr: float
    injection_resisted: bool | None = None
    failure_category: FailureCategory
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    llm_calls: int
    tool_calls: int
    erp_retries: int
    resolved_model: str
    explanation: str = ""
    timestamp_utc: str
