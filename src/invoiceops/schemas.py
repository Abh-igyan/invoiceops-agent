"""Core Pydantic domain contracts for invoice exception resolution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DecisionType = Literal["approve", "hold", "reject", "escalate_to_human"]
EvidenceType = Literal["policy_clause", "erp_record"]


class EvidenceItem(BaseModel):
    """Auditable citation pointing to a retrieved policy clause or ERP entity."""

    type: EvidenceType
    id: str = Field(min_length=1, description="Exact clause ID (e.g. POL-07#3) or ERP record ID")


class DecisionOutput(BaseModel):
    """Final structured decision contract required by Section 2 of the project brief."""

    decision: DecisionType
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
