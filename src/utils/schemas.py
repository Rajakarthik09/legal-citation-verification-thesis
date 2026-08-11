"""Shared dataclasses used across all thesis modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseInput:
    case_id: str
    sources_text: str  # concatenated/truncated source text
    granularity: str  # "long" | "short" | "tiny"
    gold_summary: str
    truncation_strategy: str = "naive"
    source_token_count: int = 0
    truncated: bool = False


@dataclass
class ExtractedCitation:
    raw_text: str
    normalized: str
    reporter: str | None
    volume: str | None
    page: str | None
    year: int | None
    party_names: tuple[str, str] | None  # (plaintiff, defendant) if parsed
    span: tuple[int, int]  # char offsets for full citation in the summary
    pin_cite: str | None = None
    reporter_span: tuple[int, int] | None = None  # reporter-only offsets for CL alignment
    citation_id: str | None = None  # stable id for paired evaluation


@dataclass
class VerificationResult:
    citation: ExtractedCitation
    label: str  # Verified | NotFound | Mismatch | Ambiguous | Malformed
    cl_status: int | None  # raw CourtListener status code
    matched_case_name: str | None
    matched_date_filed: str | None
    explanation: str  # human-auditable, one sentence


@dataclass
class CorrectionOutcome:
    citation_id: str
    draft_label: str
    final_status: str  # verified | removed | replaced_verified | still_bad | garbled
    draft_span: tuple[int, int]
