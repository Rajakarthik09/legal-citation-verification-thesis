"""Paired draft-citation hallucination rate (primary evaluation metric)."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from module_b_extraction.extract_citations import extract_citations, make_citation_id
from utils.schemas import CorrectionOutcome, ExtractedCitation, VerificationResult

BAD_OUTCOMES = frozenset({"still_bad", "garbled"})
SUCCESS_OUTCOMES = frozenset({"verified", "replaced_verified"})
PARTIAL_OUTCOMES = frozenset({"removed"})


def paired_hr_from_outcomes(outcomes: list[CorrectionOutcome]) -> float:
    """HR over original draft citations: (still_bad + garbled) / n_draft."""
    if not outcomes:
        return 0.0
    bad = sum(1 for o in outcomes if o.final_status in BAD_OUTCOMES)
    return bad / len(outcomes)


def outcome_rates(outcomes: list[CorrectionOutcome]) -> dict[str, float]:
    if not outcomes:
        return {
            "fix_rate": 0.0,
            "removal_rate": 0.0,
            "garbled_rate": 0.0,
            "still_bad_rate": 0.0,
        }
    n = len(outcomes)
    return {
        "fix_rate": sum(1 for o in outcomes if o.final_status in SUCCESS_OUTCOMES) / n,
        "removal_rate": sum(1 for o in outcomes if o.final_status in PARTIAL_OUTCOMES) / n,
        "garbled_rate": sum(1 for o in outcomes if o.final_status == "garbled") / n,
        "still_bad_rate": sum(1 for o in outcomes if o.final_status == "still_bad") / n,
    }


def micro_paired_hr(outcome_lists: list[list[CorrectionOutcome]]) -> float:
    all_outcomes = [o for lst in outcome_lists for o in lst]
    return paired_hr_from_outcomes(all_outcomes)


def macro_paired_hr(outcome_lists: list[list[CorrectionOutcome]]) -> float:
    if not outcome_lists:
        return 0.0
    per_case = [paired_hr_from_outcomes(lst) for lst in outcome_lists if lst]
    if not per_case:
        return 0.0
    return sum(per_case) / len(per_case)


def _citation_id_for(ext: ExtractedCitation) -> str:
    if ext.citation_id:
        return ext.citation_id
    return make_citation_id(ext.normalized, ext.span)


def classify_outcomes(
    draft_results: list[VerificationResult],
    final_text: str,
    final_results: list[VerificationResult],
) -> list[CorrectionOutcome]:
    """
    Classify each draft citation's post-correction status.

    Outcomes:
    - verified: draft was already Verified
    - replaced_verified: Mismatch/Ambiguous fixed (CL repair or same-case re-anchor)
    - removed: redacted (NotFound/Malformed; rarely Ambiguous)
    - still_bad: Ambiguous left unresolved, or still-bad after correction
    - garbled: unparseable remnant or garbled replacement
    """
    outcomes: list[CorrectionOutcome] = []
    for draft in draft_results:
        cite = draft.citation
        cid = _citation_id_for(cite)
        draft_label = draft.label

        if draft_label == "Verified":
            outcomes.append(
                CorrectionOutcome(
                    citation_id=cid,
                    draft_label=draft_label,
                    final_status="verified",
                    draft_span=cite.span,
                )
            )
            continue

        # Ambiguous: leave unresolved unless re-anchored to a Verified cite.
        if draft_label == "Ambiguous":
            start, end = cite.span
            region = final_text[start:end] if start < len(final_text) else ""
            if REDACTION in region or not region.strip():
                outcomes.append(
                    CorrectionOutcome(
                        citation_id=cid,
                        draft_label=draft_label,
                        final_status="removed",
                        draft_span=cite.span,
                    )
                )
                continue
            nearby = [r for r in final_results if abs(r.citation.span[0] - start) < 80]
            if nearby and nearby[0].label == "Verified":
                outcomes.append(
                    CorrectionOutcome(
                        citation_id=cid,
                        draft_label=draft_label,
                        final_status="replaced_verified",
                        draft_span=cite.span,
                    )
                )
                continue
            outcomes.append(
                CorrectionOutcome(
                    citation_id=cid,
                    draft_label=draft_label,
                    final_status="still_bad",
                    draft_span=cite.span,
                )
            )
            continue

        if draft_label in ("NotFound", "Malformed"):
            outcomes.append(
                CorrectionOutcome(
                    citation_id=cid,
                    draft_label=draft_label,
                    final_status="removed",
                    draft_span=cite.span,
                )
            )
            continue

        # Mismatch: check if fixed in final text
        start, end = cite.span
        region = final_text[start:end] if start < len(final_text) else ""
        if REDACTION in region or not region.strip():
            outcomes.append(
                CorrectionOutcome(
                    citation_id=cid,
                    draft_label=draft_label,
                    final_status="removed",
                    draft_span=cite.span,
                )
            )
            continue

        nearby = [r for r in final_results if abs(r.citation.span[0] - start) < 80]
        if nearby:
            best = nearby[0]
            if best.label == "Verified":
                outcomes.append(
                    CorrectionOutcome(
                        citation_id=cid,
                        draft_label=draft_label,
                        final_status="replaced_verified",
                        draft_span=cite.span,
                    )
                )
                continue
            if best.label in ("NotFound", "Mismatch", "Ambiguous", "Malformed"):
                outcomes.append(
                    CorrectionOutcome(
                        citation_id=cid,
                        draft_label=draft_label,
                        final_status="still_bad",
                        draft_span=cite.span,
                    )
                )
                continue

        local = final_text[max(0, start - 20) : min(len(final_text), end + 60)]
        local_cites = extract_citations(local)
        if not local_cites and " v. " in local and re.search(r"\(\d{4}\)", local):
            outcomes.append(
                CorrectionOutcome(
                    citation_id=cid,
                    draft_label=draft_label,
                    final_status="garbled",
                    draft_span=cite.span,
                )
            )
            continue

        outcomes.append(
            CorrectionOutcome(
                citation_id=cid,
                draft_label=draft_label,
                final_status="still_bad",
                draft_span=cite.span,
            )
        )

    return outcomes


REDACTION = "[unverifiable citation removed]"


def correction_outcome_to_dict(o: CorrectionOutcome) -> dict[str, Any]:
    return asdict(o)
