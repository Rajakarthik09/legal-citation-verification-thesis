"""Bounded symbolic correction loop for citation hallucinations."""

from __future__ import annotations

import re
from typing import Any, Callable

from rapidfuzz import fuzz

from evaluation.paired_hr import classify_outcomes
from module_b_extraction.extract_citations import extract_citations
from module_c_verifier.verify import (
    Verifier,
    build_verified_citation_text,
    draft_case_name_hint,
    hallucination_rate,
    is_unsafe_mismatch_repair,
)
from utils.schemas import CorrectionOutcome, VerificationResult

REDACTION = "[unverifiable citation removed]"
REANCHOR_PARTY_THRESHOLD = 70


def _safe_mismatch_repairs(
    results: list[VerificationResult],
) -> list[VerificationResult]:
    """Mismatches whose CL case name is compatible enough to substitute."""
    return [
        r
        for r in results
        if r.label == "Mismatch"
        and not is_unsafe_mismatch_repair(r.citation, r.matched_case_name)
    ]


def _needs_reanchor(r: VerificationResult) -> bool:
    """Ambiguous / unsafe mismatch that may be replaced by a verified same-case cite."""
    if r.label == "Ambiguous":
        return True
    if r.label == "Mismatch" and is_unsafe_mismatch_repair(
        r.citation, r.matched_case_name
    ):
        return True
    return False


def find_reanchor_source(
    bad: VerificationResult,
    results: list[VerificationResult],
    *,
    threshold: int = REANCHOR_PARTY_THRESHOLD,
) -> VerificationResult | None:
    """
    Find a Verified citation in the same summary whose parties match the bad cite.

    Used when a LEXIS/WL id resolves to the wrong CL cluster (e.g. Brown → Hydraulic)
    but another cite in the draft already verified the correct matter.
    """
    hint = draft_case_name_hint(bad.citation)
    if not hint:
        return None

    best: VerificationResult | None = None
    best_score = -1
    bad_id = bad.citation.citation_id
    for r in results:
        if r.label != "Verified" or not r.matched_case_name:
            continue
        if bad_id and r.citation.citation_id == bad_id:
            continue
        score = max(
            fuzz.token_set_ratio(hint, r.matched_case_name),
            fuzz.token_sort_ratio(hint, r.matched_case_name),
        )
        first = hint.split(" v. ", 1)[0].strip()
        if first and len(first) >= 3:
            score = max(score, fuzz.partial_ratio(first, r.matched_case_name))
        if score > best_score:
            best_score = score
            best = r

    if best is None or best_score < threshold:
        return None
    return best


def build_reanchor_replacement(anchor: VerificationResult) -> str:
    """Full citation string taken from a verified same-summary cite."""
    return build_verified_citation_text(
        anchor.matched_case_name or anchor.citation.raw_text,
        anchor.citation,
        matched_date_filed=anchor.matched_date_filed,
    )


def redact_span(text: str, span: tuple[int, int], replacement: str = REDACTION) -> str:
    start, end = span
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    return text[:start] + replacement + text[end:]


def replace_span(text: str, span: tuple[int, int], replacement: str) -> str:
    start, end = span
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    return text[:start] + replacement + text[end:]


def _extend_span_past_trailing_pin(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """Include a leftover ', at *1292' (or similar) just after the citation span."""
    start, end = span
    m = re.match(r"(?:,\s*)?(?:at\s*)?\*\d+", text[end:])
    if m:
        return (start, end + m.end())
    return span


def _is_garbled_replacement(text: str, span: tuple[int, int], replacement: str) -> bool:
    """Detect hybrid citation fragments after replacement."""
    start, end = span
    window_start = max(0, start - 40)
    window_end = min(len(text), end + 40)
    window = text[window_start:window_end]

    if " v. " in replacement:
        party_part = replacement.split(",", 1)[0].strip()
        if party_part and window.count(party_part) > 1:
            return True

    reporter_hint = replacement.split(",", 1)[-1] if "," in replacement else replacement
    has_reporter_digits = any(ch.isdigit() for ch in reporter_hint[:8])
    after = window[window.find(replacement) + len(replacement) :] if replacement in window else ""
    if not has_reporter_digits and re.search(r"\(\d{4}\)", after):
        return True

    return False


def correct(
    summary_text: str,
    verifier: Verifier,
    *,
    max_iterations: int = 3,
    verify_fn: Callable[[str, list | None], list[VerificationResult]] | None = None,
) -> tuple[str, int, list[VerificationResult], list[CorrectionOutcome]]:
    """
    Symbolic correction:
    - redact NotFound/Malformed
    - replace safe Mismatches with CL metadata
    - re-anchor Ambiguous / unsafe mismatches to a Verified same-case cite when available
    - otherwise leave Ambiguous in place (still_bad in paired HR)

    Returns (final_text, iterations_used, final_verification_results, correction_outcomes).
    """
    verify = verify_fn or verifier.verify_summary
    text = summary_text
    draft_results = verify(text, None)
    results = draft_results
    iterations = 0

    for i in range(max_iterations):
        redact_labels = ("NotFound", "Malformed")
        bad = [r for r in results if r.label in redact_labels]
        fixable = _safe_mismatch_repairs(results)

        reanchors: list[tuple[VerificationResult, VerificationResult]] = []
        for r in results:
            if not _needs_reanchor(r):
                continue
            if r in fixable:
                continue
            anchor = find_reanchor_source(r, results)
            if anchor is not None:
                reanchors.append((r, anchor))

        if not bad and not fixable and not reanchors:
            break

        iterations = i + 1
        edits: list[tuple[tuple[int, int], str]] = []
        for r in bad:
            edits.append((r.citation.span, REDACTION))
        for r in fixable:
            replacement = build_verified_citation_text(
                r.matched_case_name or r.citation.raw_text,
                r.citation,
                matched_date_filed=r.matched_date_filed,
            )
            edits.append((r.citation.span, replacement))
        for r, anchor in reanchors:
            edits.append((r.citation.span, build_reanchor_replacement(anchor)))

        for span, replacement in sorted(edits, key=lambda x: x[0][0], reverse=True):
            span = _extend_span_past_trailing_pin(text, span)
            if replacement == REDACTION:
                text = redact_span(text, span)
                continue
            candidate = replace_span(text, span, replacement)
            if _is_garbled_replacement(candidate, span, replacement):
                text = redact_span(text, span)
            else:
                text = candidate

        results = verify(text, None)

    outcomes = classify_outcomes(draft_results, text, results)
    return text, iterations, results, outcomes


def correct_with_config(
    summary_text: str,
    verifier: Verifier,
    config: dict[str, Any],
    *,
    rewriter=None,
) -> tuple[str, int, list[VerificationResult], list[CorrectionOutcome]]:
    mode = config.get("correction_mode", "symbolic")
    max_iter = config.get("max_correction_iterations", 3)

    if mode == "symbolic":
        return correct(summary_text, verifier, max_iterations=max_iter)

    if mode == "llm_rewrite":
        sym_text, sym_iter, _, sym_outcomes = correct(
            summary_text, verifier, max_iterations=max_iter
        )
        if rewriter is None:
            from correction_loop.llm_rewrite import create_rewriter_from_config

            rewriter = create_rewriter_from_config(config)
        smoothed = rewriter.smooth_text(sym_text)
        final_verified = verifier.verify_summary(smoothed, None)
        final_outcomes = classify_outcomes(
            verifier.verify_summary(summary_text, None),
            smoothed,
            final_verified,
        )
        return smoothed, sym_iter, final_verified, final_outcomes

    raise NotImplementedError(f"correction_mode {mode!r} not implemented")
