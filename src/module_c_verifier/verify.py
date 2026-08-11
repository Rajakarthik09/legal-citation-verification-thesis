"""Symbolic verification of case citations against CourtListener."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from module_c_verifier.cache import CitationCache
from module_c_verifier.courtlistener_client import lookup_citations
from module_c_verifier.rate_limiter import RateLimiter
from module_b_extraction.extract_citations import extract_citations
from utils.schemas import ExtractedCitation, VerificationResult

BAD_LABELS = frozenset({"NotFound", "Mismatch", "Ambiguous", "Malformed"})
CONTRADICTION_THRESHOLD = 70


def _cluster_year(cluster: dict[str, Any]) -> int | None:
    date_filed = cluster.get("date_filed") or ""
    if date_filed:
        m = re.search(r"(\d{4})", str(date_filed))
        if m:
            return int(m.group(1))
    return None


def _years_compatible(extracted_year: int | None, cluster_year: int | None) -> bool:
    if extracted_year is None or cluster_year is None:
        return True
    return abs(extracted_year - cluster_year) <= 1


def draft_case_name_hint(extracted: ExtractedCitation) -> str | None:
    """Best-effort party string from extraction or raw citation text."""
    if extracted.party_names:
        plaintiff, defendant = extracted.party_names
        if plaintiff and defendant:
            return f"{plaintiff} v. {defendant}".strip()
    raw = (extracted.raw_text or "").strip()
    if " v. " in raw:
        return raw.split(",", 1)[0].strip()
    return None


def _party_names_contradict(
    extracted: ExtractedCitation,
    case_name: str,
    *,
    threshold: int = CONTRADICTION_THRESHOLD,
) -> bool:
    """True when draft parties clearly disagree with the cluster name."""
    if not case_name:
        return False
    combined = draft_case_name_hint(extracted)
    if not combined:
        return False
    return fuzz.token_sort_ratio(combined, case_name) < threshold


def is_unsafe_mismatch_repair(
    extracted: ExtractedCitation,
    matched_case_name: str | None,
    *,
    threshold: int = CONTRADICTION_THRESHOLD,
) -> bool:
    """
    True when replacing a Mismatch with CL's case name would swap in a
    different matter (e.g. Brown → Hydraulic). Such cites should be left
    as Ambiguous, not symbolically rewritten.
    """
    if not matched_case_name:
        return True
    return _party_names_contradict(
        extracted, matched_case_name, threshold=threshold
    )


def fields_match(
    extracted: ExtractedCitation,
    cluster: dict[str, Any],
    *,
    fuzzy_threshold: int = 85,
) -> bool:
    """Fuzzy-compare party names and year against a CourtListener cluster."""
    case_name = cluster.get("case_name") or ""
    cluster_year = _cluster_year(cluster)

    if not _years_compatible(extracted.year, cluster_year):
        return False

    if extracted.party_names:
        plaintiff, defendant = extracted.party_names
        for party in (plaintiff, defendant):
            if party and fuzz.token_sort_ratio(party, case_name) >= fuzzy_threshold:
                return True
        combined = f"{plaintiff} v. {defendant}".strip()
        if combined and fuzz.token_sort_ratio(combined, case_name) >= fuzzy_threshold:
            return True
        if _party_names_contradict(extracted, case_name):
            return False

    if extracted.normalized and fuzz.partial_ratio(extracted.normalized, case_name) >= fuzzy_threshold:
        return True

    # Reporter-only cite with compatible year: trust CL 200 resolution.
    if not extracted.party_names and _years_compatible(extracted.year, cluster_year):
        return True

    return extracted.year is None and not extracted.party_names


def build_verified_citation_text(
    matched_case_name: str,
    extracted: ExtractedCitation,
    *,
    matched_date_filed: str | None = None,
) -> str:
    """Build a full corrected citation string for symbolic replacement."""
    parts = [matched_case_name, extracted.normalized]
    text = ", ".join(parts)
    if extracted.pin_cite:
        text = f"{text}, {extracted.pin_cite}"
    # Prefer CourtListener's verified year over the draft's own year, not
    # just when the draft year is missing. A Mismatch is frequently exactly
    # a year discrepancy on an otherwise-correctly-identified case (e.g. a
    # real "Obergefell v. Hodges" cite with a hallucinated year) -- keeping
    # the draft's (wrong) year here means the "repaired" text still fails
    # re-verification and the case exhausts max_correction_iterations
    # landing on still_bad, defeating the point of the repair.
    year = None
    if matched_date_filed:
        year = _cluster_year({"date_filed": matched_date_filed})
    if year is None:
        year = extracted.year
    if year is not None:
        text = f"{text} ({year})"
    return text


def _normalize_cite(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _span_for_cl_alignment(ext: ExtractedCitation) -> tuple[int, int]:
    return ext.reporter_span if ext.reporter_span is not None else ext.span


def align_citations(
    extracted: list[ExtractedCitation],
    cl_response: list[dict],
) -> list[tuple[ExtractedCitation, dict | None]]:
    """Pair extracted citations with CourtListener lookup items."""
    used: set[int] = set()
    pairs: list[tuple[ExtractedCitation, dict | None]] = []
    for ext in extracted:
        best_idx = None
        best_score = -1
        ext_norm = _normalize_cite(ext.normalized or ext.raw_text)
        align_span = _span_for_cl_alignment(ext)
        for i, item in enumerate(cl_response):
            if i in used:
                continue
            cite = item.get("citation") or item.get("normalized_citations", [""])[0]
            score = fuzz.ratio(ext_norm, _normalize_cite(str(cite)))
            start = item.get("start_index")
            end = item.get("end_index")
            if start is not None and end is not None:
                cl_span = (int(start), int(end))
                if align_span == cl_span or ext.span == cl_span:
                    score = max(score, 100)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None and best_score >= 60:
            used.add(best_idx)
            pairs.append((ext, cl_response[best_idx]))
        else:
            pairs.append((ext, None))
    return pairs


def _label_from_status(
    status: int | None,
    extracted: ExtractedCitation,
    cluster: dict | None,
    *,
    fuzzy_threshold: int,
) -> tuple[str, str, str | None, str | None]:
    if status == 404:
        return "NotFound", "No matching citation in CourtListener's database.", None, None
    if status == 300:
        return "Ambiguous", "Citation matches multiple case clusters.", None, None
    if status == 400:
        return "Malformed", "Citation could not be parsed as a valid reporter.", None, None
    if status == 200 and cluster:
        case_name = cluster.get("case_name")
        date_filed = cluster.get("date_filed")
        if fields_match(extracted, cluster, fuzzy_threshold=fuzzy_threshold):
            return (
                "Verified",
                "Citation and metadata both confirmed.",
                case_name,
                date_filed,
            )
        # Hard party contradiction: wrong CL cluster identity — leave Ambiguous.
        if case_name and _party_names_contradict(extracted, case_name):
            return (
                "Ambiguous",
                (
                    "Citation resolves to a cluster whose parties contradict the "
                    f"draft case name ({case_name}, {date_filed}); left unresolved."
                ),
                case_name,
                date_filed,
            )
        return (
            "Mismatch",
            f"Citation resolves, but metadata does not match cluster ({case_name}, {date_filed}).",
            case_name,
            date_filed,
        )
    return "Malformed", "Unexpected CourtListener response.", None, None


class Verifier:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        cache_path = config.get("cache_path", "data/processed/citation_cache.sqlite")
        self.cache = CitationCache(cache_path)
        rl = config.get("rate_limit", {})
        default_state_path = str(Path(cache_path).with_name("rate_limiter_state.json"))
        self.rate_limiter = RateLimiter(
            per_minute=rl.get("valid_citations_per_minute", 10),
            per_hour=rl.get("valid_citations_per_hour", 100),
            daily_cap=rl.get("daily_cap", 250),
            state_path=config.get("rate_limiter_state_path", default_state_path),
        )
        self.fuzzy_threshold = config.get("fuzzy_match_threshold", 85)
        self.cache_hits = 0
        self.cache_misses = 0

    def get_or_fetch(self, summary_text: str) -> list[dict]:
        cached = self.cache.get(summary_text)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        # rate_limiter.acquire() now happens inside lookup_citations() itself
        # (called once per real HTTP attempt, including tenacity retries) --
        # not here, so retries triggered by a 429 also pace/count correctly.
        response = lookup_citations(
            summary_text,
            url=self.config.get("courtlistener_url"),
            rate_limiter=self.rate_limiter,
        )
        self.cache.set(summary_text, response)
        return response

    def verify_summary(
        self,
        summary_text: str,
        extracted: list[ExtractedCitation] | None = None,
    ) -> list[VerificationResult]:
        if extracted is None:
            extracted = extract_citations(summary_text)
        if not extracted:
            return []
        cl_response = self.get_or_fetch(summary_text)
        results: list[VerificationResult] = []
        for ext, cl_item in align_citations(extracted, cl_response):
            if cl_item is None:
                results.append(
                    VerificationResult(
                        citation=ext,
                        label="NotFound",
                        cl_status=404,
                        matched_case_name=None,
                        matched_date_filed=None,
                        explanation="No aligned CourtListener lookup item for extracted citation.",
                    )
                )
                continue
            status = cl_item.get("status")
            clusters = cl_item.get("clusters") or []
            cluster = clusters[0] if clusters else None
            label, expl, case_name, date_filed = _label_from_status(
                status,
                ext,
                cluster,
                fuzzy_threshold=self.fuzzy_threshold,
            )
            results.append(
                VerificationResult(
                    citation=ext,
                    label=label,
                    cl_status=status,
                    matched_case_name=case_name,
                    matched_date_filed=date_filed,
                    explanation=expl,
                )
            )
        return results

    def close(self) -> None:
        self.cache.close()


def hallucination_rate(results: list[VerificationResult]) -> float:
    """HR among surviving extracted citations (secondary diagnostic metric)."""
    if not results:
        return 0.0
    bad = sum(1 for r in results if r.label in BAD_LABELS)
    return bad / len(results)


def verification_result_to_dict(v: VerificationResult) -> dict[str, Any]:
    d = asdict(v)
    return d
