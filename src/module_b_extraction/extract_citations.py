"""Case citation extraction via eyecite."""

from __future__ import annotations

import hashlib
import re

from eyecite import get_citations
from eyecite.models import FullCaseCitation

from utils.schemas import ExtractedCitation

# HTML tags (Multi-LexSum source documents embed markup like
# "<i>Obergefell v. Hodges</i>" or "<a href=...>...</a>" from the original
# docket/Clearinghouse pages, which PRIMERA -- being extractive-leaning --
# copies verbatim) and markdown emphasis (the LLM condition's own "**bold**"
# habit) corrupt eyecite's antecedent/party-name guessing when they sit
# directly adjacent to a case name: e.g. "Hodges</i>, 135 S. Ct. 2584"
# extracts party_names=("", "Hodges</i>") instead of ("Obergefell",
# "Hodges"). Measured impact: 18/40 "still_bad" corrections across the first
# two full-scale conditions had markup-contaminated raw_text. Blanked with
# equal-length whitespace (not deleted) so citation span offsets stay valid
# for downstream redact/replace operations on the original text.
_MARKUP_RE = re.compile(r"</?i>|</?b>|</?em>|</?strong>|\*\*|<a\s[^>]*>|</a>")


def _blank_markup(text: str) -> str:
    return _MARKUP_RE.sub(lambda m: " " * len(m.group()), text)


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _reporter_span_from_citation(citation) -> tuple[int, int]:
    span = getattr(citation, "span", None)
    if callable(span):
        span = span()
    if span is not None:
        try:
            start, end = span
            return int(start), int(end)
        except (TypeError, ValueError):
            pass
    matched = citation.matched_text()
    start = getattr(citation, "start", None)
    if start is not None:
        return int(start), int(start) + len(matched)
    return 0, len(matched)


def _full_span_from_citation(citation) -> tuple[int, int]:
    full_span = getattr(citation, "full_span", None)
    if callable(full_span):
        try:
            start, end = full_span()
            return int(start), int(end)
        except (TypeError, ValueError):
            pass
    return _reporter_span_from_citation(citation)


def _party_names_from_citation(citation) -> tuple[str, str] | None:
    meta = getattr(citation, "metadata", None)
    if meta is None:
        return None
    plaintiff = getattr(meta, "plaintiff", None)
    defendant = getattr(meta, "defendant", None)
    if plaintiff or defendant:
        return (plaintiff or "", defendant or "")
    return None


def make_citation_id(normalized: str, span: tuple[int, int]) -> str:
    """Stable identifier for pairing draft citations across conditions."""
    key = f"{normalized}|{span[0]}|{span[1]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def extract_citations(summary_text: str) -> list[ExtractedCitation]:
    """Extract full case citations from generated summary text."""
    if not summary_text or not summary_text.strip():
        return []

    cleaned_text = _blank_markup(summary_text)
    results: list[ExtractedCitation] = []
    for c in get_citations(cleaned_text):
        if not isinstance(c, FullCaseCitation):
            continue
        meta = getattr(c, "metadata", None)
        year = _safe_int(getattr(meta, "year", None)) if meta else None
        pin_cite = getattr(meta, "pin_cite", None) if meta else None
        reporter = getattr(c, "reporter", None) or getattr(c, "reporter_name", None)
        volume = getattr(c, "volume", None)
        page = getattr(c, "page", None)
        if volume is not None:
            volume = str(volume)
        if page is not None:
            page = str(page)

        normalized = str(c.corrected_citation())
        reporter_span = _reporter_span_from_citation(c)
        full_span = _full_span_from_citation(c)
        raw_text = c.corrected_citation_full()

        results.append(
            ExtractedCitation(
                raw_text=raw_text,
                normalized=normalized,
                reporter=str(reporter) if reporter else None,
                volume=volume,
                page=page,
                year=year,
                party_names=_party_names_from_citation(c),
                span=full_span,
                pin_cite=str(pin_cite) if pin_cite else None,
                reporter_span=reporter_span,
                citation_id=make_citation_id(normalized, full_span),
            )
        )
    return results
