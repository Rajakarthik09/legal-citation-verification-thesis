"""Module C — symbolic CourtListener verifier."""

from module_c_verifier.verify import (
    Verifier,
    align_citations,
    fields_match,
    hallucination_rate,
    verification_result_to_dict,
)

__all__ = [
    "Verifier",
    "align_citations",
    "fields_match",
    "hallucination_rate",
    "verification_result_to_dict",
]
