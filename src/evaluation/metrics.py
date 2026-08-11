"""Summarization quality and citation faithfulness metrics."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from rouge_score import rouge_scorer

from module_c_verifier.verify import BAD_LABELS, hallucination_rate
from utils.schemas import VerificationResult

# rougeLsum -- not plain rougeL -- is the ROUGE-L variant conventionally reported
# for multi-sentence summarization, and the one comparable to published
# Multi-LexSum baselines. Plain rougeL takes one LCS over the whole flattened
# summary, which penalises long multi-sentence text heavily (measured here:
# 0.1995 vs 0.4074 on the same PRIMERA drafts). Both are computed and reported;
# rougeLsum is the primary quality metric, rougeL is kept for transparency.
ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")
PRIMARY_ROUGE = "rougeLsum"

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences_to_lines(text: str) -> str:
    """
    Put one sentence per line.

    ``rouge_score`` computes rougeLsum as a union-LCS over newline-delimited
    sentences, so text with no newlines collapses to plain rougeL. Standard
    summarization evaluation therefore sentence-splits before scoring. Only
    rougeLsum is affected: rouge1/rouge2/rougeL tokenize away whitespace, so
    their scores are identical with or without this step.
    """
    parts = (s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text.strip()))
    return "\n".join(p for p in parts if p)


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L/Lsum F1 scores (rougeLsum sentence-split, see module notes)."""
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
    scores = scorer.score(
        split_sentences_to_lines(reference),
        split_sentences_to_lines(prediction),
    )
    return {k: scores[k].fmeasure for k in ROUGE_TYPES}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    *,
    device: str | None = None,
    batch_size: int = 8,
) -> list[float]:
    """Per-pair BERTScore F1 (returns one score per pair)."""
    if not predictions:
        return []
    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:
        raise ImportError("bert-score is required for BERTScore metrics") from exc

    import torch

    dev = device
    if dev is None:
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

    _, _, f1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        device=dev,
        batch_size=batch_size,
        verbose=False,
    )
    return [float(x) for x in f1.tolist()]


def compute_bertscore_single(
    prediction: str,
    reference: str,
    *,
    device: str | None = None,
) -> float:
    scores = compute_bertscore([prediction], [reference], device=device)
    return scores[0] if scores else 0.0


def hr_from_citation_dicts(citations: list[dict[str, Any]]) -> float:
    """HR from serialized citation verification dicts."""
    if not citations:
        return 0.0
    results = [
        VerificationResult(
            citation=_dict_to_citation(c["citation"]),
            label=c["label"],
            cl_status=c.get("cl_status"),
            matched_case_name=c.get("matched_case_name"),
            matched_date_filed=c.get("matched_date_filed"),
            explanation=c.get("explanation", ""),
        )
        for c in citations
    ]
    return hallucination_rate(results)


def _dict_to_citation(d: dict[str, Any]):
    from utils.schemas import ExtractedCitation

    parties = d.get("party_names")
    if parties is not None and not isinstance(parties, tuple):
        parties = tuple(parties) if parties else None
    span = d.get("span", (0, 0))
    if isinstance(span, list):
        span = tuple(span)
    reporter_span = d.get("reporter_span")
    if reporter_span is not None and isinstance(reporter_span, list):
        reporter_span = tuple(reporter_span)
    return ExtractedCitation(
        raw_text=d["raw_text"],
        normalized=d["normalized"],
        reporter=d.get("reporter"),
        volume=d.get("volume"),
        page=d.get("page"),
        year=d.get("year"),
        party_names=parties,
        span=span,
        pin_cite=d.get("pin_cite"),
        reporter_span=reporter_span,
        citation_id=d.get("citation_id"),
    )


def citation_label_counts(citations: list[dict[str, Any]]) -> Counter[str]:
    """Count verification labels across all citations."""
    return Counter(c.get("label", "Unknown") for c in citations)


def micro_hr(citation_lists: list[list[dict[str, Any]]]) -> float:
    """Micro-averaged HR across cases."""
    total = 0
    bad = 0
    for cites in citation_lists:
        for c in cites:
            total += 1
            if c.get("label") in BAD_LABELS:
                bad += 1
    return bad / total if total else 0.0


def macro_hr(citation_lists: list[list[dict[str, Any]]]) -> float:
    """Macro-averaged HR (mean of per-case HR)."""
    if not citation_lists:
        return 0.0
    return sum(hr_from_citation_dicts(c) for c in citation_lists) / len(citation_lists)


def correction_efficiency_stats(iterations: list[int]) -> dict[str, float]:
    """Distribution stats for correction iterations."""
    if not iterations:
        return {
            "mean_iterations": 0.0,
            "median_iterations": 0.0,
            "pct_within_1": 0.0,
            "pct_within_2": 0.0,
        }
    n = len(iterations)
    sorted_iters = sorted(iterations)
    median = sorted_iters[n // 2]
    within_1 = sum(1 for x in iterations if x <= 1) / n
    within_2 = sum(1 for x in iterations if x <= 2) / n
    return {
        "mean_iterations": sum(iterations) / n,
        "median_iterations": float(median),
        "pct_within_1": within_1,
        "pct_within_2": within_2,
    }


def compute_case_metrics(
    summary: str,
    gold: str,
    citations: list[dict[str, Any]],
    *,
    bertscore_device: str | None = None,
    skip_bertscore: bool = False,
) -> dict[str, Any]:
    """Full metric bundle for one case."""
    rouge = compute_rouge(summary, gold)
    metrics: dict[str, Any] = {
        **{f"rouge_{k.replace('rouge', '').lower()}": v for k, v in rouge.items()},
        "hr": hr_from_citation_dicts(citations),
        "n_citations": len(citations),
        "label_counts": dict(citation_label_counts(citations)),
    }
    if not skip_bertscore:
        metrics["bertscore_f1"] = compute_bertscore_single(
            summary, gold, device=bertscore_device
        )
    return metrics
