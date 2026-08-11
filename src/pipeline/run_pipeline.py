"""End-to-end dual-system pipeline: generate → verify → correct."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from correction_loop.corrector import correct_with_config
from evaluation.paired_hr import correction_outcome_to_dict, paired_hr_from_outcomes
from module_a_summarizer.model import Summarizer
from module_b_extraction.extract_citations import extract_citations
from module_c_verifier.verify import Verifier, hallucination_rate, verification_result_to_dict
from utils.schemas import CaseInput


def _serialize_verification(results) -> list[dict[str, Any]]:
    out = []
    for r in results:
        row = verification_result_to_dict(r)
        out.append(row)
    return out


def run_case(
    case: CaseInput,
    summarizer: Summarizer,
    verifier: Verifier,
    config: dict[str, Any],
    *,
    draft_summary: str | None = None,
) -> dict[str, Any]:
    """Run full pipeline on one case."""
    module_a = config.get("module_a", {})
    granularity = case.granularity
    max_new_tokens = module_a.get("max_new_tokens", {}).get(granularity, 850)
    min_new_tokens = module_a.get("min_new_tokens", {}).get(granularity)
    length_penalty = module_a.get("length_penalty", 2.0)
    early_stopping = module_a.get("early_stopping", False)
    num_beams = module_a.get("num_beams", 5)

    draft = draft_summary
    if draft is None:
        draft = summarizer.generate(
            case.sources_text,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )

    draft_extracted = extract_citations(draft)
    draft_verified = verifier.verify_summary(draft, draft_extracted)
    hr_draft = hallucination_rate(draft_verified)

    final_text, n_iter, final_verified, outcomes = correct_with_config(draft, verifier, config)
    hr_final_paired = paired_hr_from_outcomes(outcomes)
    hr_final_surviving = hallucination_rate(final_verified)

    return {
        "case_id": case.case_id,
        "granularity": case.granularity,
        "gold_summary": case.gold_summary,
        "draft_summary": draft,
        "final_summary": final_text,
        "citations_draft": _serialize_verification(draft_verified),
        "citations_final": _serialize_verification(final_verified),
        "hr_draft": hr_draft,
        "hr_final": hr_final_paired,
        "hr_final_surviving": hr_final_surviving,
        "correction_outcomes": [correction_outcome_to_dict(o) for o in outcomes],
        "correction_iterations": n_iter,
        "truncated": case.truncated,
        "source_token_count": case.source_token_count,
    }


def run_batch(
    cases: list[CaseInput],
    summarizer: Summarizer,
    verifier: Verifier,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    generations: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run pipeline on a batch; write per-case JSON + batch_summary.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    for case in cases:
        draft = generations.get(case.case_id) if generations else None
        report = run_case(case, summarizer, verifier, config, draft_summary=draft)
        reports.append(report)
        case_path = output_dir / f"{report['case_id']}.json"
        case_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "n_cases": len(reports),
        "mean_hr_draft": sum(r["hr_draft"] for r in reports) / max(len(reports), 1),
        "mean_hr_final": sum(r["hr_final"] for r in reports) / max(len(reports), 1),
        "cache_hits": verifier.cache_hits,
        "cache_misses": verifier.cache_misses,
        "cases": [
            {
                "case_id": r["case_id"],
                "hr_draft": r["hr_draft"],
                "hr_final": r["hr_final"],
                "correction_iterations": r["correction_iterations"],
                "n_citations_draft": len(r["citations_draft"]),
            }
            for r in reports
        ],
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return reports


def merge_configs(
    experiment: dict[str, Any],
    module_a: dict[str, Any],
    module_c: dict[str, Any],
) -> dict[str, Any]:
    """Merge YAML configs into a single runtime config dict."""
    return {
        "split": experiment.get("split", "validation"),
        "granularity": experiment.get("granularity", "long"),
        "max_cases": experiment.get("max_cases", 10),
        "output_dir": experiment.get("output_dir"),
        "module_a": module_a,
        "module_c": module_c,
        **module_c,
    }
