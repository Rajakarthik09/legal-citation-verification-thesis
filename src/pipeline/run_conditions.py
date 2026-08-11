"""Multi-condition pipeline runner (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tenacity import RetryError
from tqdm import tqdm

from correction_loop.corrector import correct
from correction_loop.llm_rewrite import LLMRewriter, create_rewriter_from_config
from evaluation.paired_hr import (
    classify_outcomes,
    correction_outcome_to_dict,
    paired_hr_from_outcomes,
)
from module_b_extraction.extract_citations import extract_citations
from module_c_verifier.verify import Verifier, hallucination_rate, verification_result_to_dict
from utils.schemas import CaseInput


def is_api_budget_error(exc: BaseException) -> bool:
    """
    True if CourtListener's daily cap or a 429 rate limit was hit.

    Walks __cause__/__context__ because tenacity wraps the underlying
    RuntimeError (raised in courtlistener_client.lookup_citations on a 429)
    in its own RetryError once retries are exhausted -- str(RetryError(...))
    doesn't contain the original message, only "raised RuntimeError".
    """
    return _budget_error_message(exc) is not None


def _budget_error_message(exc: BaseException) -> str | None:
    """The underlying rate-limit/cap message, or None if exc isn't one."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current)
        low = msg.lower()
        if "daily courtlistener cap" in low or "rate limited" in low:
            return msg
        current = current.__cause__ or current.__context__
    return None


def _serialize_verification(results) -> list[dict[str, Any]]:
    return [verification_result_to_dict(r) for r in results]


def _serialize_outcomes(outcomes) -> list[dict[str, Any]]:
    return [correction_outcome_to_dict(o) for o in outcomes]


def run_all_conditions(
    case: CaseInput,
    verifier: Verifier,
    config: dict[str, Any],
    draft_summary: str,
    *,
    rewriter: LLMRewriter | None = None,
) -> dict[str, Any]:
    """
    Run baseline, flagging, symbolic, and llm_rewrite conditions for one case.

    Uses a shared draft_summary (from Module A) — no re-generation.
    Symbolic correction runs once; llm_rewrite reuses sym_text (saves API calls).
    """
    draft_extracted = extract_citations(draft_summary)
    draft_verified = verifier.verify_summary(draft_summary, draft_extracted)
    hr_draft = hallucination_rate(draft_verified)

    sym_text, sym_iter, sym_verified, sym_outcomes = correct(
        draft_summary, verifier, max_iterations=config.get("max_correction_iterations", 3)
    )
    hr_sym_paired = paired_hr_from_outcomes(sym_outcomes)
    hr_sym_surviving = hallucination_rate(sym_verified)

    llm_text, llm_iter, llm_verified, llm_outcomes = _correct_llm_rewrite(
        verifier,
        config,
        draft_verified=draft_verified,
        sym_text=sym_text,
        sym_iter=sym_iter,
        rewriter=rewriter,
    )
    hr_llm_paired = paired_hr_from_outcomes(llm_outcomes)
    hr_llm_surviving = hallucination_rate(llm_verified)

    return {
        "case_id": case.case_id,
        "granularity": case.granularity,
        "gold_summary": case.gold_summary,
        "draft_summary": draft_summary,
        "citations_draft": _serialize_verification(draft_verified),
        "hr_draft": hr_draft,
        "truncated": case.truncated,
        "source_token_count": case.source_token_count,
        "conditions": {
            "baseline": {
                "summary": draft_summary,
                "citations": _serialize_verification(draft_verified),
                "hr": hr_draft,
                "hr_paired": hr_draft,
            },
            "flagging": {
                "summary": draft_summary,
                "citations": _serialize_verification(draft_verified),
                "hr": hr_draft,
                "hr_paired": hr_draft,
                "verification_only": True,
            },
            "symbolic": {
                "final_summary": sym_text,
                "citations_final": _serialize_verification(sym_verified),
                "hr_final": hr_sym_paired,
                "hr_surviving": hr_sym_surviving,
                "hr_paired": hr_sym_paired,
                "correction_outcomes": _serialize_outcomes(sym_outcomes),
                "correction_iterations": sym_iter,
            },
            "llm_rewrite": {
                "final_summary": llm_text,
                "citations_final": _serialize_verification(llm_verified),
                "hr_final": hr_llm_paired,
                "hr_surviving": hr_llm_surviving,
                "hr_paired": hr_llm_paired,
                "correction_outcomes": _serialize_outcomes(llm_outcomes),
                "correction_iterations": llm_iter,
            },
        },
    }


def _correct_llm_rewrite(
    verifier: Verifier,
    config: dict[str, Any],
    *,
    draft_verified,
    sym_text: str,
    sym_iter: int,
    rewriter: LLMRewriter | None = None,
) -> tuple[str, int, list, list]:
    """Smooth already-corrected symbolic text; verify once (no duplicate symbolic loop)."""
    if rewriter is None:
        rewriter = create_rewriter_from_config(config)
    smoothed = rewriter.smooth_text(sym_text)
    final_verified = verifier.verify_summary(smoothed, None)
    outcomes = classify_outcomes(draft_verified, smoothed, final_verified)
    return smoothed, sym_iter, final_verified, outcomes


def _load_completed_ids(output_dir: Path) -> set[str]:
    done: set[str] = set()
    if not output_dir.exists():
        return done
    for p in output_dir.glob("*.json"):
        if p.name == "batch_summary.json":
            continue
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            done.add(row["case_id"])
        except Exception:
            continue
    return done


def load_condition_reports(output_dir: str | Path) -> list[dict[str, Any]]:
    """Load all per-case condition reports from disk (for Phase 4 reload after session restart)."""
    return _load_all_reports(Path(output_dir))


def run_conditions_batch(
    cases: list[CaseInput],
    verifier: Verifier,
    config: dict[str, Any],
    output_dir: str | Path,
    generations: dict[str, str],
    *,
    rewriter: LLMRewriter | None = None,
    load_rewriter: bool = True,
) -> list[dict[str, Any]]:
    """
    Resume-safe batch runner for all conditions.

    Skips case_ids already present in output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done = _load_completed_ids(output_dir)

    shared_rewriter = rewriter
    if load_rewriter and shared_rewriter is None and "llm_rewrite" in config.get("conditions", []):
        shared_rewriter = create_rewriter_from_config(config)

    pending = [
        c for c in cases
        if c.case_id not in done and generations.get(c.case_id)
    ]
    daily_cap = verifier.rate_limiter.daily_cap

    stopped_early = False
    stop_reason = ""

    for case in tqdm(pending, desc="Phase3 conditions"):
        draft = generations[case.case_id]
        try:
            report = run_all_conditions(
                case, verifier, config, draft, rewriter=shared_rewriter
            )
        except (RuntimeError, RetryError) as exc:
            budget_msg = _budget_error_message(exc)
            if budget_msg is not None:
                stopped_early = True
                stop_reason = budget_msg
                break
            raise

        out_path = output_dir / f"{case.case_id}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        done.add(case.case_id)

        remaining = verifier.rate_limiter.remaining_daily
        print(
            f"  [{case.case_id}] HR draft={report['hr_draft']:.3f} "
            f"→ symbolic={report['conditions']['symbolic']['hr_final']:.3f} (paired) | "
            f"CL budget: {remaining}/{daily_cap} remaining"
        )

    all_reports = _load_all_reports(output_dir)
    summary = _batch_summary(all_reports, stopped_early, stop_reason, verifier)
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return all_reports


def _load_all_reports(output_dir: Path) -> list[dict[str, Any]]:
    reports = []
    for p in sorted(output_dir.glob("*.json")):
        if p.name == "batch_summary.json":
            continue
        reports.append(json.loads(p.read_text(encoding="utf-8")))
    return reports


def _batch_summary(
    reports: list[dict[str, Any]],
    stopped_early: bool,
    stop_reason: str,
    verifier: Verifier,
) -> dict[str, Any]:
    rl = verifier.rate_limiter
    return {
        "n_cases": len(reports),
        "mean_hr_draft": sum(r["hr_draft"] for r in reports) / max(len(reports), 1),
        "mean_hr_symbolic": sum(
            r["conditions"]["symbolic"]["hr_final"] for r in reports
        ) / max(len(reports), 1),
        "mean_hr_llm_rewrite": sum(
            r["conditions"]["llm_rewrite"]["hr_final"] for r in reports
        ) / max(len(reports), 1),
        "cache_hits": verifier.cache_hits,
        "cache_misses": verifier.cache_misses,
        "api_calls_used": rl.day_count,
        "remaining_daily": rl.remaining_daily,
        "daily_cap": rl.daily_cap,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "cases": [
            {
                "case_id": r["case_id"],
                "hr_draft": r["hr_draft"],
                "hr_symbolic": r["conditions"]["symbolic"]["hr_final"],
                "hr_llm_rewrite": r["conditions"]["llm_rewrite"]["hr_final"],
            }
            for r in reports
        ],
    }
