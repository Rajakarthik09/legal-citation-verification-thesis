"""Statistical testing for Phase 5."""

from __future__ import annotations

import random
from typing import Any

from module_c_verifier.verify import BAD_LABELS

SUCCESS_OUTCOMES = frozenset({"verified", "replaced_verified", "removed"})


def _citation_hallucinated(label: str) -> bool:
    return label in BAD_LABELS


def _outcome_success(final_status: str) -> bool:
    return final_status in SUCCESS_OUTCOMES


def mcnemar_citation_test(
    draft_citations: list[list[dict[str, Any]]],
    correction_outcomes: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    McNemar-style test on paired per-draft-citation outcomes.

    Pairs by citation_id across draft labels and correction outcomes.
  All draft citations are included.
    """
    b = 0  # draft hallucinated, final success (fixed or removed)
    c = 0  # draft ok, final still bad/garbled
    n_paired = 0

    for draft_list, outcome_list in zip(draft_citations, correction_outcomes):
        outcome_by_id = {o["citation_id"]: o for o in outcome_list}
        for draft in draft_list:
            cite = draft.get("citation") or {}
            cid = cite.get("citation_id")
            if not cid:
                continue
            outcome = outcome_by_id.get(cid)
            if outcome is None:
                continue
            d_hall = _citation_hallucinated(draft.get("label", ""))
            f_success = _outcome_success(outcome.get("final_status", ""))
            n_paired += 1
            if d_hall and f_success:
                b += 1
            elif not d_hall and not f_success:
                c += 1

    if b + c == 0:
        p_value = 1.0
    else:
        try:
            from scipy import stats

            result = stats.binomtest(b, b + c, 0.5)
            p_value = float(result.pvalue)
        except Exception:
            p_value = None

    return {
        "n_paired_citations": n_paired,
        "draft_hallucinated_final_success": b,
        "draft_ok_final_bad": c,
        "draft_only_hallucinated": b,
        "final_only_hallucinated": c,
        "p_value": p_value,
        "interpretation": (
            "Significant reduction in hallucination" if p_value is not None and p_value < 0.05 and b > c
            else "No significant change or insufficient discordant pairs"
        ),
    }


def bootstrap_ci(
    baseline_values: list[float],
    treatment_values: list[float],
    *,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap CI for mean delta (treatment - baseline)."""
    if len(baseline_values) != len(treatment_values) or not baseline_values:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    deltas = [t - b for b, t in zip(baseline_values, treatment_values)]
    mean_delta = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    n = len(deltas)
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap) - 1
    return {
        "mean_delta": mean_delta,
        "ci_low": boot_means[lo_idx],
        "ci_high": boot_means[hi_idx],
        "n": n,
        "ci_level": ci,
    }


def hr_reduction_stats(
    hr_draft_per_case: list[float],
    hr_final_per_case: list[float],
) -> dict[str, Any]:
    """Absolute and relative HR reduction (same metric for draft and final)."""
    if not hr_draft_per_case:
        return {"absolute_reduction": 0.0, "relative_reduction_pct": 0.0}
    abs_red = [
        d - f for d, f in zip(hr_draft_per_case, hr_final_per_case)
    ]
    mean_abs = sum(abs_red) / len(abs_red)
    mean_draft = sum(hr_draft_per_case) / len(hr_draft_per_case)
    rel_pct = (mean_abs / mean_draft * 100) if mean_draft > 0 else 0.0
    return {
        "absolute_reduction": mean_abs,
        "relative_reduction_pct": rel_pct,
        "mean_hr_draft": mean_draft,
        "mean_hr_final": sum(hr_final_per_case) / len(hr_final_per_case),
    }
