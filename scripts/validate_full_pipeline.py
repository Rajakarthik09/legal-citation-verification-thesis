#!/usr/bin/env python3
"""Dry-run validation for full pipeline Phases 4–5 (no GPU/API required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _synthetic_report(case_id: str, hr_draft: float) -> dict:
    cite_bad = {
        "label": "NotFound",
        "citation": {
            "raw_text": "123 F.3d 456 (2099)",
            "normalized": "123 F.3d 456",
            "reporter": "F.3d",
            "volume": "123",
            "page": "456",
            "year": 2099,
            "party_names": None,
            "span": [10, 30],
        },
        "cl_status": 404,
        "matched_case_name": None,
        "matched_date_filed": None,
        "explanation": "not found",
    }
    cite_ok = {**cite_bad, "label": "Verified", "cl_status": 200}
    draft = f"Case {case_id} cites Fake v. State, 123 F.3d 456 (2099)."
    final_sym = f"Case {case_id} cites [unverifiable citation removed]."
    final_llm = f"Case {case_id} had an unverifiable citation removed."
    return {
        "case_id": case_id,
        "granularity": "long",
        "gold_summary": f"Gold summary for {case_id}.",
        "draft_summary": draft,
        "citations_draft": [cite_bad] if hr_draft > 0 else [cite_ok],
        "hr_draft": hr_draft,
        "conditions": {
            "baseline": {"summary": draft, "citations": [cite_bad], "hr": hr_draft},
            "flagging": {"summary": draft, "citations": [cite_bad], "hr": hr_draft},
            "symbolic": {
                "final_summary": final_sym,
                "citations_final": [],
                "hr_final": 0.0,
                "correction_iterations": 1,
            },
            "llm_rewrite": {
                "final_summary": final_llm,
                "citations_final": [],
                "hr_final": 0.0,
                "correction_iterations": 1,
            },
        },
    }


def main() -> int:
    from evaluation.aggregate import aggregate_reports, metrics_to_dataframe_rows
    from evaluation.audit_export import export_audit_template
    from evaluation.error_analysis import export_error_analysis_csv
    from evaluation.rq_report import generate_rq_report, save_rq_report
    from evaluation.stats import bootstrap_ci, hr_reduction_stats, mcnemar_citation_test
    from smoke.colab_smoke import (
        smoke_correction,
        smoke_evaluation,
        smoke_llm_rewrite,
        smoke_module_b,
    )
    from utils.config import load_yaml

    print("=== Phase 4–5 dry-run validation ===")

    for fn in (smoke_module_b, smoke_correction, smoke_evaluation):
        r = fn()
        assert r.passed, f"{r.name} failed: {r.message}"
        print(f"  [PASS] {r.name}")

    r = smoke_llm_rewrite(mode="mock")
    assert r.passed, f"{r.name} failed: {r.message}"
    print(f"  [PASS] {r.name}")

    exp_cfg = load_yaml("experiment_validation50.yaml")
    assert exp_cfg["granularity"] == "long"

    prof_cfg = load_yaml("experiment_professor10.yaml")
    assert prof_cfg["max_cases"] == 10
    assert len(prof_cfg.get("case_ids", [])) == 10
    assert prof_cfg.get("case_selection") == "citation_rich"

    full_cfg = load_yaml("experiment_test100.yaml")
    assert full_cfg["split"] == "test"
    assert full_cfg["granularity"] == "long"
    assert full_cfg["max_cases"] == 100
    assert len(full_cfg.get("case_ids", [])) == 100
    assert len(set(full_cfg["case_ids"])) == 100, "duplicate case_ids in full-scale config"
    assert full_cfg.get("case_selection") == "citation_rich"
    assert full_cfg.get("run_bertscore") is True
    print(f"  [PASS] experiment_test100.yaml: {len(full_cfg['case_ids'])} unique case_ids, split={full_cfg['split']}")

    llm_cfg = load_yaml("experiment_test100_llm.yaml")
    llm_rag_cfg = load_yaml("experiment_test100_llm_rag.yaml")
    assert llm_cfg["case_ids"] == full_cfg["case_ids"], "LLM condition must use the same 100 case_ids as PRIMERA"
    assert llm_rag_cfg["case_ids"] == full_cfg["case_ids"], "LLM+RAG condition must use the same 100 case_ids as PRIMERA"
    assert llm_cfg["model_tag"] == "llm" and llm_cfg.get("use_retrieval") is False
    assert llm_rag_cfg["model_tag"] == "llm_rag" and llm_rag_cfg.get("use_retrieval") is True
    assert llm_cfg["generations_filename"] == "test_llm.jsonl"
    assert llm_rag_cfg["generations_filename"] == "test_llm_rag.jsonl"
    mod_llm_cfg = load_yaml("module_llm.yaml")
    assert mod_llm_cfg["max_input_tokens"] > 4096, "LLM input budget should exceed PRIMERA's fixed 4096 ceiling"
    print(f"  [PASS] experiment_test100_llm.yaml + experiment_test100_llm_rag.yaml: "
          f"same 100 case_ids as PRIMERA, distinct output paths, max_input_tokens={mod_llm_cfg['max_input_tokens']}")

    reports = [_synthetic_report(f"V{i:03d}", hr_draft=0.5 if i % 2 == 0 else 0.0) for i in range(5)]
    aggregated = aggregate_reports(reports, skip_bertscore=True)
    rows = metrics_to_dataframe_rows(aggregated)
    assert len(rows) >= 3

    hr_stats = hr_reduction_stats(
        [r["hr_draft"] for r in reports],
        [r["conditions"]["symbolic"]["hr_final"] for r in reports],
    )
    mcnemar = mcnemar_citation_test(
        [r["citations_draft"] for r in reports],
        [r["conditions"]["symbolic"]["citations_final"] for r in reports],
    )
    assert "relative_reduction_pct" in hr_stats

    out = ROOT / "outputs" / "dry_run_validation"
    out.mkdir(parents=True, exist_ok=True)
    rq = generate_rq_report(
        reports,
        aggregated,
        split_name="validation",
        n_cases=5,
        partial_run=False,
        presentation_label="dry-run",
        run_bertscore=False,
    )
    save_rq_report(rq, out)
    export_error_analysis_csv(reports, out / "error_analysis.csv", n_corrected=2, n_failed=2)

    pool = [{"id": f"A{i}", "summary/long": f"Summary {i} citing 576 U.S. 644."} for i in range(300)]
    export_audit_template(pool, out / "audit_template.csv", n_cases=50, seed=42)

    nb_path = ROOT / "notebooks" / "thesis_full_pipeline_colab.ipynb"
    nb = json.loads(nb_path.read_text())
    assert len(nb["cells"]) >= 20
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"  Notebook: {len(code_cells)} code cells")

    runpod_nb_path = ROOT / "notebooks" / "runpod_generation.ipynb"
    runpod_nb = json.loads(runpod_nb_path.read_text())
    runpod_code_cells = [c for c in runpod_nb["cells"] if c["cell_type"] == "code"]
    assert len(runpod_code_cells) >= 7
    for c in runpod_code_cells:
        compile("".join(c["source"]), "<runpod_generation.ipynb cell>", "exec")
    print(f"  RunPod notebook: {len(runpod_code_cells)} code cells, all parse as valid Python")

    runpod_llm_nb_path = ROOT / "notebooks" / "runpod_llm_generation.ipynb"
    runpod_llm_nb = json.loads(runpod_llm_nb_path.read_text())
    runpod_llm_code_cells = [c for c in runpod_llm_nb["cells"] if c["cell_type"] == "code"]
    assert len(runpod_llm_code_cells) >= 9
    for c in runpod_llm_code_cells:
        compile("".join(c["source"]), "<runpod_llm_generation.ipynb cell>", "exec")
    print(f"  RunPod LLM notebook: {len(runpod_llm_code_cells)} code cells, all parse as valid Python")

    print("=== All dry-run checks passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
