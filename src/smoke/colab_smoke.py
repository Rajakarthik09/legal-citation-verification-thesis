"""Colab smoke tests — Phase 0 gate before experiments."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Any, Literal


_OBERGEFELL_SMOKE_TEXT = (
    "Obergefell v. Hodges (576 U.S. 644) established the right to marriage"
)


@dataclass
class SmokeResult:
    name: str
    passed: bool
    message: str = ""


@dataclass
class SmokeReport:
    results: list[SmokeResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def add(self, name: str, passed: bool, message: str = "") -> None:
        self.results.append(SmokeResult(name, passed, message))

    def print_summary(self) -> None:
        print("\n=== Smoke Test Summary ===")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            msg = f" — {r.message}" if r.message else ""
            print(f"  [{status}] {r.name}{msg}")
        print(f"\nOverall: {'ALL PASSED' if self.all_passed else 'FAILURES DETECTED'}")


def _free_cuda() -> None:
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def smoke_environment(*, require_cuda: bool = False) -> SmokeResult:
    """Check Python packages, numpy stack, datasets version, optional CUDA."""
    try:
        import numpy as np
        import pyarrow  # noqa: F401
        import pandas  # noqa: F401
        import datasets
        import torch

        # Colab force-reinstall can leave numpy 2.x + old pyarrow → _center ImportError
        if require_cuda and np.__version__.startswith("2."):
            return SmokeResult(
                "environment",
                False,
                f"numpy 2.x breaks Colab stack (got {np.__version__}); re-run install cell",
            )

        ds_ok = datasets.__version__.startswith(("2.", "3."))
        cuda = torch.cuda.is_available()
        if require_cuda and not cuda:
            return SmokeResult("environment", False, "CUDA required but not available")
        msg = f"numpy={np.__version__}, datasets={datasets.__version__}, cuda={cuda}"
        if not ds_ok:
            return SmokeResult("environment", False, f"datasets must be <4, got {datasets.__version__}")
        return SmokeResult("environment", True, msg)
    except ImportError as exc:
        hint = " Re-run install cell (numpy<2, pyarrow<18)." if require_cuda else ""
        return SmokeResult("environment", False, f"{exc}.{hint}")
    except Exception as exc:
        return SmokeResult("environment", False, str(exc))


def smoke_secrets() -> SmokeResult:
    hf = os.environ.get("HF_TOKEN", "")
    cl = os.environ.get("COURTLISTENER_API_TOKEN", "")
    if not hf or hf.startswith("your-"):
        return SmokeResult("secrets", False, "HF_TOKEN not set")
    if not cl or cl.startswith("your-"):
        return SmokeResult("secrets", False, "COURTLISTENER_API_TOKEN not set")
    return SmokeResult("secrets", True, "HF_TOKEN and COURTLISTENER_API_TOKEN present")


def smoke_hf_token_only() -> SmokeResult:
    """Generation-only environments (e.g. a RunPod pod) never hold
    COURTLISTENER_API_TOKEN — Module C runs locally instead — so don't gate on it."""
    hf = os.environ.get("HF_TOKEN", "")
    if not hf or hf.startswith("your-"):
        return SmokeResult("secrets", False, "HF_TOKEN not set")
    return SmokeResult("secrets", True, "HF_TOKEN present")


def smoke_multilexsum() -> SmokeResult:
    try:
        from data.load_multilexsum import get_split

        split = get_split("validation")
        if len(split) == 0:
            return SmokeResult("multilexsum", False, "validation split empty")
        row = split[0]
        required = {"sources", "summary/long", "case_metadata", "id"}
        missing = required - set(row.keys())
        if missing:
            return SmokeResult("multilexsum", False, f"missing keys {missing}")
        return SmokeResult("multilexsum", True, f"{len(split)} validation cases")
    except Exception as exc:
        return SmokeResult("multilexsum", False, str(exc))


def smoke_courtlistener() -> SmokeResult:
    import requests

    token = os.environ.get("COURTLISTENER_API_TOKEN", "")
    if not token:
        return SmokeResult("courtlistener", False, "token missing")
    text = _OBERGEFELL_SMOKE_TEXT
    try:
        resp = requests.post(
            "https://www.courtlistener.com/api/rest/v4/citation-lookup/",
            headers={"Authorization": f"Token {token}"},
            data={"text": text},
            timeout=30,
        )
        if resp.status_code != 200:
            return SmokeResult("courtlistener", False, f"HTTP {resp.status_code}")
        return SmokeResult("courtlistener", True, "Obergefell lookup OK")
    except Exception as exc:
        return SmokeResult("courtlistener", False, str(exc))


def smoke_module_b() -> SmokeResult:
    try:
        from module_b_extraction.extract_citations import extract_citations

        text = (
            "Obergefell v. Hodges, 576 U.S. 644 (2015) and "
            "Fake v. State, 123 F.3d 456 (2099)."
        )
        cites = extract_citations(text)
        if len(cites) < 2:
            return SmokeResult("module_b", False, f"expected 2 citations, got {len(cites)}")
        return SmokeResult("module_b", True, f"extracted {len(cites)} citations")
    except Exception as exc:
        return SmokeResult("module_b", False, str(exc))


def smoke_module_c(tmp_cache: str | None = None) -> SmokeResult:
    try:
        from module_c_verifier.verify import Verifier

        config = {
            "cache_path": tmp_cache or ":memory:",
            "rate_limit": {"valid_citations_per_minute": 60, "daily_cap": 125},
            "fuzzy_match_threshold": 85,
        }
        if tmp_cache is None:
            import tempfile
            config["cache_path"] = tempfile.mktemp(suffix=".sqlite")

        verifier = Verifier(config)
        text = _OBERGEFELL_SMOKE_TEXT
        results = verifier.verify_summary(text, None)
        verifier.close()
        labels = {r.label for r in results}
        if "Verified" not in labels:
            return SmokeResult("module_c", False, f"missing Verified, got {labels}")
        return SmokeResult("module_c", True, f"labels={labels}")
    except Exception as exc:
        return SmokeResult("module_c", False, str(exc))


def smoke_correction() -> SmokeResult:
    try:
        from unittest.mock import MagicMock

        from correction_loop.corrector import correct
        from module_b_extraction.extract_citations import extract_citations
        from utils.schemas import ExtractedCitation, VerificationResult

        text = "Cited Fake v. State, 123 F.3d 456 (2099) here."
        cites = extract_citations(text)
        if not cites:
            span = (6, 40)
            raw = "123 F.3d 456 (2099)"
        else:
            fake = cites[0]
            span = fake.span
            raw = fake.raw_text

        call_count = 0

        def mock_verify(t, extracted=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                cite = ExtractedCitation(
                    raw_text=raw, normalized=raw, reporter=None, volume=None,
                    page=None, year=None, party_names=None, span=span,
                )
                return [VerificationResult(cite, "NotFound", 404, None, None, "not found")]
            return []

        verifier = MagicMock()
        final, n_iter, _, _ = correct(text, verifier, max_iterations=3, verify_fn=mock_verify)
        if "[unverifiable citation removed]" not in final:
            return SmokeResult("correction", False, "redaction marker missing")
        return SmokeResult("correction", True, f"iterations={n_iter}")
    except Exception as exc:
        return SmokeResult("correction", False, str(exc))


def smoke_module_a(
    *,
    mode: Literal["lightweight", "full", "skip"] = "lightweight",
    skip_if_no_cuda: bool = True,
) -> SmokeResult:
    """
    Module A smoke check.

    lightweight (default): load tokenizer + PRIMERA weights only, no generation.
    full: generate one short summary (slow).
    skip: no-op pass.
    """
    if mode == "skip":
        return SmokeResult("module_a", True, "skipped by config")

    try:
        import torch

        if skip_if_no_cuda and not torch.cuda.is_available():
            return SmokeResult("module_a", True, "skipped (no CUDA — run on Colab)")

        from module_a_summarizer.model import Summarizer

        summarizer = Summarizer(granularity="long", device="cuda", use_fp16=True)
        if mode == "lightweight":
            msg = f"weights loaded on {summarizer.device}"
            del summarizer
            _free_cuda()
            return SmokeResult("module_a", True, msg)

        from data.load_multilexsum import get_split
        from data.preprocessing import build_case_input, GRANULARITY_CHECKPOINTS
        from transformers import AutoTokenizer

        split = get_split("validation")
        tokenizer = AutoTokenizer.from_pretrained(GRANULARITY_CHECKPOINTS["long"])
        case = None
        for i in range(min(5, len(split))):
            case = build_case_input(split[i], "long", tokenizer, 4096)
            if case is not None:
                break
        if case is None:
            return SmokeResult("module_a", False, "no valid case found")

        out = summarizer.generate(case.sources_text, max_new_tokens=64, num_beams=1)
        del summarizer
        _free_cuda()
        if not out.strip():
            return SmokeResult("module_a", False, "empty generation")
        return SmokeResult("module_a", True, f"generated {len(out.split())} words")
    except Exception as exc:
        return SmokeResult("module_a", False, str(exc))


def smoke_llm_rewrite(*, mode: Literal["load_only", "mock"] = "load_only") -> SmokeResult:
    """Load flan-t5-base and run one rewrite (Colab) or mock locally."""
    try:
        if mode == "mock":
            from unittest.mock import patch

            from correction_loop.llm_rewrite import LLMRewriter

            rewriter = LLMRewriter(model_name="google/flan-t5-base")
            with patch.object(
                rewriter, "rewrite_sentence",
                return_value="The court removed an invalid citation.",
            ):
                out = rewriter.smooth_text("Bad cite [unverifiable citation removed] here.")
            if not out:
                return SmokeResult("llm_rewrite", False, "empty output")
            return SmokeResult("llm_rewrite", True, "mock rewrite OK")

        from correction_loop.llm_rewrite import LLMRewriter

        rewriter = LLMRewriter(model_name="google/flan-t5-base")
        out = rewriter.rewrite_sentence(
            "The court cited [unverifiable citation removed] in its analysis."
        )
        rewriter.unload()
        _free_cuda()
        if not out.strip():
            return SmokeResult("llm_rewrite", False, "empty rewrite")
        return SmokeResult("llm_rewrite", True, f"flan-t5 loaded, rewrite len={len(out)}")
    except Exception as exc:
        return SmokeResult("llm_rewrite", False, str(exc))


def smoke_evaluation() -> SmokeResult:
    try:
        from evaluation.metrics import compute_rouge, micro_hr

        rouge = compute_rouge("the cat sat", "the cat sat on the mat")
        if rouge["rouge1"] <= 0:
            return SmokeResult("evaluation", False, "rouge failed")
        cites = [[{"label": "Verified", "citation": _minimal_citation()}]]
        hr = micro_hr(cites)
        if hr != 0.0:
            return SmokeResult("evaluation", False, f"expected hr=0, got {hr}")
        return SmokeResult("evaluation", True, f"rouge1={rouge['rouge1']:.2f}")
    except Exception as exc:
        return SmokeResult("evaluation", False, str(exc))


def _minimal_citation() -> dict[str, Any]:
    return {
        "raw_text": "576 U.S. 644",
        "normalized": "576 U.S. 644",
        "reporter": "U.S.",
        "volume": "576",
        "page": "644",
        "year": 2015,
        "party_names": None,
        "span": [0, 12],
    }


def smoke_run_all(
    *,
    require_cuda: bool = False,
    module_a_mode: Literal["lightweight", "full", "skip"] = "lightweight",
    llm_rewrite_mode: Literal["load_only", "mock"] = "load_only",
    include_live_api: bool = True,
) -> SmokeReport:
    """
    Run all smoke tests. Returns report.

    On Colab: module_a_mode='lightweight', llm_rewrite_mode='load_only'.
    Locally without CUDA: use module_a_mode='skip', llm_rewrite_mode='mock'.
    """
    import torch

    report = SmokeReport()

    report.results.append(smoke_environment(require_cuda=require_cuda))
    report.results.append(smoke_secrets())

    for fn in (smoke_multilexsum, smoke_module_b, smoke_correction, smoke_evaluation):
        report.results.append(fn())

    if include_live_api:
        report.results.append(smoke_courtlistener())
        report.results.append(smoke_module_c())

    if llm_rewrite_mode == "load_only" and not torch.cuda.is_available():
        report.results.append(smoke_llm_rewrite(mode="mock"))
    else:
        report.results.append(smoke_llm_rewrite(mode=llm_rewrite_mode))

    if module_a_mode == "skip":
        report.results.append(SmokeResult("module_a", True, "skipped by config"))
    elif not torch.cuda.is_available():
        report.results.append(SmokeResult("module_a", True, "skipped (no CUDA)"))
    else:
        report.results.append(smoke_module_a(mode=module_a_mode, skip_if_no_cuda=False))

    report.print_summary()
    return report


def assert_smoke_gate(report: SmokeReport) -> None:
    """Raise RuntimeError if any smoke test failed."""
    failures = [r for r in report.results if not r.passed]
    if failures:
        names = ", ".join(r.name for r in failures)
        raise RuntimeError(f"Smoke gate failed: {names}")
