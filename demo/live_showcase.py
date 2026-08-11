#!/usr/bin/env python3
"""
Live (on-demand) dual-system verification/correction demo for one case.

Companion to the Cursor Canvas:
- The Canvas replays all 10 cases from saved outputs.
- This Streamlit app runs real eyecite extraction, CourtListener verification,
  and symbolic correction for SD-KS-0001 on demand.
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any

CASE_ID = "SD-KS-0001"
CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LABEL_COLORS = {
    "Verified": "#16a34a",
    "Mismatch": "#ea580c",
    "Ambiguous": "#ca8a04",
    "NotFound": "#dc2626",
    "Malformed": "#dc2626",
}


def _load_case_inputs() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    case_path = root / "full_pipeline_results" / "conditions" / f"{CASE_ID}.json"
    if not case_path.exists():
        raise FileNotFoundError(f"Missing case report: {case_path}")
    return json.loads(case_path.read_text(encoding="utf-8"))


def _display_citations(st: Any, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError:
        st.dataframe(
            [
                {
                    "raw_text": (r.get("citation") or {}).get("raw_text"),
                    "label": r.get("label"),
                    "matched_case_name": r.get("matched_case_name"),
                    "explanation": r.get("explanation"),
                }
                for r in rows
            ],
            use_container_width=True,
            hide_index=True,
        )
        return

    df = pd.DataFrame(
        [
            {
                "raw_text": (r.get("citation") or {}).get("raw_text"),
                "label": r.get("label"),
                "cl_status": r.get("cl_status"),
                "matched_case_name": r.get("matched_case_name"),
                "explanation": r.get("explanation"),
            }
            for r in rows
        ]
    )

    def _color_label(val: str) -> str:
        color = LABEL_COLORS.get(str(val), "#6b7280")
        return f"color: {color}; font-weight: 600"

    if not df.empty and "label" in df.columns:
        st.dataframe(
            df.style.map(_color_label, subset=["label"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _highlight_citations_html(text: str, verified_rows: list[Any]) -> str:
    """Highlight citation spans; Ambiguous gets a yellow background."""
    spans: list[tuple[int, int, str]] = []
    for v in verified_rows:
        cite = getattr(v, "citation", None)
        if cite is None and isinstance(v, dict):
            cite = v.get("citation") or {}
            label = v.get("label") or ""
            span = cite.get("span") or (0, 0)
        else:
            label = getattr(v, "label", "") or ""
            span = getattr(cite, "span", (0, 0))
        if not span or span[1] <= span[0]:
            continue
        spans.append((int(span[0]), int(span[1]), str(label)))
    spans.sort(key=lambda x: x[0])

    parts: list[str] = []
    cursor = 0
    for start, end, label in spans:
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        if start < cursor:
            continue
        parts.append(html.escape(text[cursor:start]))
        color = LABEL_COLORS.get(label, "#6b7280")
        chunk = html.escape(text[start:end])
        if label == "Ambiguous":
            parts.append(
                f'<mark style="background:#fef08a;color:{color};padding:0 2px;'
                f'border-bottom:2px solid {color}" title="{label}">{chunk}</mark>'
            )
        else:
            parts.append(
                f'<span style="color:{color};text-decoration:underline;'
                f'text-underline-offset:3px" title="{label}">{chunk}</span>'
            )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return (
        '<div style="white-space:pre-wrap;font-family:ui-monospace,monospace;'
        f'font-size:13px;line-height:1.55">{"".join(parts)}</div>'
    )


def main() -> None:
    try:
        import streamlit as st
    except Exception as e:
        raise RuntimeError(
            "Streamlit is not installed. Install it with `pip install streamlit`."
        ) from e

    st.set_page_config(page_title="Live Verification Showcase", layout="wide")
    st.title("Live dual-system demo")
    st.caption(f"Case {CASE_ID} · extract → CourtListener verify → symbolic correct")

    st.info(
        "**Canvas** = replay all 10 saved cases with RQ storytelling.  \n"
        "**This app** = live API path on one case (SD-KS-0001).  \n"
        "**Policy:** Ambiguous cites stay highlighted unless a **verified same-case cite** "
        "in the summary can re-anchor them. Never substitute a contradicting CL case name."
    )

    with st.expander("Why this case?", expanded=False):
        st.markdown(
            "- One **Verified** reporter cite (`98 F. Supp. 797`, *Brown*).\n"
            "- One hard **LEXIS** cite that CL maps to the wrong cluster — re-anchored to the "
            "verified Brown cite when possible.\n"
            "- Companion to **PB-CA-0016** on the Canvas (Ambiguous left when no anchor)."
        )

    st.sidebar.header("Secrets")
    st.sidebar.write(
        {
            "HF_TOKEN_present": bool(os.environ.get("HF_TOKEN")),
            "COURTLISTENER_API_TOKEN_present": bool(
                os.environ.get("COURTLISTENER_API_TOKEN")
            ),
        }
    )
    if not os.environ.get("COURTLISTENER_API_TOKEN"):
        st.warning("COURTLISTENER_API_TOKEN is missing. Set it and reload.")
        st.stop()

    from dataclasses import asdict
    from correction_loop.corrector import correct
    from module_b_extraction.extract_citations import extract_citations
    from module_c_verifier.verify import Verifier, hallucination_rate
    from evaluation.paired_hr import paired_hr_from_outcomes

    st.sidebar.header("Symbolic correction")
    max_iters = st.sidebar.slider("max_correction_iterations", 1, 3, 2)

    st.sidebar.header("CourtListener rate limiting")
    daily_cap = st.sidebar.number_input(
        "daily_cap", value=125, min_value=1, max_value=500
    )
    per_minute = st.sidebar.number_input(
        "per_minute",
        value=60,
        min_value=1,
        max_value=200,
        help="Lower this if you hit transient API throttling.",
    )

    inputs = _load_case_inputs()
    draft_text: str = inputs["draft_summary"]

    st.subheader("Input: PRIMERA draft")
    st.text_area("Draft summary", value=draft_text[:4000], height=220)

    st.divider()

    st.session_state.setdefault("extracted", None)
    st.session_state.setdefault("verified", None)
    st.session_state.setdefault("corrected_text", None)
    st.session_state.setdefault("corrected_iterations", None)
    st.session_state.setdefault("corrected_verified", None)
    st.session_state.setdefault("correction_outcomes", None)

    cache_path = str(Path(__file__).resolve().parents[0] / "cl_demo_cache.sqlite")
    st.sidebar.write(f"Cache path: {cache_path}")

    verifier = Verifier(
        {
            "cache_path": cache_path,
            "rate_limit": {
                "valid_citations_per_minute": int(per_minute),
                "daily_cap": int(daily_cap),
            },
            "fuzzy_match_threshold": 85,
        }
    )

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("1) Extract citations (eyecite)", use_container_width=True):
            st.session_state.extracted = extract_citations(draft_text)
    with colB:
        if st.button("2) Verify (CourtListener)", use_container_width=True):
            extracted = st.session_state.get("extracted") or extract_citations(draft_text)
            st.session_state.verified = verifier.verify_summary(
                draft_text, extracted=extracted
            )
    with colC:
        if st.button("3) Symbolic correct", use_container_width=True):
            corrected_text, iters_used, verified_final, outcomes = correct(
                draft_text,
                verifier=verifier,
                max_iterations=int(max_iters),
            )
            st.session_state.corrected_text = corrected_text
            st.session_state.corrected_iterations = iters_used
            st.session_state.corrected_verified = verified_final
            st.session_state.correction_outcomes = outcomes

    st.subheader("Step 1–2: Extract + Verify")
    extracted = st.session_state.get("extracted")
    verified = st.session_state.get("verified")
    if extracted is not None:
        st.write(f"Extracted citations: **{len(extracted)}**")
    if verified is not None:
        st.write(
            f"Verified: **{len(verified)}** · draft micro HR = "
            f"**{hallucination_rate(verified):.1%}**"
        )
        _display_citations(st, [asdict(v) for v in verified])
        st.markdown("**Draft with citation highlights**")
        st.markdown(_highlight_citations_html(draft_text, verified), unsafe_allow_html=True)

    st.divider()
    st.subheader("Step 3: Symbolic correction")
    corrected_text = st.session_state.get("corrected_text")
    outcomes = st.session_state.get("correction_outcomes")
    if corrected_text is not None:
        st.write(f"Iterations used: **{st.session_state.get('corrected_iterations')}**")
        if outcomes:
            paired = paired_hr_from_outcomes(outcomes)
            st.success(
                f"Paired HR after correction: **{paired:.1%}** "
                f"(Ambiguous left without a same-case anchor counts as still_bad)"
            )
            st.dataframe(
                [
                    {
                        "citation_id": o.citation_id,
                        "draft_label": o.draft_label,
                        "final_status": o.final_status,
                        "draft_span": o.draft_span,
                    }
                    for o in outcomes
                ],
                use_container_width=True,
                hide_index=True,
            )
        corrected_verified = st.session_state.get("corrected_verified") or []
        st.markdown("**Corrected text with highlights**")
        st.markdown(
            _highlight_citations_html(corrected_text, corrected_verified),
            unsafe_allow_html=True,
        )
        with st.expander("Raw corrected text"):
            st.text_area("Corrected summary", value=corrected_text, height=280)
        if corrected_verified is not None:
            st.write(f"Citations after correction: **{len(corrected_verified)}**")
            _display_citations(st, [asdict(v) for v in corrected_verified])


if __name__ == "__main__":
    main()
