#!/usr/bin/env python3
"""Build thesis_pipeline_story.pptx (EDA → pipeline → Canvas → Streamlit)."""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "demo" / "demo_bundle.json"
FIG = ROOT / "reports" / "presentation" / "figures"
OUT = ROOT / "reports" / "presentation" / "thesis_pipeline_story.pptx"

NAVY = RGBColor(0x0F, 0x27, 0x44)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
SLATE = RGBColor(0x33, 0x41, 0x55)
MUTED = RGBColor(0x64, 0x74, 0x8B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
AMBER = RGBColor(0xB4, 0x53, 0x09)
ROSE = RGBColor(0xBE, 0x12, 0x3C)


def _set_run(run, *, size=18, bold=False, color=NAVY, name="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name


def _add_textbox(slide, left, top, width, height, text, *, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _set_run(run, size=size, bold=bold, color=color)
    return box


def _add_bullets(slide, left, top, width, height, items, *, size=16, color=SLATE):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = f"•  {item}"
        _set_run(run, size=size, color=color)
    return box


def _add_notes(slide, text: str) -> None:
    notes = slide.notes_slide
    notes.notes_text_frame.text = text


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


def _header_bar(slide, title: str, subtitle: str | None = None) -> None:
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0), Inches(13.333), Inches(0.95),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY
    shape.line.fill.background()
    _add_textbox(slide, Inches(0.4), Inches(0.18), Inches(12), Inches(0.45), title,
                 size=26, bold=True, color=WHITE)
    if subtitle:
        _add_textbox(slide, Inches(0.4), Inches(0.55), Inches(12), Inches(0.35), subtitle,
                     size=12, color=RGBColor(0xCB, 0xD5, 0xE1))


def _footer(slide, page: str) -> None:
    _add_textbox(slide, Inches(0.4), Inches(7.15), Inches(10), Inches(0.3),
                 "Citation-faithful legal summarization · PRELIMINARY n=10",
                 size=10, color=MUTED)
    _add_textbox(slide, Inches(11.5), Inches(7.15), Inches(1.5), Inches(0.3), page,
                 size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def _picture(slide, path: Path, left, top, width=None, height=None):
    if not path.exists():
        _add_textbox(slide, left, top, Inches(4), Inches(0.4), f"[missing {path.name}]", size=12, color=ROSE)
        return
    kwargs = {}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    slide.shapes.add_picture(str(path), left, top, **kwargs)


def build() -> Path:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    rq = bundle["aggregate"]["rq"]
    m = bundle["aggregate"]["metrics_by_condition"]
    draft_hr = m["baseline"]["hr_paired_micro"] * 100
    sym_hr = m["symbolic"]["hr_paired_micro"] * 100
    rel = rq["RQ2"]["reduction"]["relative_reduction_pct"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    n = 0

    # --- 1 Title ---
    n += 1
    s = _blank(prs)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    _add_textbox(s, Inches(0.8), Inches(2.0), Inches(11.5), Inches(1),
                 "Citation-Faithful Legal Summarization", size=36, bold=True, color=WHITE)
    _add_textbox(s, Inches(0.8), Inches(3.0), Inches(11.5), Inches(0.6),
                 "From Multi-LexSum EDA → dual-system pipeline → Canvas results → live demo",
                 size=18, color=RGBColor(0x99, 0xF6, 0xE4))
    _add_textbox(s, Inches(0.8), Inches(4.2), Inches(11), Inches(0.8),
                 f"PRELIMINARY · n=10 citation-rich validation · D→L only\n"
                 f"Paired HR  {draft_hr:.1f}%  →  {sym_hr:.1f}%   ({rel:.1f}% relative reduction)",
                 size=16, color=RGBColor(0xCB, 0xD5, 0xE1))
    _add_textbox(s, Inches(0.8), Inches(6.4), Inches(11), Inches(0.4),
                 "Professor progress demo · story arc ending in Streamlit",
                 size=12, color=MUTED)
    _add_notes(s, "Open with the problem: neural summaries invent citations. Promise: show EDA → architecture → metrics → live fix.")

    # --- 2 Agenda ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Story arc", "What we will walk through")
    _add_bullets(s, Inches(0.8), Inches(1.4), Inches(11), Inches(5), [
        "EDA — why Multi-LexSum, why D→L, what truncation & citations taught us",
        "Dual-system idea — neural draft + symbolic verify/correct",
        "Architecture & key code — Modules A/B/C, re-anchor, paired HR",
        "Canvas results — RQ1–RQ3 charts from the interactive demo",
        "Hard case — SD-KS-0001 (Brown / bad LEXIS → re-anchor, not Hydraulic)",
        "Live Streamlit — one-case on-demand verification",
        "Limits & next steps — clean full run / dashboards",
    ], size=18)
    _footer(s, f"{n}")
    _add_notes(s, "Keep this as a map. Emphasize we end with a live demo, not just slides.")

    # --- 3 Problem ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "The problem", "Citation hallucinations in legal summarization")
    _add_bullets(s, Inches(0.8), Inches(1.4), Inches(6), Inches(5), [
        "Long-document summarizers can invent or misstate case cites",
        "A fluent wrong citation is worse than a missing one",
        "Need: generate useful summaries AND keep citations faithful",
        "Approach: dual-system — neural draft, symbolic verify against CourtListener, bounded correction",
    ], size=17)
    _add_textbox(s, Inches(7.2), Inches(1.6), Inches(5.3), Inches(4.5),
                 "Research questions\n\n"
                 "RQ1  What hallucination types?\n\n"
                 "RQ2  Does symbolic correction reduce HR?\n\n"
                 "RQ3  ≥30% HR cut without harming ROUGE-L?",
                 size=15, color=NAVY)
    _footer(s, f"{n}")
    _add_notes(s, "Hook: lawyers cannot trust invented cites. Dual-system is the thesis spine.")

    # --- 4 EDA corpus ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "EDA — corpus shape", "Multi-LexSum inventory")
    _picture(s, FIG / "eda_corpus_shape.png", Inches(0.5), Inches(1.2), width=Inches(12.2))
    _footer(s, f"{n}")
    _add_notes(s, "Train 3177 / val 454 / test 908. Long summaries are complete; short/tiny incomplete → descope to D→L.")

    # --- 5 EDA lengths ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "EDA — length & truncation", "Why Module A needs honesty about context")
    _picture(s, FIG / "eda_lengths_truncation.png", Inches(0.4), Inches(1.15), width=Inches(12.4))
    _footer(s, f"{n}")
    _add_notes(s, "At 4096 tokens most sources are truncated — known limitation of the draft generator setup.")

    # --- 6 EDA citations ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "EDA — citation density", "Why we evaluate on long (D→L) summaries")
    _picture(s, FIG / "eda_citation_density.png", Inches(1.2), Inches(1.2), width=Inches(10.5))
    _add_bullets(s, Inches(0.8), Inches(5.7), Inches(11.5), Inches(1.2), [
        "Long summaries carry more citation signal → prioritize D→L for RQ1–RQ3",
        "Heuristic cite hits motivated Module B (eyecite) + Module C (CourtListener)",
    ], size=14)
    _footer(s, f"{n}")
    _add_notes(s, "Bridge: EDA → design of verification pipeline.")

    # --- 7 Dual system ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Dual-system design", "Neural generate · symbolic verify/correct")
    _add_textbox(s, Inches(0.8), Inches(1.4), Inches(5.5), Inches(2.2),
                 "System 1 — Neural\n\nPRIMERA (Module A) drafts a long summary from truncated sources.",
                 size=16, color=NAVY)
    _add_textbox(s, Inches(7), Inches(1.4), Inches(5.5), Inches(2.2),
                 "System 2 — Symbolic\n\neyecite extract → CourtListener verify → bounded correction (redact / safe repair / re-anchor).",
                 size=16, color=NAVY)
    _add_bullets(s, Inches(0.8), Inches(4.0), Inches(11), Inches(2.5), [
        "Conditions: baseline · flagging · symbolic · llm_rewrite",
        "Primary metric: paired draft-citation hallucination rate (not surviving-only HR)",
        "Policy: never substitute a contradicting CL case name (e.g. Hydraulic into Brown)",
    ], size=16)
    _footer(s, f"{n}")
    _add_notes(s, "Say 'dual-system' clearly — this is the methodological claim.")

    # --- 8 Architecture ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Pipeline architecture", "End-to-end flow")
    _picture(s, FIG / "architecture_pipeline.png", Inches(0.4), Inches(1.2), width=Inches(12.4))
    _footer(s, f"{n}")
    _add_notes(s, "Walk left→right once. Point at Evaluation box for paired HR / RQ.")

    # --- 9 Code snippets ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Important code (snippets)", "What makes the pipeline honest")
    snippets = [
        ("Module B — full citation spans",
         "raw_text / full_span / pin_cite\n(not reporter-only substrings)"),
        ("Module C — unsafe identity",
         "party contradict CL cluster\n→ Ambiguous (not Mismatch replace)"),
        ("Correction — re-anchor",
         "find_reanchor_source()\nAmbiguous → verified same-case cite"),
        ("Eval — paired HR",
         "still_bad + garbled\nover draft citations"),
    ]
    for i, (title, body) in enumerate(snippets):
        x = 0.45 + (i % 4) * 3.2
        y = 1.4
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(3.0), Inches(4.6))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT
        box.line.color.rgb = TEAL
        _add_textbox(s, Inches(x + 0.15), Inches(y + 0.2), Inches(2.7), Inches(1),
                     title, size=13, bold=True, color=TEAL)
        _add_textbox(s, Inches(x + 0.15), Inches(y + 1.4), Inches(2.7), Inches(2.8),
                     body, size=14, color=NAVY)
    _footer(s, f"{n}")
    _add_notes(s, "Don't read code aloud; explain the four ideas. Re-anchor is the Brown/LEXIS fix.")

    # --- 10 Metrics definition ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "How we measure", "Paired HR + conditions")
    _add_bullets(s, Inches(0.8), Inches(1.4), Inches(11.5), Inches(5), [
        "Paired HR = (still_bad + garbled) / n_draft_citations — tracks each original cite after correction",
        "Labels: Verified · Mismatch · Ambiguous · NotFound · Malformed",
        "Safe Mismatch → expand with CL metadata; unsafe / Ambiguous → re-anchor if possible, else leave",
        "RQ3 gates: ≥30% relative HR reduction AND ROUGE-L within ±2 percentage points",
        f"This run: draft {draft_hr:.1f}% → symbolic {sym_hr:.1f}% ({rel:.1f}% relative) · ROUGE-L Δ ≈ 0 pp → PASS / PASS",
    ], size=17)
    _footer(s, f"{n}")
    _add_notes(s, "Stress paired HR vs naive empty-list HR=0 after bad redactions.")

    # --- 11 Canvas dashboard ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Canvas dashboard", "Interactive replay of all 10 cases")
    _picture(s, FIG / "canvas_dashboard_screenshot.png", Inches(0.8), Inches(1.15), width=Inches(11.6))
    _footer(s, f"{n}")
    _add_notes(s, "This is the Canvas UI stylized for slides. In the talk, open the real Cursor Canvas too.")

    # --- 12 HR chart ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Canvas charts — paired HR", "Draft vs symbolic vs LLM rewrite")
    _picture(s, FIG / "canvas_hr_by_condition.png", Inches(1.5), Inches(1.2), width=Inches(10))
    _footer(s, f"{n}")
    _add_notes(s, f"Headline: {draft_hr:.1f}% → {sym_hr:.1f}%. LLM rewrite tied with symbolic on this run.")

    # --- 13 RQ charts ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "RQ1–RQ3 results", "From Canvas research-question cards")
    _picture(s, FIG / "canvas_rq1_labels.png", Inches(0.3), Inches(1.2), width=Inches(5.8))
    _picture(s, FIG / "canvas_outcome_mix.png", Inches(6.5), Inches(1.3), width=Inches(6.3))
    _add_textbox(s, Inches(0.5), Inches(6.2), Inches(12), Inches(0.7),
                 f"RQ2: relative HR cut {rel:.1f}%  ·  RQ3: HR target PASS · ROUGE-L PASS  ·  McNemar p=0.25 (n=7, underpowered)",
                 size=14, color=NAVY)
    _footer(s, f"{n}")
    _add_notes(s, "RQ1 pie: Mismatch/Ambiguous dominate. Outcome mix: ~71% fixed, ~29% still_bad.")

    # --- 14 Case story ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Hard case — SD-KS-0001", "Why we do not trust CL blindly")
    _picture(s, FIG / "case_story_sd_ks.png", Inches(0.5), Inches(1.15), width=Inches(12.2))
    _footer(s, f"{n}")
    _add_notes(s, "Tell the Hydraulic story briefly, then re-anchor to verified 98 F. Supp. 797. Bridge to Streamlit.")

    # --- 15 Limitations ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Limitations (honest)", "What the Canvas also says")
    lims = rq.get("limitations") or []
    if isinstance(lims, str):
        lims = [lims]
    _add_bullets(s, Inches(0.8), Inches(1.3), Inches(11.5), Inches(5.5), lims[:8], size=15)
    _footer(s, f"{n}")
    _add_notes(s, "Be proud of honesty — small n, truncation, McNemar underpowered, BERTScore off.")

    # --- 16 Streamlit live ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Live demo — Streamlit", "On-demand extract → verify → correct")
    _add_textbox(s, Inches(0.8), Inches(1.4), Inches(11.5), Inches(0.5),
                 "Case SD-KS-0001 · real eyecite + CourtListener + symbolic correction",
                 size=16, bold=True, color=TEAL)
    _add_bullets(s, Inches(0.8), Inches(2.1), Inches(11.5), Inches(4), [
        "Activate venv and run:  streamlit run demo/live_showcase.py",
        "Point out Verified F. Supp. cite vs Ambiguous/bad LEXIS path",
        "Show re-anchor / no Hydraulic substitution",
        "Yellow = Ambiguous left when no same-case anchor (e.g. compare PB-CA-0016 on Canvas)",
        "Backup if API is down: Canvas case explorer already embeds saved outputs",
    ], size=17)
    _footer(s, f"{n}")
    _add_notes(
        s,
        "DEMO SCRIPT:\n"
        "1) Switch to Streamlit (already running if possible).\n"
        "2) Click run for SD-KS-0001.\n"
        "3) Narrate labels and final text.\n"
        "4) Return to slides for close.",
    )

    # --- 17 Close ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Takeaways & next", "Where we are")
    _add_bullets(s, Inches(0.8), Inches(1.4), Inches(11.5), Inches(5), [
        f"Dual-system works on this pilot: paired HR {draft_hr:.1f}% → {sym_hr:.1f}% ({rel:.1f}% relative), ROUGE preserved",
        "EDA justified D→L focus and exposed truncation / citation structure",
        "Correction policy is conservative: re-anchor when safe, leave Ambiguous otherwise",
        "Next: clean Colab full run → refresh Canvas/dashboards with final numbers",
        "Interactive artifacts: Cursor Canvas (10-case replay) + Streamlit (live one-case)",
    ], size=17)
    _footer(s, f"{n}")
    _add_notes(s, "End on next steps. Invite questions. Offer to open Canvas again if asked.")

    # --- 18 Backup Q&A ---
    n += 1
    s = _blank(prs)
    _header_bar(s, "Backup — commands", "If asked how to reproduce")
    _add_textbox(
        s, Inches(0.8), Inches(1.4), Inches(11.5), Inches(5),
        "cd code && source .venv/bin/activate\n\n"
        "# Canvas data + UI\n"
        "python scripts/bundle_demo_data.py --results-dir ../full_pipeline_results "
        "--output demo/demo_bundle.json\n"
        "python scripts/build_demo_canvas.py --bundle demo/demo_bundle.json\n\n"
        "# Live demo\n"
        "streamlit run demo/live_showcase.py\n\n"
        "# Rebuild this deck\n"
        "python scripts/export_presentation_figures.py\n"
        "python scripts/build_thesis_story_pptx.py",
        size=14, color=NAVY,
    )
    _footer(s, f"{n}")
    _add_notes(s, "Only show if asked about reproducibility.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Wrote {OUT} ({n} slides)")
    return OUT


if __name__ == "__main__":
    build()
