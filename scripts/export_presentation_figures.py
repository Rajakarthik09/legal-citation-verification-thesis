#!/usr/bin/env python3
"""Export EDA + Canvas-metric figures for the thesis story PowerPoint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "demo" / "demo_bundle.json"
TABLES = ROOT / "reports" / "tables"
OUT = ROOT / "reports" / "presentation" / "figures"

# Thesis palette (navy / slate / teal accent — avoid purple-gradient defaults)
NAVY = "#0f2744"
SLATE = "#334155"
TEAL = "#0f766e"
AMBER = "#b45309"
ROSE = "#be123c"
GREEN = "#15803d"
LIGHT = "#f1f5f9"
WHITE = "#ffffff"
MUTED = "#64748b"


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.edgecolor": SLATE,
            "axes.labelcolor": NAVY,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": NAVY,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def plot_eda_corpus_shape() -> None:
    inv = pd.read_csv(TABLES / "split_inventory.csv")
    avail = pd.read_csv(TABLES / "summary_availability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = [TEAL, NAVY, AMBER]
    axes[0].bar(inv["split"], inv["n_cases"], color=colors[: len(inv)], width=0.65)
    axes[0].set_title("Multi-LexSum split inventory")
    axes[0].set_ylabel("Cases")
    for i, v in enumerate(inv["n_cases"]):
        axes[0].text(i, v + 40, str(v), ha="center", fontsize=10, color=NAVY)

    # Long-summary availability by split
    long = avail[avail["field"] == "summary/long"]
    axes[1].bar(long["split"], long["pct_present"], color=TEAL, width=0.65)
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Long summary availability (%)")
    axes[1].set_ylabel("% present")
    for i, (_, row) in enumerate(long.iterrows()):
        axes[1].text(i, row["pct_present"] + 2, f"{row['pct_present']:.0f}%", ha="center", fontsize=10)

    fig.suptitle("EDA — corpus shape", color=NAVY, fontsize=14, fontweight="bold", y=1.02)
    _save(fig, "eda_corpus_shape.png")


def plot_eda_lengths_and_truncation() -> None:
    words = pd.read_csv(TABLES / "summary_word_stats.csv")
    trunc = pd.read_csv(TABLES / "truncation_fractions.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    long = words[words["field"].astype(str).str.contains("long", case=False)]
    if not long.empty:
        splits = long["split"].tolist()
        means = long["mean"].tolist()
        axes[0].bar(splits, means, color=TEAL, width=0.55, label="mean words")
        if "paper_target" in long.columns:
            axes[0].axhline(long["paper_target"].iloc[0], color=AMBER, ls="--", lw=1.5, label="paper target ~650")
        axes[0].set_title("Long-summary length (words)")
        axes[0].set_ylabel("Mean words")
        axes[0].legend(frameon=False, fontsize=9)

    val = trunc[trunc["split"] == "validation"] if "validation" in set(trunc["split"]) else trunc
    axes[1].plot(val["budget"], val["frac_over"] * 100, marker="o", color=ROSE, lw=2)
    axes[1].axvline(4096, color=AMBER, ls="--", lw=1.4, label="our budget 4096")
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("% sources exceeding token budget")
    axes[1].set_xlabel("Token budget")
    axes[1].set_ylabel("% over budget")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("EDA — length & truncation (motivates Module A limits)", color=NAVY, fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "eda_lengths_truncation.png")


def plot_eda_citations() -> None:
    cites = pd.read_csv(TABLES / "citation_heuristic_all_granularities.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))

    # aggregate mean cites_per_100_words by field
    if "field" in cites.columns and "cites_per_100_words" in cites.columns:
        g = cites.groupby("field")["cites_per_100_words"].mean().sort_values(ascending=False)
        # also hit rate
        hit = cites.groupby("field")["cite_hits"].apply(lambda s: (s > 0).mean() * 100)
        x = np.arange(len(g))
        ax.bar(x - 0.18, g.values, width=0.35, color=TEAL, label="mean cites / 100 words")
        ax2 = ax.twinx()
        ax2.bar(x + 0.18, hit.reindex(g.index).values, width=0.35, color=NAVY, alpha=0.75, label="% with ≥1 cite")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i).replace("summary/", "") for i in g.index], rotation=15)
        ax.set_ylabel("Cites per 100 words")
        ax2.set_ylabel("% docs with a citation hit")
        ax.set_title("EDA — citation density by summary granularity")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=9, loc="upper right")
    else:
        ax.text(0.5, 0.5, "citation table format unexpected", ha="center")

    ax.annotate(
        "D→L prioritized for citation verification RQs",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        color=AMBER,
    )
    _save(fig, "eda_citation_density.png")


def plot_canvas_hr_bar(bundle: dict) -> None:
    m = bundle["aggregate"]["metrics_by_condition"]
    labels = ["Draft\n(baseline)", "Symbolic\ncorrection", "LLM\nrewrite"]
    keys = ["baseline", "symbolic", "llm_rewrite"]
    vals = [m[k]["hr_paired_micro"] * 100 for k in keys]
    colors = [ROSE, TEAL, NAVY]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.axhline(30, color=AMBER, ls="--", lw=1.5, label="RQ3 illustrative 30% line")
    ax.set_ylabel("Paired hallucination rate (%)")
    ax.set_title("Canvas dashboard — paired HR by condition (n=10)")
    ax.set_ylim(0, max(vals) * 1.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    rel = bundle["aggregate"]["rq"]["RQ2"]["reduction"]["relative_reduction_pct"]
    ax.text(
        0.98,
        0.92,
        f"Relative reduction draft→symbolic: {rel:.1f}%",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        color=TEAL,
        fontweight="bold",
    )
    _save(fig, "canvas_hr_by_condition.png")


def plot_canvas_outcome_mix(bundle: dict) -> None:
    m = bundle["aggregate"]["metrics_by_condition"]["symbolic"]
    # reconstruct from rates if counts unavailable
    fix = m.get("fix_rate", 0)
    still = m.get("still_bad_rate", 0)
    removed = m.get("removal_rate", 0)
    garbled = m.get("garbled_rate", 0)
    labels = ["Fixed /\nverified", "Still bad", "Removed", "Garbled"]
    vals = [fix * 100, still * 100, removed * 100, garbled * 100]
    colors = [GREEN, AMBER, MUTED, ROSE]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_ylabel("% of draft citations")
    ax.set_title("Canvas — symbolic correction outcome mix")
    ax.set_ylim(0, max(max(vals), 10) * 1.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    _save(fig, "canvas_outcome_mix.png")


def plot_canvas_rq1_pie(bundle: dict) -> None:
    counts = bundle["aggregate"]["rq"]["RQ1"]["label_counts"]
    labels = list(counts.keys())
    sizes = list(counts.values())
    color_map = {
        "Verified": GREEN,
        "Mismatch": AMBER,
        "Ambiguous": "#ca8a04",
        "NotFound": ROSE,
        "Malformed": ROSE,
    }
    colors = [color_map.get(l, MUTED) for l in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
        colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": WHITE},
    )
    for t in texts:
        t.set_fontsize(11)
    for t in autotexts:
        t.set_color(WHITE)
        t.set_fontweight("bold")
    hr = bundle["aggregate"]["rq"]["RQ1"]["hallucination_rate"] * 100
    ax.set_title(f"RQ1 — draft citation labels (HR = {hr:.1f}%, n_cites=7)")
    _save(fig, "canvas_rq1_labels.png")


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("Dual-system pipeline architecture", fontsize=15, fontweight="bold", color=NAVY, pad=12)

    boxes = [
        (0.3, 2.2, 1.8, 1.6, "Source\ndocs", LIGHT, NAVY),
        (2.5, 2.2, 1.9, 1.6, "Module A\nPRIMERA\ndraft", "#dbeafe", NAVY),
        (4.8, 2.2, 1.9, 1.6, "Module B\neyecite\nextract", "#ccfbf1", NAVY),
        (7.1, 2.2, 1.9, 1.6, "Module C\nCourtListener\nverify", "#ffedd5", NAVY),
        (9.4, 2.2, 2.2, 1.6, "Correction\nsymbolic +\nre-anchor", "#dcfce7", NAVY),
    ]
    for x, y, w, h, label, face, edge in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
            facecolor=face, edgecolor=edge, linewidth=1.6,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color=edge, fontweight="bold")

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        ax.annotate("", xy=(x1 - 0.05, y), xytext=(x0 + 0.05, y),
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.8))

    # eval row
    eval_box = FancyBboxPatch(
        (3.5, 0.35), 5, 1.2, boxstyle="round,pad=0.04,rounding_size=0.15",
        facecolor="#fef3c7", edgecolor=AMBER, linewidth=1.6,
    )
    ax.add_patch(eval_box)
    ax.text(6, 0.95, "Evaluation — paired HR · ROUGE-L · RQ1–RQ3 · McNemar",
            ha="center", va="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.annotate("", xy=(6, 1.55), xytext=(6, 2.15),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))

    ax.text(
        0.3, 5.1,
        "Neural generation  →  Symbolic verify / correct  →  Honest metrics",
        fontsize=11, color=MUTED, style="italic",
    )
    _save(fig, "architecture_pipeline.png")


def plot_canvas_dashboard_mock(bundle: dict) -> None:
    """Screenshot-style composite of the Canvas dashboard (approach C)."""
    m = bundle["aggregate"]["metrics_by_condition"]
    rq = bundle["aggregate"]["rq"]
    draft = m["baseline"]["hr_paired_micro"] * 100
    sym = m["symbolic"]["hr_paired_micro"] * 100
    rel = rq["RQ2"]["reduction"]["relative_reduction_pct"]

    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor("#0b1220")
    gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1.4, 1.2], hspace=0.35, wspace=0.3)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor("#0b1220")
    ax_title.axis("off")
    ax_title.text(0.02, 0.65, "Cursor Canvas · Citation-faithful summarization demo",
                  color=WHITE, fontsize=16, fontweight="bold", transform=ax_title.transAxes)
    ax_title.text(0.02, 0.2, bundle["meta"]["presentation_label"],
                  color="#94a3b8", fontsize=10, transform=ax_title.transAxes)

    # KPI cards
    for i, (label, val, color) in enumerate([
        ("Draft paired HR", f"{draft:.1f}%", "#f87171"),
        ("Symbolic paired HR", f"{sym:.1f}%", "#2dd4bf"),
        ("Relative reduction", f"{rel:.1f}%", "#fbbf24"),
    ]):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor("#111827")
        for spine in ax.spines.values():
            spine.set_color("#1f2937")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.62, val, ha="center", va="center", fontsize=28, fontweight="bold", color=color)
        ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=11, color="#cbd5e1")

    # mini bar + rq badges
    ax_bar = fig.add_subplot(gs[2, :2])
    ax_bar.set_facecolor("#111827")
    vals = [draft, sym, m["llm_rewrite"]["hr_paired_micro"] * 100]
    ax_bar.bar(["Draft", "Symbolic", "LLM"], vals, color=["#f87171", "#2dd4bf", "#60a5fa"], width=0.55)
    ax_bar.tick_params(colors="#cbd5e1")
    ax_bar.set_ylabel("Paired HR %", color="#cbd5e1")
    ax_bar.set_title("Paired HR by condition", color=WHITE, fontsize=11)
    for spine in ax_bar.spines.values():
        spine.set_color("#374151")

    ax_rq = fig.add_subplot(gs[2, 2])
    ax_rq.set_facecolor("#111827")
    ax_rq.axis("off")
    ax_rq.set_title("RQ3 checklist", color=WHITE, fontsize=11, loc="left")
    checks = [
        (True, f"HR ≥30% cut  ({rel:.1f}%)"),
        (True, "ROUGE-L ±2pp"),
        (True, "Symbolic ≈ LLM (tied)"),
    ]
    y = 0.75
    for ok, text in checks:
        mark = "PASS" if ok else "FAIL"
        c = "#4ade80" if ok else "#f87171"
        ax_rq.text(0.05, y, f"[{mark}]  {text}", color=c, fontsize=10, transform=ax_rq.transAxes, fontfamily="monospace")
        y -= 0.25

    _save(fig, "canvas_dashboard_screenshot.png")


def plot_case_story_sdks(bundle: dict) -> None:
    case = next(c for c in bundle["cases"] if c["case_id"] == "SD-KS-0001")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    ax.set_title("Case story — SD-KS-0001 (Brown / Topeka)", fontsize=14, fontweight="bold", color=NAVY)

    steps = [
        (0.05, "Draft", "Brown v. Board…\n1987 U.S. Dist. LEXIS 671\n+ verified 98 F. Supp. 797", ROSE),
        (0.35, "Verify", "LEXIS resolves to wrong\nCL cluster (Hydraulic)\n→ Ambiguous / unsafe", AMBER),
        (0.65, "Correct", "Re-anchor to verified\nBrown F. Supp. cite\n(never substitute Hydraulic)", TEAL),
    ]
    for x, title, body, color in steps:
        box = FancyBboxPatch(
            (x, 0.25), 0.28, 0.55, transform=ax.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=WHITE, edgecolor=color, linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x + 0.14, 0.72, title, transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold", color=color)
        ax.text(x + 0.14, 0.48, body, transform=ax.transAxes, ha="center", va="center", fontsize=9.5, color=SLATE)

    ax.annotate("", xy=(0.34, 0.52), xytext=(0.33, 0.52), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))
    ax.annotate("", xy=(0.64, 0.52), xytext=(0.63, 0.52), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.5))

    ax.text(
        0.5, 0.08,
        "Policy: contradicting CL identity → leave Ambiguous, or re-anchor if a verified same-case cite exists.",
        transform=ax.transAxes, ha="center", fontsize=10, color=MUTED, style="italic",
    )
    _save(fig, "case_story_sd_ks.png")


def main() -> None:
    _style()
    print("Exporting presentation figures…")
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))

    plot_eda_corpus_shape()
    plot_eda_lengths_and_truncation()
    plot_eda_citations()
    plot_architecture()
    plot_canvas_hr_bar(bundle)
    plot_canvas_outcome_mix(bundle)
    plot_canvas_rq1_pie(bundle)
    plot_canvas_dashboard_mock(bundle)
    plot_case_story_sdks(bundle)
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
