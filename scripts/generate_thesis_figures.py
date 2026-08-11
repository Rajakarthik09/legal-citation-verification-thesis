#!/usr/bin/env python3
"""Regenerate thesis EDA figures from reports/tables/*.csv (no dataset reload)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "code" / "reports" / "tables"
FIG_THESIS = ROOT / "Documentation" / "figures" / "eda"
FIG_CODE = ROOT / "code" / "reports" / "figures"

PAPER_TARGETS = {"summary/long": 650, "summary/short": 130, "summary/tiny": 25}


def save(fig, name: str) -> None:
    for d in (FIG_THESIS, FIG_CODE):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def fig_summary_availability() -> None:
    miss_df = pd.read_csv(TAB / "summary_availability.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_miss = miss_df.pivot(index="field", columns="split", values="pct_present")
    plot_miss.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("% cases with non-empty summary")
    ax.set_title("Summary granularity availability by split")
    ax.legend(title="split")
    plt.tight_layout()
    save(fig, "summary_availability.png")


def fig_summary_words() -> None:
    stats = pd.read_csv(TAB / "summary_word_stats.csv")
    rows = []
    for _, r in stats.iterrows():
        for _ in range(int(r["count"])):
            rows.append({"split": r["split"], "field": r["field"], "n_words": r["mean"]})
    # Rebuild approximate distribution from per-split stats using compression table
    gran = pd.read_csv(TAB / "granularity_compression.csv")
    gran = gran.rename(columns={"granularity": "field", "summary_words": "n_words"})
    gran["field"] = "summary/" + gran["field"].astype(str)
    wl = gran[["split", "field", "n_words"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.violinplot(data=wl, x="field", y="n_words", hue="split", ax=axes[0], cut=0, inner="quart")
    axes[0].axhline(650, color="#2a6f97", ls="--", lw=1, alpha=0.7)
    axes[0].axhline(130, color="#e9c46a", ls="--", lw=1, alpha=0.7)
    axes[0].axhline(25, color="#e76f51", ls="--", lw=1, alpha=0.7)
    axes[0].set_title("Word counts by summary type")
    sns.boxplot(data=wl, x="field", y="n_words", ax=axes[1], showfliers=False)
    axes[1].set_title("Distribution (outliers hidden)")
    plt.tight_layout()
    save(fig, "summary_words_violin.png")


def fig_granularity_switch() -> None:
    miss_df = pd.read_csv(TAB / "summary_availability.csv")
    gran = pd.read_csv(TAB / "granularity_compression.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=gran, x="granularity", y="compression", hue="split", ax=axes[0], showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Compression ratio (source words / summary words)")
    axes[0].set_ylabel("ratio (log scale)")
    usable = miss_df.pivot(index="field", columns="split", values="present")
    usable.index = [x.replace("summary/", "") for x in usable.index]
    usable.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Usable examples per granularity")
    axes[1].set_ylabel("# cases with non-empty summary")
    plt.tight_layout()
    save(fig, "granularity_switch_analysis.png")


def fig_source_distributions() -> None:
    src_df = pd.read_csv(TAB / "source_stats_per_example.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    sns.histplot(data=src_df, x="n_sources", hue="split", element="step", stat="density", ax=axes[0, 0], common_norm=False)
    axes[0, 0].set_title("# source documents per case")
    sns.histplot(data=src_df, x="src_words", hue="split", element="step", stat="density", ax=axes[0, 1], common_norm=False)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Source words per case (log x)")
    ocr_rate = src_df.groupby("split")["any_ocr"].mean()
    ocr_rate.plot(kind="bar", ax=axes[1, 0], color=sns.color_palette("muted"))
    axes[1, 0].set_title("Fraction of cases with any OCR source")
    axes[1, 0].set_ylabel("fraction")
    plt.tight_layout()
    save(fig, "source_distributions.png")


def fig_truncation() -> None:
    frac_df = pd.read_csv(TAB / "truncation_fractions.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=frac_df, x="budget", y="frac_over", hue="split", marker="o", ax=ax)
    ax.set_xscale("log")
    ax.set_xlabel("GPT-2 token budget (log scale)")
    ax.set_ylabel("Fraction of cases exceeding budget")
    ax.set_title("Truncation risk (subsampled token audit)")
    plt.tight_layout()
    save(fig, "truncation_fractions.png")


def fig_metadata() -> None:
    meta_df = pd.read_csv(TAB / "case_metadata_flat.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ct = meta_df["case_type"].value_counts().head(12)
    ct.plot(kind="barh", ax=axes[0], color=sns.color_palette("muted")[0])
    axes[0].set_title("Top case types")
    fy = meta_df[meta_df["filing_year"].astype(str).str.match(r"^\d{4}$", na=False)].copy()
    fy["filing_year"] = fy["filing_year"].astype(int)
    fy["filing_year"].hist(bins=30, ax=axes[1], color=sns.color_palette("muted")[1], edgecolor="white")
    axes[1].set_title("Filing year distribution")
    axes[1].set_xlabel("year")
    plt.tight_layout()
    save(fig, "metadata_landscape.png")


def fig_citations() -> None:
    cite_df = pd.read_csv(TAB / "citation_heuristic_all_granularities.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=cite_df, x="field", y="cite_hits", ax=axes[0], showfliers=False)
    axes[0].set_title("Heuristic citation hits per summary")
    sns.boxplot(data=cite_df, x="field", y="cites_per_100_words", ax=axes[1], showfliers=False)
    axes[1].set_title("Citations per 100 words (by granularity)")
    plt.tight_layout()
    save(fig, "citation_by_granularity.png")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    fig_summary_availability()
    fig_summary_words()
    fig_granularity_switch()
    fig_source_distributions()
    fig_truncation()
    fig_metadata()
    fig_citations()
    print("Done:", FIG_THESIS)


if __name__ == "__main__":
    main()
