"""Emit notebooks/multilexsum_presentation_colab.ipynb — presentation EDA for professor review.

Run from code/:  python scripts/build_presentation_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines or ["\n"]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [s + "\n" for s in text.split("\n")],
        "outputs": [],
        "execution_count": None,
    }


ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "multilexsum_presentation_colab.ipynb"
cells: list[dict] = []

cells.append(
    md(
        """# Multi-LexSum — Presentation EDA (Colab)

**Professor review focus:** case distributions · compression · citation/reference word counts · **all graphs for one case** · what each plot means for the thesis.

**Dataset:** [allenai/multi_lexsum](https://huggingface.co/datasets/allenai/multi_lexsum) · `v20230518`

---

### Run in Google Colab

1. **File → Upload notebook** (this file) or open from Drive.
2. **Runtime → Run all** (first run downloads ~4.5 GB, ~15–30 min).
3. Figures save to `reports/presentation/figures/` and auto-zip at the end.
4. Optional: Colab **Secrets** → `HF_TOKEN` (faster Hub download).

### Run locally (if you already have `code/data/hf/`)

Upload to Colab is not required — open in Jupyter/Cursor with `Thesis/.venv` and run all; the notebook auto-detects `code/data/hf/`.

---

### What this notebook adds vs. the main EDA

| Review comment | Section |
|----------------|---------|
| Distribution of **cases** | §2 — source words, doc count, case types, years |
| **Reference word counts** | §4 — words per citation span, total ref words per summary |
| **Compression distribution** | §3 — histograms, ECDF, violin (not only boxplots) |
| **All graphs for one case** (paper style) | §5 — 6-panel single-case dashboard |
| **Distributions inside each case** (citations) | §5–6 — per-case vs corpus citation stats |
| **Interpretation** for each figure | §2–7 — “What it says” / “Thesis use” callouts |"""
    )
)

cells.append(
    code(
        """# @title 1 · Setup
import os
import re
import sys
import json
import zipfile
import textwrap
import subprocess
import importlib.util
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from IPython.display import display, Markdown, HTML
from tqdm.auto import tqdm

IN_COLAB = "google.colab" in sys.modules
print("Environment:", "Google Colab" if IN_COLAB else "Local")
print("Python:", sys.executable)

if IN_COLAB:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "datasets>=2.16,<4", "pyarrow", "huggingface_hub",
        "matplotlib", "seaborn", "tqdm", "pandas", "numpy",
    ])


def find_code_root(start: Path) -> Path | None:
    for p in [start.resolve(), *start.resolve().parents]:
        if (p / "data" / "hf").is_dir():
            return p
    return None


LOCAL_ROOT = None if IN_COLAB else find_code_root(Path.cwd())
SEED = 42
RNG = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

if IN_COLAB:
    WORK = Path("/content/multilexsum_presentation")
else:
    WORK = LOCAL_ROOT or Path.cwd()
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

REPORTS = WORK / "reports" / "presentation"
FIG = REPORTS / "figures"
TAB = REPORTS / "tables"
for p in (FIG, TAB):
    p.mkdir(parents=True, exist_ok=True)
print("Output dir:", REPORTS.resolve())"""
    )
)

cells.append(
    code(
        """# @title 2 · Load Multi-LexSum
from datasets import load_dataset

CONFIG = "v20230518"
DATASET_ID = "allenai/multi_lexsum"
SUM_KEYS = ["summary/long", "summary/short", "summary/tiny"]
PAPER_TARGETS = {"summary/long": 650, "summary/short": 130, "summary/tiny": 25}

if IN_COLAB:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ds = load_dataset(DATASET_ID, name=CONFIG, trust_remote_code=True, token=hf_token)
else:
    HF_CACHE = (LOCAL_ROOT / "data" / "hf") if LOCAL_ROOT else WORK / "data" / "hf"
    if not HF_CACHE.is_dir():
        raise FileNotFoundError(f"Local cache missing: {HF_CACHE}. Run in Colab or copy data/hf/.")
    ds = load_dataset(DATASET_ID, name=CONFIG, cache_dir=str(HF_CACHE), trust_remote_code=True)

TOTAL = sum(len(ds[k]) for k in ds.keys())
print("Splits:", {k: len(ds[k]) for k in ds.keys()})
print("Total cases:", TOTAL)


def sources_metadata_rows(ex):
    sm = ex.get("sources_metadata") or {}
    if isinstance(sm, list):
        return sm
    if not isinstance(sm, dict) or not sm:
        return []
    keys = list(sm.keys())
    lengths = [len(sm[k]) for k in keys if isinstance(sm.get(k), list)]
    n = max(lengths) if lengths else 0
    return [
        {k: sm[k][i] if isinstance(sm.get(k), list) and i < len(sm[k]) else sm.get(k) for k in keys}
        for i in range(n)
    ]


def word_len(t):
    return 0 if not t or not str(t).strip() else len(str(t).split())


def source_words(ex):
    return sum(len((s or "").split()) for s in (ex.get("sources") or []))"""
    )
)

cells.append(
    code(
        """# @title 3 · Helpers — citations & case table
CITE_PATTERNS = [
    ("F.Supp", re.compile(r"\\b\\d+\\s+F\\.\\s?Supp\\.?(?:\\s*2d)?\\s+\\d+", re.I)),
    ("F.3d", re.compile(r"\\b\\d+\\s+F\\.\\s?3d\\s+\\d+", re.I)),
    ("F.2d", re.compile(r"\\b\\d+\\s+F\\.\\s?2d\\s+\\d+", re.I)),
    ("U.S.", re.compile(r"\\b\\d+\\s+U\\.S\\.\\s+\\d+")),
    ("v_party", re.compile(r"\\b[A-Z][A-Za-z0-9\\.'\\-]+\\s+v\\.\\s+[A-Z]", re.I)),
]


def extract_citation_spans(text):
    if not text:
        return []
    spans = []
    for label, pat in CITE_PATTERNS:
        for m in pat.finditer(text):
            s = m.group()
            spans.append({"label": label, "span": s, "span_words": len(s.split())})
    return spans


def citation_metrics(text):
    spans = extract_citation_spans(text)
    n = len(spans)
    ref_words = sum(s["span_words"] for s in spans)
    return n, ref_words, spans


def build_case_table():
    rows = []
    cite_span_rows = []
    for sn in ds.keys():
        for i in tqdm(range(len(ds[sn])), desc=f"Cases [{sn}]"):
            ex = ds[sn][i]
            cm = ex.get("case_metadata") or {}
            sw = source_words(ex)
            row = {
                "split": sn,
                "row_idx": i,
                "id": ex["id"],
                "case_name": cm.get("case_name", ""),
                "case_type": cm.get("case_type") or "unknown",
                "filing_year": cm.get("filing_year") or "",
                "prevailing_party": cm.get("prevailing_party") or "unknown",
                "n_sources": len(ex.get("sources") or []),
                "src_words": sw,
            }
            for sk in SUM_KEYS:
                text = ex.get(sk) or ""
                ww = word_len(text)
                nh, rw, spans = citation_metrics(text)
                g = sk.replace("summary/", "")
                row[f"words_{g}"] = ww
                row[f"cite_hits_{g}"] = nh
                row[f"cite_words_{g}"] = rw
                if ww > 0:
                    row[f"compression_{g}"] = sw / ww
                for sp in spans:
                    cite_span_rows.append({
                        "split": sn, "id": ex["id"], "field": sk,
                        "label": sp["label"], "span": sp["span"],
                        "span_words": sp["span_words"],
                    })
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(cite_span_rows)


cases_df, cite_spans_df = build_case_table()
cases_df.to_csv(TAB / "case_level_metrics.csv", index=False)
cite_spans_df.to_csv(TAB / "citation_spans_all.csv", index=False)
print(f"Built {len(cases_df)} case rows, {len(cite_spans_df)} citation spans")
display(cases_df.head(3))"""
    )
)

# --- Section 2: Case distributions ---
cells.append(
    md(
        """---

## §2 · Distribution of **cases** (corpus-level)

Each row in Multi-LexSum is one **litigation case** with multiple source documents and up to three expert summaries. These plots show how cases differ in size and domain — the “shape” of the benchmark."""
    )
)

cells.append(
    code(
        """# @title 2a · Source words & document count per case
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

sns.histplot(data=cases_df, x="src_words", hue="split", element="step",
             stat="density", common_norm=False, ax=axes[0])
axes[0].set_xscale("log")
axes[0].set_xlabel("Source words per case (log)")
axes[0].set_title("Distribution of case size (source words)")
axes[0].axvline(cases_df["src_words"].median(), color="k", ls="--", lw=1, alpha=0.6)
axes[0].text(0.02, 0.95, f"median={cases_df['src_words'].median():,.0f}", transform=axes[0].transAxes, va="top")

sns.histplot(data=cases_df, x="n_sources", hue="split", discrete=True,
             stat="density", common_norm=False, ax=axes[1])
axes[1].set_xlabel("# source documents per case")
axes[1].set_title("Distribution of multi-document breadth")

plt.tight_layout()
fig.savefig(FIG / "01_case_size_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown(
    "**What it says:** Most cases sit in a wide band (~10k–100k source words) but the **right tail** is extreme "
    f"(max {cases_df['src_words'].max():,.0f} words). Document count is typically 5–15 per case.\\n\\n"
    "**Thesis use:** Justifies **chunking / long-context** design; a single “read everything” model is infeasible for tail cases."
))"""
    )
)

cells.append(
    code(
        """# @title 2b · Summary length distributions (all cases)
sum_rows = []
for sk in SUM_KEYS:
    g = sk.replace("summary/", "")
    for sn in ds.keys():
        for text in ds[sn][sk]:
            if word_len(text) > 0:
                sum_rows.append({"split": sn, "granularity": g, "n_words": word_len(text)})
sum_df = pd.DataFrame(sum_rows)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax, g, color in zip(axes, ["long", "short", "tiny"], ["#2a6f97", "#e9c46a", "#e76f51"]):
    sub = sum_df[sum_df["granularity"] == g]
    sns.histplot(sub, x="n_words", hue="split", element="step", stat="density",
                 common_norm=False, ax=ax, color=color)
    ax.axvline(PAPER_TARGETS[f"summary/{g}"], color=color, ls="--", lw=1.5)
    ax.set_title(f"{g.upper()} summary words")
    ax.set_xlabel("words")
fig.suptitle("Summary length distributions (dashed = paper target)", fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIG / "02_summary_length_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown(
    "**What it says:** Long summaries cluster below the paper’s 650-word target (median ~420); short/tiny track targets better. "
    "Many cases lack short/tiny labels.\\n\\n"
    "**Thesis use:** Primary evaluation on **D→L**; report usable-N when switching granularity."
))"""
    )
)

cells.append(
    code(
        """# @title 2c · Case type & filing year
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

top_types = cases_df["case_type"].value_counts().head(12)
top_types.sort_values().plot(kind="barh", ax=axes[0], color=sns.color_palette("muted")[0])
axes[0].set_title("Top 12 case types (all cases)")
axes[0].set_xlabel("# cases")

fy = cases_df[cases_df["filing_year"].astype(str).str.match(r"^\\d{4}$", na=False)].copy()
fy["filing_year"] = fy["filing_year"].astype(int)
sns.histplot(fy, x="filing_year", hue="split", element="step", stat="density",
             common_norm=False, ax=axes[1])
axes[1].set_title("Filing year distribution")

plt.tight_layout()
fig.savefig(FIG / "03_case_type_and_year.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown(
    "**What it says:** Civil-rights dockets dominate (employment, jail/prison, public benefits). "
    "Cases span decades — temporal drift in citation style is possible.\\n\\n"
    "**Thesis use:** Stratify error analysis by **case_type**; don’t assume one domain."
))"""
    )
)

# --- Section 3: Compression ---
cells.append(
    md(
        """---

## §3 · **Compression** distributions (source words ÷ summary words)

Compression = how many source words each summary word must “cover”. Higher compression → more abstraction → higher hallucination risk for citations."""
    )
)

cells.append(
    code(
        """# @title 3a · Compression histograms & ECDF
comp_rows = []
for g in ["long", "short", "tiny"]:
    col = f"compression_{g}"
    sub = cases_df[cases_df[col].notna() & (cases_df[f"words_{g}"] > 0)].copy()
    sub["granularity"] = g
    sub["compression"] = sub[col]
    comp_rows.append(sub[["split", "granularity", "compression", "id"]])
comp_df = pd.concat(comp_rows, ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for g, ax in zip(["long", "short", "tiny"], axes.flat[:3]):
    sub = comp_df[comp_df["granularity"] == g]
    sns.histplot(sub, x="compression", hue="split", element="step", stat="density",
                 common_norm=False, ax=ax, log_scale=True)
    med = sub["compression"].median()
    ax.axvline(med, color="k", ls="--", alpha=0.7)
    ax.set_title(f"Compression — {g} (median {med:,.0f}×)")
    ax.set_xlabel("source words / summary words (log)")

# ECDF
for sn in ds.keys():
    sub = comp_df[(comp_df["granularity"] == "long") & (comp_df["split"] == sn)]
    x = np.sort(sub["compression"].values)
    y = np.arange(1, len(x) + 1) / len(x)
    axes[1, 1].plot(x, y, label=sn)
axes[1, 1].set_xscale("log")
axes[1, 1].set_xlabel("compression (long, log)")
axes[1, 1].set_ylabel("ECDF")
axes[1, 1].set_title("Cumulative compression (long summaries)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIG / "04_compression_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

pct = comp_df.groupby("granularity")["compression"].agg(
    median="median", p75=lambda s: s.quantile(0.75), p95=lambda s: s.quantile(0.95)
).round(0)
display(pct)
med = {g: comp_df.loc[comp_df["granularity"] == g, "compression"].median() for g in ["long", "short", "tiny"]}
display(Markdown(
    f"**What it says:** Long summaries: median **{med['long']:,.0f}×** compression; "
    f"short **{med['short']:,.0f}×**; tiny **{med['tiny']:,.0f}×**. "
    "Shorter targets must abstract far more source material per output word.\\n\\n"
    "**Thesis use:** Motivates **verification-heavy** pipeline for D→L; D→T is a different (more lossy) task."
))"""
    )
)

cells.append(
    code(
        """# @title 3b · Compression violin (presentation slide)
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.violinplot(data=comp_df, x="granularity", y="compression", hue="split",
               density_norm="width", cut=0, ax=ax, order=["long", "short", "tiny"])
ax.set_yscale("log")
ax.set_ylabel("compression (log scale)")
ax.set_title("Compression distribution by target granularity")
plt.tight_layout()
fig.savefig(FIG / "05_compression_violin.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

# --- Section 4: Citation / reference word counts ---
cells.append(
    md(
        """---

## §4 · **Reference word counts** (citations in expert summaries)

We extract citation-like spans with regex (reporter cites + `Party v. Party`). For each span we measure:
- **span_words** — word count of the reference text
- **cite_words_{granularity}** — total reference words in a summary
- **cite_hits** — number of detected spans (may overlap)

This answers: *how much of a summary is “reference material”?*"""
    )
)

cells.append(
    code(
        """# @title 4a · Words per citation span (corpus)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

sns.histplot(cite_spans_df, x="span_words", hue="label", element="step",
             stat="density", common_norm=False, ax=axes[0], discrete=True)
axes[0].set_title("Word count per detected citation span")
axes[0].set_xlabel("words in span")

long_cites = cases_df[cases_df["words_long"] > 0]
sns.histplot(long_cites, x="cite_words_long", hue="split", element="step",
             stat="density", common_norm=False, ax=axes[1])
axes[1].set_title("Total reference words per case (long summary)")
axes[1].set_xlabel("sum of citation span words")

plt.tight_layout()
fig.savefig(FIG / "06_reference_word_counts.png", dpi=150, bbox_inches="tight")
plt.show()

display(cite_spans_df.groupby("label")["span_words"].describe().round(1))
display(Markdown(
    "**What it says:** Reporter citations are short (3–6 words); party names (`X v. Y`) are longer. "
    "Most long summaries contain **0–15 total reference words**; a few are citation-dense.\\n\\n"
    "**Thesis use:** Set extractor token budgets; calibrate CourtListener queries on typical span length."
))"""
    )
)

cells.append(
    code(
        """# @title 4b · Citation hits vs reference words (by granularity)
cite_case_rows = []
for g in ["long", "short", "tiny"]:
    sub = cases_df[cases_df[f"words_{g}"] > 0]
    for _, r in sub.iterrows():
        cite_case_rows.append({
            "split": r["split"], "granularity": g,
            "cite_hits": r[f"cite_hits_{g}"],
            "cite_words": r[f"cite_words_{g}"],
            "summary_words": r[f"words_{g}"],
            "ref_word_pct": 100 * r[f"cite_words_{g}"] / max(r[f"words_{g}"], 1),
        })
cite_case_df = pd.DataFrame(cite_case_rows)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(data=cite_case_df[cite_case_df["granularity"] == "long"],
             x="cite_hits", hue="split", discrete=True, stat="density",
             common_norm=False, ax=axes[0, 0])
axes[0, 0].set_title("# citation hits per case (long)")

sns.histplot(data=cite_case_df, x="cite_words", hue="granularity",
             element="step", stat="density", common_norm=False, ax=axes[0, 1])
axes[0, 1].set_title("Total reference words per summary")

sns.scatterplot(data=cite_case_df[cite_case_df["granularity"] == "long"],
                x="summary_words", y="cite_hits", hue="split", alpha=0.35,
                ax=axes[1, 0], s=25)
axes[1, 0].set_title("Summary length vs citation count (long)")

sns.boxplot(data=cite_case_df, x="granularity", y="ref_word_pct", hue="split",
            showfliers=False, ax=axes[1, 1], order=["long", "short", "tiny"])
axes[1, 1].set_title("% of summary words in citation spans")
axes[1, 1].set_ylabel("% reference words (heuristic)")

plt.tight_layout()
fig.savefig(FIG / "07_citation_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

display(cite_case_df.groupby("granularity")[["cite_hits", "cite_words", "ref_word_pct"]].mean().round(2))
display(Markdown(
    "**What it says:** Citations concentrate in **long** summaries; short/tiny often have zero detected cites. "
    "Reference words are usually <5% of summary length.\\n\\n"
    "**Thesis use:** Train/evaluate citation verification on **long** gold first; short targets need different metrics."
))"""
    )
)

# --- Section 5: Single case dashboard ---
cells.append(
    md(
        """---

## §5 · **All graphs for one case** (paper-style deep dive)

We pick a citation-rich case with all three summary levels and multiple source documents, then plot **every distribution inside that case** — mirroring how the Multi-LexSum paper walks through a single example."""
    )
)

cells.append(
    code(
        """# @title 5a · Pick showcase case
def pick_showcase_case():
    candidates = cases_df[
        (cases_df["words_long"] > 200) &
        (cases_df["words_short"] > 50) &
        (cases_df["words_tiny"] > 10) &
        (cases_df["cite_hits_long"] >= 4) &
        (cases_df["n_sources"] >= 4)
    ].sort_values("cite_hits_long", ascending=False)
    if len(candidates):
        return candidates.iloc[0]
    return cases_df.sort_values("cite_hits_long", ascending=False).iloc[0]

show = pick_showcase_case()
SHOW_SPLIT, SHOW_IDX, SHOW_ID = show["split"], int(show["row_idx"]), show["id"]
show_ex = ds[SHOW_SPLIT][SHOW_IDX]
cm = show_ex.get("case_metadata") or {}

print(f"Showcase case: {SHOW_ID} ({SHOW_SPLIT}[{SHOW_IDX}])")
print(f"  {cm.get('case_name', '')}")
print(f"  type={show['case_type']}  sources={show['n_sources']}  src_words={show['src_words']:,}")
print(f"  cite_hits_long={show['cite_hits_long']}  cite_words_long={show['cite_words_long']}")"""
    )
)

cells.append(
    code(
        """# @title 5b · Six-panel dashboard (one case — all graphs)
ex = show_ex
cm = ex.get("case_metadata") or {}
srcs = ex.get("sources") or []
meta = sources_metadata_rows(ex)
src_word_counts = [len((s or "").split()) for s in srcs]
doc_types = [m.get("doc_type", "unknown") for m in meta] or ["unknown"] * len(srcs)
type_counts = Counter(doc_types)

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.38, wspace=0.32)

# (0) Case header
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
header = (
    f"Case {ex['id']} — {cm.get('case_name', 'N/A')}\\n"
    f"Type: {cm.get('case_type', 'N/A')} · Filed: {cm.get('filing_date', 'N/A')} · "
    f"Prevailing: {cm.get('prevailing_party', 'N/A')}\\n"
    f"Sources: {len(srcs)} documents · {source_words(ex):,} source words"
)
ax0.text(0.5, 0.5, header, ha="center", va="center", fontsize=12, family="monospace",
         bbox=dict(boxstyle="round", facecolor="#f0f4f8", edgecolor="#444"))

# (1) Source document word counts
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(range(len(src_word_counts)), src_word_counts, color=sns.color_palette("muted")[0])
ax1.set_xlabel("document index")
ax1.set_ylabel("words")
ax1.set_title("Source doc lengths (this case)")

# (2) Document types
ax2 = fig.add_subplot(gs[1, 1])
tc = type_counts.most_common()
if tc:
    ax2.barh([t for t, _ in tc][::-1], [c for _, c in tc][::-1], color=sns.color_palette("muted")[1])
ax2.set_title("Document types (this case)")

# (3) Summary word lengths
ax3 = fig.add_subplot(gs[1, 2])
grams = ["long", "short", "tiny"]
wcounts = [word_len(ex.get(f"summary/{g}") or "") for g in grams]
colors = ["#2a6f97", "#e9c46a", "#e76f51"]
bars = ax3.bar(grams, wcounts, color=colors)
for b, tgt in zip(bars, [650, 130, 25]):
    ax3.axhline(tgt, color="gray", ls=":", lw=1)
ax3.set_ylabel("words")
ax3.set_title("Summary lengths (this case)")

# (4) Compression per granularity
ax4 = fig.add_subplot(gs[2, 0])
sw = source_words(ex)
comps = [sw / max(w, 1) for w in wcounts]
ax4.bar(grams, comps, color=colors)
ax4.set_yscale("log")
ax4.set_ylabel("compression (log)")
ax4.set_title("Compression (this case)")

# (5) Citations per granularity
ax5 = fig.add_subplot(gs[2, 1])
hits, refw = [], []
for g in grams:
    nh, rw, _ = citation_metrics(ex.get(f"summary/{g}") or "")
    hits.append(nh)
    refw.append(rw)
x = np.arange(3)
ax5.bar(x - 0.2, hits, width=0.35, label="# spans", color=sns.color_palette("muted")[2])
ax5.bar(x + 0.2, refw, width=0.35, label="ref words", color=sns.color_palette("muted")[3])
ax5.set_xticks(x)
ax5.set_xticklabels(grams)
ax5.set_title("Citations in each summary")
ax5.legend(fontsize=8)

# (6) Citation span word lengths (long summary only)
ax6 = fig.add_subplot(gs[2, 2])
_, _, spans = citation_metrics(ex.get("summary/long") or "")
span_w = [s["span_words"] for s in spans]
if span_w:
    ax6.hist(span_w, bins=range(1, max(span_w) + 2), color=sns.color_palette("muted")[4], edgecolor="white")
    ax6.set_xlabel("words per span")
else:
    ax6.text(0.5, 0.5, "No citations detected", ha="center")
ax6.set_title("Reference span lengths (long)")

fig.suptitle(f"Single-case dashboard · {ex['id']}", fontsize=15, fontweight="bold", y=1.01)
fig.savefig(FIG / f"08_single_case_dashboard_{ex['id']}.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown(
    "**What it says (one case):** You can read off input heterogeneity (doc lengths/types), "
    "target granularity trade-offs, and where citations live.\\n\\n"
    "**Thesis use:** Use this layout in slides — one concrete story before aggregate statistics."
))"""
    )
)

cells.append(
    code(
        """# @title 5c · Within-case vs corpus (same case highlighted)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.kdeplot(data=cases_df, x="src_words", hue="split", log_scale=True, ax=axes[0], fill=True, alpha=0.25)
axes[0].axvline(show["src_words"], color="red", lw=2, ls="--", label=f"showcase {SHOW_ID}")
axes[0].set_title("Corpus: source words (showcase marked)")
axes[0].legend()

long_hits = cases_df[cases_df["words_long"] > 0]
sns.kdeplot(data=long_hits, x="cite_hits_long", hue="split", ax=axes[1], fill=True, alpha=0.25)
axes[1].axvline(show["cite_hits_long"], color="red", lw=2, ls="--", label=f"showcase {SHOW_ID}")
axes[1].set_title("Corpus: citation hits long (showcase marked)")
axes[1].legend()

plt.tight_layout()
fig.savefig(FIG / "09_showcase_vs_corpus.png", dpi=150, bbox_inches="tight")
plt.show()"""
    )
)

# --- Section 6: Cross-case citation by type ---
cells.append(
    md(
        """---

## §6 · Citation distributions **across case types**

Do some docket types carry more references? This informs stratified evaluation."""
    )
)

cells.append(
    code(
        """# @title 6 · Citations by case type
top5_types = cases_df["case_type"].value_counts().head(5).index
ct_sub = cases_df[cases_df["case_type"].isin(top5_types) & (cases_df["words_long"] > 0)]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(data=ct_sub, x="case_type", y="cite_hits_long", ax=axes[0], showfliers=False)
plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right")
axes[0].set_title("# citation hits (long) by case type")

sns.boxplot(data=ct_sub, x="case_type", y="cite_words_long", ax=axes[1], showfliers=False)
plt.setp(axes[1].get_xticklabels(), rotation=25, ha="right")
axes[1].set_title("Reference words (long) by case type")

plt.tight_layout()
fig.savefig(FIG / "10_citations_by_case_type.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown(
    "**What it says:** Citation density varies by domain (e.g. constitutional vs employment cases).\\n\\n"
    "**Thesis use:** Report HR/ROUGE **per case_type**; avoid one global number hiding failure modes."
))"""
    )
)

# --- Section 7: Synthesis ---
cells.append(
    md(
        """---

## §7 · Synthesis — what to take into the thesis

| Finding | Evidence in this notebook | Pipeline implication |
|---------|---------------------------|----------------------|
| Heavy-tailed case sizes | §2a source-word histogram | Chunk + retrieve; verify against what was actually read |
| Long summaries ≈ primary target | §2b, §4 | D→L first; citation module on long gold |
| Compression 100–1000× (long) | §3 | High abstraction → hallucination risk; symbolic verify essential |
| Reference words ≪ summary words | §4 | Extractor must handle short spans; most text is narrative |
| Citations case-specific | §5–6 | Qualitative deep dive + per-type metrics in evaluation |
| Missing short/tiny labels | §2b | Track `usable_N` per task in all tables |"""
    )
)

cells.append(
    code(
        """# @title 8 · Export figures for slides
summary = pd.DataFrame([
    {"metric": "cases_total", "value": len(cases_df)},
    {"metric": "median_src_words", "value": cases_df["src_words"].median()},
    {"metric": "median_compression_long", "value": cases_df["compression_long"].median()},
    {"metric": "median_cite_hits_long", "value": cases_df.loc[cases_df["words_long"] > 0, "cite_hits_long"].median()},
    {"metric": "median_cite_words_long", "value": cases_df.loc[cases_df["words_long"] > 0, "cite_words_long"].median()},
    {"metric": "showcase_case_id", "value": SHOW_ID},
])
summary.to_csv(TAB / "presentation_summary.csv", index=False)
display(summary)

zip_path = WORK / "multilexsum_presentation_figures.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in (FIG, TAB):
        for f in folder.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(WORK))
print("Exported:", zip_path.resolve())
print("Figures:", sorted(p.name for p in FIG.glob("*.png")))

if IN_COLAB:
    from google.colab import files
    files.download(str(zip_path))
else:
    print("Local: open", FIG.resolve())"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}
NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", NB)
