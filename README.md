# Dual-System Hallucination-Free Legal Document Summarization

A neuro-symbolic pipeline for legal case summarization that verifies every case citation a neural summarizer generates against [CourtListener](https://www.courtlistener.com/), and corrects the ones that don't check out, before the summary ships. Master's thesis, SRH University of Applied Sciences, 2026.

**Paper:** [`paper/thesis.pdf`](paper/thesis.pdf) · **LaTeX source:** [`paper/latex/`](paper/latex/)

## Headline results

Full-scale evaluation, 100 citation-dense Multi-LexSum test-split cases, three architecturally different generators (PRIMERA, an LLM zero-shot variant, an LLM+retrieval variant), n=100 each:

| Generator | Baseline HR | HR after correction | Relative reduction | McNemar p |
|---|---|---|---|---|
| PRIMERA | 82.8% | 27.6% | 71.8% | $3.05\times10^{-5}$ |
| LLM zero-shot | 57.5% | 22.2% | 56.1% | $2.94\times10^{-39}$ |
| LLM + retrieval | 70.5% | 24.2% | 65.9% | $1.87\times10^{-96}$ |

- **Zero regression:** across all 1,083 paired citations evaluated, symbolic correction never turned a good citation into a bad one.
- **Quality preserved:** ROUGE-L$_{\text{sum}}$ moves by at most 0.07 percentage points against a ±2pp tolerance; BERTScore F1 moves by at most 0.0003.
- Full tables and statistical tests: [`results/`](results/).

## How it works

```
draft summary (Module A)  →  eyecite extraction (Module B)  →  CourtListener verify (Module C)  →  bounded symbolic correction
```

- **Module A** — neural draft generation. Three interchangeable generators: PRIMERA (`src/module_a_summarizer/`), an LLM zero-shot variant, and an LLM+retrieval variant (`src/module_a_llm/`).
- **Module B** — [eyecite](https://github.com/freelawproject/eyecite)-based full-span citation extraction, with length-preserving markup sanitization (`src/module_b_extraction/`).
- **Module C** — CourtListener lookup, fuzzy field matching, and a five-label taxonomy (Verified / Mismatch / Ambiguous / NotFound / Malformed) (`src/module_c_verifier/`).
- **Correction loop** — redacts unverifiable citations, safely repairs soft mismatches, and re-anchors Ambiguous citations to a Verified same-case citation elsewhere in the summary; never substitutes a contradicting case identity (`src/correction_loop/`).

See `paper/thesis.pdf`, Chapter 6 ("Methodology"), for the full design rationale, the formal correction-policy decision rule, and inline code listings for each stage above.

## Try it live

`demo/live_showcase.py` is a Streamlit app that runs the real pipeline (not a replay) against a live CourtListener call, one case at a time:

```bash
pip install -r requirements.txt streamlit
cp .env.example .env   # fill in your own CourtListener + HF tokens
streamlit run demo/live_showcase.py
```

## Repository structure

```
paper/        LaTeX source and the compiled thesis PDF
src/          Modules A/B/C, correction loop, evaluation harness
scripts/      Batch generation, verification, and comparison scripts
notebooks/    RunPod generation notebooks (PRIMERA + LLM conditions)
demo/         Live Streamlit pipeline demo
config/       Experiment YAML configs
docs/         Setup notes (RunPod, design memo, advisor gate)
results/      Aggregate metrics, statistical tests, and RQ answers per generator
```

## What's not in this repository

Multi-LexSum (Shen et al., 2022) is licensed CC BY-NC by its original authors. Raw source documents, gold summaries, and full generated-summary text are not redistributed here — only aggregate statistics (hallucination rates, ROUGE/BERTScore deltas, p-values). See `paper/thesis.pdf`, Chapter 4 ("Data Source"), for the dataset and how to obtain it directly from the [Multi-LexSum release](https://huggingface.co/datasets/allenai/multi_lexsum).

## Citation

```bibtex
@mastersthesis{chirumamilla2026legalcitation,
  author = {Chirumamilla, Raja Karthik},
  title  = {Dual-System Hallucination-Free Legal Document Summarization: A Neuro-Symbolic Framework for Legal Summarization with Verifiable Case Citations},
  school = {SRH University of Applied Sciences},
  year   = {2026}
}
```

## Licence

Code is MIT-licensed (see [`LICENSE`](LICENSE)). The thesis text is all rights reserved.
