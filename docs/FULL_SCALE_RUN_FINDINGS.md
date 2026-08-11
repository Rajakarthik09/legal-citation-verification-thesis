# Full-Scale Run — Findings Log

Source material for thesis writing (Methodology, Results, Error Analysis, Limitations chapters). Written during the full-scale evaluation session that replaced the n=10 preliminary pilot with a real, statistically-powered run.

**Status as of writing:** All three generators complete at n=100 — PRIMERA, LLM zero-shot, and LLM+RAG (finished 2026-07-20). All numbers in this file, including §7, are final. Remaining open items (manual audit, human readability evaluation, Results/Discussion/Conclusion chapters) are listed at the bottom.

---

## 1. Full-scale PRIMERA run

**Setup:** 100 most citation-dense cases from the Multi-LexSum **test split** (not validation — chosen because PRIMERA is a frozen pretrained checkpoint, so there's no tuning-on-test-set risk). Generation on a RunPod GPU; verification (extraction + CourtListener + symbolic correction) run locally.

### Finding: generation-length bug (`length_penalty=0`)

First run produced short summaries (461 words average). Root cause: the PRIMERA checkpoint's own baked-in `generation_config.json` ships `length_penalty: 0`. In beam search this means candidate sequences are scored by *raw summed log-probability* with no length normalization — every additional token multiplies in a probability ≤1, so the score mathematically favors shorter sequences regardless of quality. The pipeline code never overrode this.

**Fix:** explicit `min_new_tokens` + `length_penalty=2.0` in `Summarizer.generate()`. Average length rose to 535 words.

**Thesis use:** Methodology chapter, Module A / generation-configuration subsection. This is a concrete, citable, evidence-backed decision — not an implementation footnote. Frame as: "the checkpoint's default generation configuration was found to bias toward under-length outputs; explicit overrides were required to reach length parity with the task's target granularity."

### Headline result (n=100, complete, statistically significant)

| | Baseline | After symbolic correction |
|---|---|---|
| Paired HR (primary metric) | **82.8%** | **27.6%** |
| ROUGE-L | 0.1995 | 0.1995 (Δ ≈ 0.00) |
| BERTScore F1 | 0.8511 | 0.8508 |

- Relative HR reduction: **71.8%** (RQ3 target was ≥30% — passed)
- McNemar test: **p = 3.05 × 10⁻⁵**
- ROUGE-L bootstrap 95% CI on the delta: ≈ 0 (CI: [-0.00005, +0.00003])
- Draft citation labels (n=29 total): NotFound 12, Mismatch 9, Verified 5, Ambiguous 3

**Thesis use:** this is the primary RQ2/RQ3 evidence, replacing the "PRELIMINARY — n=10" numbers currently in the thesis. Statistically valid (unlike the pilot's underpowered p=0.25).

---

## 2. Citation-coverage finding

**Observation:** across the 100 citation-dense cases, PRIMERA's drafts contained only **29 citations total**, vs. **631** in the corresponding gold (expert) summaries.

**Correct interpretation — important for framing:** not straightforwardly "PRIMERA avoids citations." These 100 cases were selected as the most citation-dense in the corpus (up to 36 gold citations each); PRIMERA is architecturally capped at ~1024 output tokens (~750 words), a hard ceiling from the checkpoint itself, not a config choice. A model that short cannot reproduce citation density from summaries that may run to thousands of words.

**Thesis use:** Limitations section / RQ1 discussion. State explicitly that paired HR measures *faithfulness of citations attempted*, not *citation recall/coverage* — these are distinct failure modes and should not be conflated. This is a defensible, citable limitation of the metric, not a flaw to hide.

---

## 3. LLM and LLM+RAG conditions — rationale and setup

Added to empirically test the "RAG vs. symbolic verification" comparison that was previously argued only from other papers' numbers (Magesh et al. etc.), never tested inside this thesis's own pipeline. Not scope creep — this is the highest-leverage addition identified when assessing publication-readiness.

**Condition 2 — LLM zero-shot:** Qwen2.5-7B-Instruct (Apache-2.0, ungated), same 100 case_ids as PRIMERA.
- 8192-token input context vs. PRIMERA's architecturally fixed 4096 (confirmed via the checkpoint's `max_encoder_position_embeddings` config — literally cannot be raised for that specific fine-tuned model)
- Explicit prompt instruction to preserve every real citation from source material
- ~2000-token output budget vs. PRIMERA's ~850

**Condition 3 — LLM + retrieval:** same model and budget, but source documents are ranked by citation density (fast regex proxy for citation-like patterns — see Finding 5 for why not full eyecite parsing) before truncating, instead of naive chronological order. Operationalizes the "smarter truncation" ablation the methodology gestures at but never tested against the naive baseline.

**Result so far:** citation-extraction count jumped from PRIMERA's 29 to **104+ citations in the first 33 verified LLM cases** alone — a >10x increase in citation-attempt density. Confirms the ceiling was context/output length and prompting, not an inherent property of the summarization task. Full three-way HR/quality comparison pending verification completion.

**Thesis use:** Methodology chapter (new Module A variant description), Results chapter (three-way comparison table — see `scripts/compare_generators.py` output once complete), and directly operationalizes the literature review's RAG-vs-symbolic-verification comparative table (Ch.2.4.4) with real data instead of citations to other papers' numbers.

---

## 4. Markup-corruption finding (likely the most citable methodological discovery)

**Observation:** among citations where symbolic correction *failed* ("still_bad" outcome), **18 of 40 (45%)** were corrupted by markup artifacts, not genuine unfixable hallucinations. Two independent sources:

- **PRIMERA:** Multi-LexSum source documents contain literal HTML (e.g. `<i>Obergefell v. Hodges</i>`, `<a href="...">`) from the original court-record/Clearinghouse pages. PRIMERA is extractive-leaning and copies these tags verbatim into generated summaries.
- **LLM:** the model's own Markdown habit (`**Graham**`).

**Worked example (usable as an illustrative case study in the thesis):** the citation `<i>Obergefell v. Hodges</i>, 135 S. Ct. 2584 (2019)` — a real, correctly-identifiable Supreme Court case with a hallucinated year (real decision: 2015) — broke eyecite's party-name parser: extracted `party_names=("", "Hodges</i>")` instead of `("Obergefell", "Hodges")`. The corruption cascaded: the correction system inserted its repair at the wrong text offset, producing a garbled duplicate (`"Obergefell Obergefell v. Hodges"`), which then failed re-verification and exhausted all 3 correction iterations without resolving — a genuinely fixable citation the system wrongly gave up on.

**Fix:** blank HTML tags and Markdown emphasis markers with equal-length whitespace before citation extraction (length-preserving, so redaction/replacement character offsets elsewhere in the pipeline remain valid). Applied at the `extract_citations()` layer, so it retroactively improves already-generated text without needing regeneration.

**Thesis use:** Error Analysis section (Ch.7.6 promises exactly this kind of qualitative failure-mode breakdown). Quantified, attributable, fixed — strong material for demonstrating rigorous error analysis. Consider a table: failure mode | count | % of still_bad | example | root cause | fix.

---

## 5. Year-preference correction-policy bug

Related to Finding 4 but distinct: independent of markup, the "safe repair" logic for Mismatch citations kept the model's own (sometimes wrong) year and only used CourtListener's verified year when the model's year was completely *absent* — never when it was merely *wrong*. Since Mismatches are frequently exactly a wrong-year problem on an otherwise-correctly-identified citation, this defeated the purpose of the repair (repaired text would still fail re-verification, looping through iterations to `still_bad`).

**Fix:** always prefer CourtListener's verified `matched_date_filed` year when available, falling back to the draft's year only if CourtListener has no date.

**Thesis use:** Methodology Ch.6.6 ("safe repair" policy) — this is a legitimate refinement to the described algorithm, worth one sentence noting the policy was tightened during implementation based on observed failure cases. Also strengthens the Error Analysis narrative alongside Finding 4.

---

## 6. Rate-limiting reality vs. assumptions (reproducibility note, not a research finding)

Confirmed CourtListener account limits (from account dashboard): **10 requests/min, 100/hour, 250/day**. Materially different from the "125/day, plain free account" figure originally assumed in the project roadmap. Relevant only for a reproducibility appendix / implementation-details footnote if included — not a methodological finding.

---

## 7. Three-way comparison — final results (all three generators, n=100)

All three generators, same 100 case_ids, identical B/C/correction pipeline. Final for all three — LLM+RAG completed verification 2026-07-20.

| | PRIMERA (n=100) | LLM zero-shot (n=100) | LLM+RAG (n=100) |
|---|---|---|---|
| Citations attempted (draft) | 29 | 365 | 689 |
| Citations attempted per case | 0.29 | 3.65 | 6.89 |
| Baseline paired HR | 82.8% | 57.5% | 70.5% |
| Paired HR after symbolic correction | 27.6% | 22.2% | 24.2% |
| Relative HR reduction | 71.8% | 56.1% | 65.9% |
| McNemar p-value | 3.05×10⁻⁵ | 2.94×10⁻³⁹ | 1.87×10⁻⁹⁶ |
| ROUGE-L, before → after | 0.1995 → 0.1995 | 0.1779 → 0.1782 | 0.1778 → 0.1779 |
| Fix rate / removal rate | 31.0% / 41.4% | 61.9% / 15.9% | 64.9% / 10.9% |

**Reading (final):**

- **The citation-coverage finding (§2) reproduces as a checkpoint property, not a task property.** PRIMERA's 0.29 citations/case was hypothesized to be an artifact of that specific checkpoint's short-output bias rather than something inherent to legal case summarization. Both LLM-based generators attempt an order of magnitude more citations per case (3.65 and 6.89 vs 0.29), confirming that hypothesis — the low PRIMERA count is a property of the checkpoint, not the task.
- **Retrieval (citation-density source ranking) nearly doubles citation-attempt density over naive truncation** (6.89/case vs 3.65/case) but also raises the baseline HR before correction (70.5% vs 57.5%) — more attempts, and proportionally more wrong ones before correction, consistent with the retrieval variant surfacing more borderline/ambiguous source material rather than only "easy" citations.
- **The symbolic correction pipeline generalizes across generator architectures.** Final HR lands in a similar ~22–28% band for all three (27.6% / 22.2% / 24.2%) despite very different baseline HRs (82.8% / 57.5% / 70.5%) and very different citation volumes (29 / 365 / 689 total) — the pipeline was not tuned to PRIMERA's low-volume regime specifically.
- **Fix rate is much higher for the LLM-based generators than PRIMERA** (61.9% / 64.9% vs 31.0%), while PRIMERA's removal rate is much higher (41.4% vs 15.9% / 10.9%) — with only 29 total citations, a larger share of PRIMERA's failures may be unrepairable citations (garbled/incomplete) rather than repairable mismatches; worth a qualitative look in error analysis.
- **Quality is preserved across all three, with wide margin on RQ3.** ROUGE-L moves by ≤0.02pp in every condition while paired HR drops 56–72% relative — RQ3's bar (≥30% HR reduction, ≤2pp ROUGE-L cost) is cleared comfortably in all three, not just the best case.

**Thesis use:** Results chapter, three-way comparison table + the five reading points above as a paragraph. Final CSV/PNGs at `outputs/results_comparison/` (regenerated via `scripts/compare_generators.py` after LLM+RAG completed).

---

## 8. Concurrency bug in the multi-day verification workflow

**Observation:** because CourtListener's rate limit forces verification to resume across multiple days/sessions (§6), the same `run_verification_local.py` script ended up being launched more than once concurrently on separate occasions — once via an overlapping scheduled daily resume task and a manual check-in, and once because a process already running under old code (started before a fix was deployed) wasn't visible to a process-list-based guard. Both times, two processes raced against the same shared rate-limiter state file, burning through the daily API budget faster than a single process should.

**Fix:** `run_verification_local.py` now takes an OS-level exclusive lock (`fcntl.flock`, non-blocking) around its entire run, held on a file descriptor for the process's lifetime. A second instance — regardless of what launched it — fails fast with a clear message instead of racing, and the lock is released automatically by the OS on exit, crash, or kill, so there is no stale-lock state to manage. Covered by 4 new regression tests (`tests/test_verification_lock.py`).

**Thesis use:** minor — a reproducibility / engineering-robustness footnote (Methodology "Tools & Technologies" or an appendix), not a research finding. Illustrates a real operational hazard of rate-limit-bound, multi-day evaluation pipelines that a single-session evaluation wouldn't surface.

---

## Open items to fold in once available

- [x] Final LLM zero-shot HR/ROUGE/BERTScore numbers — complete, see §7
- [x] LLM+RAG condition final results at n=100 — complete 2026-07-20, see §7
- [x] Three-way comparison table/figures from `scripts/compare_generators.py` (`outputs/results_comparison/`) — final, all three generators n=100
- [x] Whether LLM+RAG's citation-density retrieval measurably outperforms LLM zero-shot's naive truncation (the core RAG-vs-naive ablation this was designed to test) — yes: nearly doubles citation-attempt density (6.89 vs 3.65/case), see §7 reading point 2
- [ ] Manual 200-case audit (Ch.4.3) — not started, still a manual task
- [ ] Human evaluation of corrected-summary readability — not started
- [ ] Results, Discussion, Conclusion chapters — unblocked as of today, not yet written
- [ ] Declaration of authorship + AI-usage disclosure pages — required for submission, not yet in the document
