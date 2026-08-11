# Research Question Answers

**Full-scale evaluation — n=100 citation-rich test-split cases, D→L, LLM + citation-density retrieval (RAG condition)**

## RQ1

**Question:** What types and frequencies of case-citation hallucinations do baseline summarizers produce?

**Answer:** Across 100 cases, 689 citations extracted from draft summaries. Overall HR=70.5%. Most common labels: Ambiguous(215), Mismatch(212), Verified(203), NotFound(59).

- **n_cases:** 100
- **total_citations:** 689
- **label_counts:** {'Mismatch': 212, 'Ambiguous': 215, 'Verified': 203, 'NotFound': 59}
- **label_rates:** {'Mismatch': 0.3077, 'Ambiguous': 0.312, 'Verified': 0.2946, 'NotFound': 0.0856}
- **hallucination_rate:** 0.7054

## RQ2

**Question:** Does symbolic verification + correction reduce HR vs neural-only baseline?

**Answer:** Paired HR dropped from 70.5% (baseline) to 24.2% (symbolic correction), relative reduction 65.9%. Surviving-citation HR=30.0%. McNemar p=1.8726705418768793e-96.

- **hr_baseline_paired_micro:** 0.7054
- **hr_symbolic_paired_micro:** 0.2424
- **hr_symbolic_surviving_micro:** 0.2997
- **fix_rate:** 0.6488
- **removal_rate:** 0.1089
- **garbled_rate:** 0.0
- **reduction:** {'absolute_reduction': 0.3362, 'relative_reduction_pct': 65.9281, 'mean_hr_draft': 0.51, 'mean_hr_final': 0.1738}
- **mcnemar_test:** {'n_paired_citations': 689, 'draft_hallucinated_final_success': 319, 'draft_ok_final_bad': 0, 'draft_only_hallucinated': 319, 'final_only_hallucinated': 0, 'p_value': 1.8726705418768793e-96, 'interpretation': 'Significant reduction in hallucination'}

## RQ3

**Question:** Can the dual-system approach preserve quality while reducing HR by >=30%?

**Answer:** Paired HR relative reduction 65.9% (target 30.0%): PASS. ROUGE-Lsum delta -0.07 pp (tolerance ±2.0): PASS.

- **hr_relative_reduction_pct:** 65.93
- **hr_target_pct:** 30.0
- **hr_target_met:** True
- **primary_rouge_variant:** rougeLsum
- **rougeLsum_delta_pct_points:** -0.07
- **rougeL_delta_pct_points:** 0.01
- **rouge_tolerance_pct:** 2.0
- **rouge_target_met:** True
- **rougeLsum_bootstrap_ci:** {'mean_delta': -0.0007280579334497012, 'ci_low': -0.0013050779872983338, 'ci_high': -0.00012092915437586115, 'n': 100, 'ci_level': 0.95}
- **symbolic_vs_llm_rewrite:** {'hr_paired_symbolic': 0.24238026124818576, 'hr_paired_llm_rewrite': 0.2960812772133527, 'rougeLsum_symbolic': 0.42952477876584944, 'rougeLsum_llm_rewrite': 0.42975763192273286, 'rougeL_symbolic': 0.1779202442871324, 'rougeL_llm_rewrite': 0.17836078553456083}

## Limitations

- Status: Full-scale evaluation — n=100 citation-rich test-split cases, D→L, LLM + citation-density retrieval (RAG condition)
- Evaluation on test split, n=100 (not full test set).
- D→L (long summaries) only; short/tiny descoped after EDA.
- Primary HR is paired draft-citation HR; surviving-citation HR reported separately.
- Primary quality metric is ROUGE-Lsum (sentence-split union-LCS), the variant conventionally reported for multi-sentence summarization; plain ROUGE-L is also reported but is not comparable to published summarization baselines.
- CourtListener access level 10 req/min, 100/hr, 250/day; caching used; multi-day runs required at full scale.
- Source truncation to the generator's context budget (known limitation).
- Manual audit template exported separately; labels not included in automated RQ answers.