# Research Question Answers

**Full-scale evaluation — n=100 citation-rich test-split cases, D→L**

## RQ1

**Question:** What types and frequencies of case-citation hallucinations do baseline summarizers produce?

**Answer:** Across 100 cases, 29 citations extracted from draft summaries. Overall HR=82.8%. Most common labels: NotFound(12), Mismatch(9), Verified(5), Ambiguous(3).

- **n_cases:** 100
- **total_citations:** 29
- **label_counts:** {'NotFound': 12, 'Mismatch': 9, 'Ambiguous': 3, 'Verified': 5}
- **label_rates:** {'NotFound': 0.4138, 'Mismatch': 0.3103, 'Ambiguous': 0.1034, 'Verified': 0.1724}
- **hallucination_rate:** 0.8276

## RQ2

**Question:** Does symbolic verification + correction reduce HR vs neural-only baseline?

**Answer:** Paired HR dropped from 82.8% (baseline) to 27.6% (symbolic correction), relative reduction 71.8%. Surviving-citation HR=47.1%. McNemar p=3.0517578125e-05.

- **hr_baseline_paired_micro:** 0.8276
- **hr_symbolic_paired_micro:** 0.2759
- **hr_symbolic_surviving_micro:** 0.4706
- **fix_rate:** 0.3103
- **removal_rate:** 0.4138
- **garbled_rate:** 0.0
- **reduction:** {'absolute_reduction': 0.14, 'relative_reduction_pct': 71.7949, 'mean_hr_draft': 0.195, 'mean_hr_final': 0.055}
- **mcnemar_test:** {'n_paired_citations': 29, 'draft_hallucinated_final_success': 16, 'draft_ok_final_bad': 0, 'draft_only_hallucinated': 16, 'final_only_hallucinated': 0, 'p_value': 3.0517578125e-05, 'interpretation': 'Significant reduction in hallucination'}

## RQ3

**Question:** Can the dual-system approach preserve quality while reducing HR by >=30%?

**Answer:** Paired HR relative reduction 71.8% (target 30.0%): PASS. ROUGE-Lsum delta -0.05 pp (tolerance ±2.0): PASS.

- **hr_relative_reduction_pct:** 71.79
- **hr_target_pct:** 30.0
- **hr_target_met:** True
- **primary_rouge_variant:** rougeLsum
- **rougeLsum_delta_pct_points:** -0.05
- **rougeL_delta_pct_points:** -0.0
- **rouge_tolerance_pct:** 2.0
- **rouge_target_met:** True
- **rougeLsum_bootstrap_ci:** {'mean_delta': -0.0005069543435954329, 'ci_low': -0.0008953451526782605, 'ci_high': -0.00019779879534818635, 'n': 100, 'ci_level': 0.95}
- **symbolic_vs_llm_rewrite:** {'hr_paired_symbolic': 0.27586206896551724, 'hr_paired_llm_rewrite': 0.27586206896551724, 'rougeLsum_symbolic': 0.40691414657879327, 'rougeLsum_llm_rewrite': 0.4054650331978944, 'rougeL_symbolic': 0.199503713754282, 'rougeL_llm_rewrite': 0.19928713634306056}

## Limitations

- Status: Full-scale evaluation — n=100 citation-rich test-split cases, D→L
- Evaluation on test split, n=100 (not full test set).
- D→L (long summaries) only; short/tiny descoped after EDA.
- Primary HR is paired draft-citation HR; surviving-citation HR reported separately.
- Primary quality metric is ROUGE-Lsum (sentence-split union-LCS), the variant conventionally reported for multi-sentence summarization; plain ROUGE-L is also reported but is not comparable to published summarization baselines.
- CourtListener access level 10 req/min, 100/hr, 250/day; caching used; multi-day runs required at full scale.
- Source truncation to the generator's context budget (known limitation).
- Manual audit template exported separately; labels not included in automated RQ answers.