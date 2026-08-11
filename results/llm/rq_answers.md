# Research Question Answers

**Full-scale evaluation — n=100 citation-rich test-split cases, D→L, LLM zero-shot (naive truncation, larger context)**

## RQ1

**Question:** What types and frequencies of case-citation hallucinations do baseline summarizers produce?

**Answer:** Across 100 cases, 365 citations extracted from draft summaries. Overall HR=57.5%. Most common labels: Verified(155), Ambiguous(94), NotFound(58), Mismatch(58).

- **n_cases:** 100
- **total_citations:** 365
- **label_counts:** {'Ambiguous': 94, 'Verified': 155, 'NotFound': 58, 'Mismatch': 58}
- **label_rates:** {'Ambiguous': 0.2575, 'Verified': 0.4247, 'NotFound': 0.1589, 'Mismatch': 0.1589}
- **hallucination_rate:** 0.5753

## RQ2

**Question:** Does symbolic verification + correction reduce HR vs neural-only baseline?

**Answer:** Paired HR dropped from 57.5% (baseline) to 22.2% (symbolic correction), relative reduction 56.1%. Surviving-citation HR=31.0%. McNemar p=2.938735877055719e-39.

- **hr_baseline_paired_micro:** 0.5753
- **hr_symbolic_paired_micro:** 0.2219
- **hr_symbolic_surviving_micro:** 0.3095
- **fix_rate:** 0.6192
- **removal_rate:** 0.1589
- **garbled_rate:** 0.0
- **reduction:** {'absolute_reduction': 0.1605, 'relative_reduction_pct': 56.063, 'mean_hr_draft': 0.2864, 'mean_hr_final': 0.1258}
- **mcnemar_test:** {'n_paired_citations': 365, 'draft_hallucinated_final_success': 129, 'draft_ok_final_bad': 0, 'draft_only_hallucinated': 129, 'final_only_hallucinated': 0, 'p_value': 2.938735877055719e-39, 'interpretation': 'Significant reduction in hallucination'}

## RQ3

**Question:** Can the dual-system approach preserve quality while reducing HR by >=30%?

**Answer:** Paired HR relative reduction 56.1% (target 30.0%): PASS. ROUGE-Lsum delta +0.00 pp (tolerance ±2.0): PASS.

- **hr_relative_reduction_pct:** 56.06
- **hr_target_pct:** 30.0
- **hr_target_met:** True
- **primary_rouge_variant:** rougeLsum
- **rougeLsum_delta_pct_points:** 0.0
- **rougeL_delta_pct_points:** 0.03
- **rouge_tolerance_pct:** 2.0
- **rouge_target_met:** True
- **rougeLsum_bootstrap_ci:** {'mean_delta': 3.492804398548877e-05, 'ci_low': -0.000696065990224376, 'ci_high': 0.0009257463187758119, 'n': 100, 'ci_level': 0.95}
- **symbolic_vs_llm_rewrite:** {'hr_paired_symbolic': 0.2219178082191781, 'hr_paired_llm_rewrite': 0.2465753424657534, 'rougeLsum_symbolic': 0.4251790913871807, 'rougeLsum_llm_rewrite': 0.4245981714301553, 'rougeL_symbolic': 0.17820108509951552, 'rougeL_llm_rewrite': 0.17839480211258826}

## Limitations

- Status: Full-scale evaluation — n=100 citation-rich test-split cases, D→L, LLM zero-shot (naive truncation, larger context)
- Evaluation on test split, n=100 (not full test set).
- D→L (long summaries) only; short/tiny descoped after EDA.
- Primary HR is paired draft-citation HR; surviving-citation HR reported separately.
- Primary quality metric is ROUGE-Lsum (sentence-split union-LCS), the variant conventionally reported for multi-sentence summarization; plain ROUGE-L is also reported but is not comparable to published summarization baselines.
- CourtListener access level 10 req/min, 100/hr, 250/day; caching used; multi-day runs required at full scale.
- Source truncation to the generator's context budget (known limitation).
- Manual audit template exported separately; labels not included in automated RQ answers.