# Human-Calibrated 100-Episode Scale-Curve Summary

Snapshot: `snap_2026_05_11_1dde3edf5d_human_calibrated`

Source snapshot: `snap_2026_05_11_1dde3edf5d`

Human review status:

- reviewed episodes: `100 / 100`
- score changes versus Gemini auto curation: `58 / 100`
- review notes entered: `0`
- subtask-boundary accept toggles: none
- subgoal accept toggles: none
- calibration reasons: GUI-default reason text was saved, but Kevin did not intentionally enter free-form reasons

Score distributions:

| Source | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Gemini auto curation | 0 | 5 | 0 | 9 | 86 |
| Kevin human calibration | 0 | 6 | 15 | 30 | 49 |

Agreement before applying gold scores:

| Metric | Value |
|---|---:|
| quality exact agreement | 0.42 |
| quality within-one agreement | 0.77 |
| keep/reject agreement | 0.76 |
| subtask-boundary IoU | n/a, not reviewed |
| subgoal agreement | n/a, not reviewed |

Quality gate on calibrated snapshot:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 6, 3: 15, 4: 30, 5: 49}
```

Scale curve: real LeWM frozen-adapter evaluation, CUDA, two seeds per family, quality-stratified train mixtures, held-out count 10. The held-out split is now mixed quality: `{2: 1, 3: 1, 4: 1, 5: 7}`.

| N | baseline | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|---:|
| 25 | 0.042079 | 0.041079 | 0.039682 | 0.038022 |
| 50 | 0.031014 | 0.030213 | 0.028289 | 0.027482 |
| 100 | 0.022522 | 0.023123 | 0.022831 | 0.020807 |

Delta versus baseline, lower is better:

| N | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|
| 25 | -2.38% | -5.70% | -9.64% |
| 50 | -2.58% | -8.79% | -11.39% |
| 100 | +2.67% | +1.37% | -7.61% |

Headline: after human score calibration, the metadata+subgoal family beats baseline at all three sizes, including a `-7.61%` mean latent-MSE delta at N=100. Text-only and metadata-only help at 25/50 but regress slightly at 100. This strengthens the result relative to the Gemini-only run, while still remaining a two-seed, 100-episode smoke-scale ablation.
