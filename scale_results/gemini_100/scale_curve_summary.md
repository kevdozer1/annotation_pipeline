# Gemini 100 Scale-Curve Summary

Snapshot: `snap_2026_05_11_1dde3edf5d`

Label source: Gemini 2.5 Flash, merged from `snap_2026_05_11_de43f7bf0b_gemini50` and `snap_2026_05_11_48710ffc52` with `bridgeengine.snapshot_merge`.

Quality gate: PASS. The merged 100-episode label set has curation-quality counts `{2: 5, 4: 9, 5: 86}` and mistake counts `{false: 89, true: 11}`.

Cost: `$1.188603` for 100 episodes, or `$0.011886` per episode. At the measured rate, rough projections are `$2.38` for 200, `$11.89` for 1000, and `$713.16` for 60000 episodes. These projections assume similar episode length and Gemini pricing.

OpenAI vs Gemini on the first 50 episodes: curation exact agreement `0.440`, within-one agreement `0.860`, and keep-decision agreement `0.860`. Gemini is much cheaper but more generous, assigning `5/5` to 42 of the first 50 where OpenAI assigned `5/5` to 25.

Scale curve: real LeWM frozen-adapter evaluation, CUDA, two seeds per family, quality-stratified train mixtures, held-out count 10.

| N | baseline | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|---:|
| 25 | 0.044892 | 0.045292 | 0.041925 | 0.041327 |
| 50 | 0.022307 | 0.022268 | 0.020579 | 0.021179 |
| 100 | 0.016242 | 0.016079 | 0.016096 | 0.015647 |

Headline: richer conditioning trends better than baseline at all three sizes, with the clearest mean gap at 25 and 50 episodes. At 100 episodes, the improvement is small and overlaps two-seed noise, so this is a smoke-scale trend probe rather than a robust conclusion.

Main caveat: the held-out split contains only Gemini `5/5` episodes because the Gemini score distribution is top-heavy. Human gold labels and score calibration are the next reliability step.
