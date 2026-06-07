# BridgeEngine Head-To-Head Preregistration

Last updated: 2026-06-07

## Headline Claim

Primary research question:

> On the same 100 BridgeData subset, do pi0.7-style structured annotations outperform the perceptive vision signals used in the previous LeWM experiment?

This is the controlled comparison that should close the internal research loop. Same data, same held-out split, same LeWM metric, same seed set. The only intended variable is the conditioning signal family.

## Frozen Data

Semantic source snapshot:

```text
snap_2026_05_11_1dde3edf5d
```

Current benchmark snapshot:

```text
snap_2026_05_11_1dde3edf5d_human_gold_labels
```

Current human calibration:

```text
score-reviewed clips: 100 / 100
score changes versus Gemini: 58 / 100
Gemini auto score counts: {2: 5, 4: 9, 5: 86}
Kevin calibrated score counts: {2: 6, 3: 15, 4: 30, 5: 49}
boundary-reviewed clips: 50 / 50 subset
subgoal labels: derived from reviewed subtask end_step values
```

Source Gemini reliability against Kevin's reviewed labels:

```text
quality exact agreement: 0.42
quality within-one agreement: 0.77
subtask-boundary temporal IoU mean: 0.683
derived subgoal frame agreement: 0.347
```

This is enough to treat the corrected snapshot as the current best pi0.7-side training cut, but it is not enough to claim that the uncorrected VLM labels are production-grade.

## Human Reliability Gate Before Final Run

Current state versus target:

Target thresholds before final claim:

| Metric | Target |
|---|---:|
| quality within-one agreement | >= 0.85 |
| keep/reject agreement | >= 0.90 |
| subtask-boundary temporal IoU | >= 0.70 |
| subgoal-selection agreement | >= 0.75 |

The current VLM-vs-human reliability misses the quality within-one, boundary-IoU, and subgoal-agreement targets. The applied human-gold snapshot is still valid for the pi0.7-side smoke benchmark because it uses Kevin-corrected labels. Do not treat the raw Gemini labels as final-quality labels without another prompt/rubric iteration or more human correction.

## Perceptive-Signal Gate Before Final Run

The head-to-head must not use synthetic perceptive fallbacks.

Run:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.perceptive_status --snapshot snap_2026_05_11_1dde3edf5d_human_gold_labels --require-real
```

This must pass before the perceptive comparison is valid. Each perceptive labeler needs one real payload per episode:

- `perceptive_masks`
- `perceptive_depth`
- `perceptive_tracks`

If the command reports synthetic adapters, missing payloads, or missing label rows, fix the extractor path first. Do not run or report the head-to-head on synthetic perceptive labels.

## Metric

Primary metric:

```text
held-out latent MSE from the real LeWM frozen-adapter evaluation
```

The held-out split is fixed and disjoint. At 100 episodes, the calibrated split has held-out quality distribution:

```text
{2: 1, 3: 1, 4: 1, 5: 7}
```

## Seeds

Final run seed set:

```text
0, 1, 2, 3, 4
```

Three seeds is the minimum acceptable final run. Five seeds is preferred because the current two-seed deltas are in the single-digit-percent range.

## Families

Primary families:

1. `baseline`: task instruction only.
2. `pi07_full`: rich text + metadata + subgoal, equivalent to current `rich_text_metadata_subgoal`.
3. `perceptive_all`: masks + depth + tracks.

Secondary families:

1. `rich_text`
2. `rich_text_metadata`
3. `perceptive_masks`
4. `perceptive_depth`
5. `perceptive_tracks`

## Injection Decision

Use a unified conditioning interface for the main claim.

Reason: the claim is about signal content, not about each method's native architecture. The base LeWM checkpoint, frozen modules, trainable adapter budget, held-out split, optimizer, and epoch count should stay fixed. The conditioning interface should change only in the information it receives.

Document separately if a secondary "methods as practiced" comparison is ever run with native injection paths.

## Stop Rules

- Do not change the score rubric after this preregistration unless the final result is explicitly marked as post-rubric.
- Do not change the held-out split after seeing final metrics.
- Do not report perceptive results if `perceptive_status --require-real` fails.
- Do not increase N beyond 100 without a new data and cost gate.
- Do not claim general video or human-motion validation; the current validation class is robot manipulation video.

## Current Best Pre-Final Result

Human-score and human-boundary corrected 100-episode scale curve, two seeds:

| N | baseline | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|---:|
| 25 | 0.038967 | 0.038332 | 0.036301 | 0.034922 |
| 50 | 0.035255 | 0.035348 | 0.033385 | 0.030203 |
| 100 | 0.021839 | 0.022000 | 0.021849 | 0.020468 |

At N=100, `rich_text_metadata_subgoal` is 6.28% lower mean latent MSE than baseline. This is the result to beat or falsify with the perception head-to-head and more seeds.
