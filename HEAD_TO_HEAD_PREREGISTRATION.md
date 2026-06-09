# BridgeEngine vs LeWM Head-To-Head Preregistration

Last updated: 2026-06-07

## Headline Claim

Primary research question:

> On the same 100 BridgeData subset, does the pi0.7 class of structured annotations outperform the perceptive CV signals from Kevin's previous LeWM experiment?

This is an annotation-strategy comparison, not a pure signal-content comparison. The mechanisms differ:

- LeWM perceptive signals are auxiliary prediction targets through native aux heads.
- BridgeEngine pi0.7 signals are conditioning inputs through the BridgeEngine adapter path.

Every report, figure, paper section, and resume bullet needs to state that distinction plainly.

## Required References Read

LeWM testbed:

```text
C:\Users\Kevin\projects\LeWM_testbed\README.md
C:\Users\Kevin\projects\pipeline\kevdozer1.github.io\blog\2026\lewm-finetune\index.html
C:\Users\Kevin\projects\LeWM_testbed\configs\finetune\fullscale_A_baseline.yaml
C:\Users\Kevin\projects\LeWM_testbed\configs\finetune\fullscale_B_depth.yaml
C:\Users\Kevin\projects\LeWM_testbed\configs\finetune\fullscale_D_tracks.yaml
C:\Users\Kevin\projects\LeWM_testbed\configs\finetune\fullscale_E_depth_tracks.yaml
```

Confirmed LeWM fullscale training config:

```text
optimizer: AdamW
learning rate: 5e-5
batch size: 16
auxiliary loss weight: 0.1
epochs: 20
precision: bf16-mixed
metric: held-out next-latent MSE
```

Confirmed fullscale LeWM CV conditions:

| condition | signal | mechanism |
|---|---|---|
| A | baseline | no aux head |
| B | Video-Depth-Anything depth | aux head predicts 56x56 depth |
| D | CoTracker3 tracks | aux head predicts 400x2 point displacement |
| E | depth + tracks | two aux heads, each weight 0.1 |

Pilot-only LeWM conditions F centroid and G shape exist, and masks exist in the pilot HDF5, but the shared 100-episode HDF5 currently contains only `depth`, `tracks`, `track_visibility`, `contact`, pixels, actions, and observations. The first valid 100-episode head-to-head should therefore use A/B/D/E plus BridgeEngine pi0.7 conditions.

## Frozen Data

BridgeEngine semantic snapshot:

```text
snap_2026_05_11_1dde3edf5d_human_gold_labels
```

LeWM manifest:

```text
D:\bridgedata_v2_subset\manifest_100.json
```

LeWM HDF5 cache:

```text
D:\bridgedata_v2_subset\datasets\bridgedata_v2_100ep.h5
```

The BridgeEngine human-gold snapshot and LeWM `manifest_100.json` contain the same 100 episode IDs exactly. First ten shared IDs:

```text
episode_000352
episode_000392
episode_000425
episode_001146
episode_001860
episode_001972
episode_003087
episode_004196
episode_004417
episode_004558
```

## Human Calibration State

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

The corrected snapshot is valid for the pi0.7-side smoke benchmark because Kevin-corrected labels are applied. It does not prove the raw Gemini labels are production-grade.

## Fixed Split

Generated plan:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.head_to_head --snapshot snap_2026_05_11_1dde3edf5d_human_gold_labels --output-dir head_to_head_results\preregistered_100 --sizes 25 50 100 --heldout-count 10 --split-seed 0 --train-seeds 42 137 256
```

Frozen split files:

```text
head_to_head_results/preregistered_100/splits/scale_25_split.json
head_to_head_results/preregistered_100/splits/scale_50_split.json
head_to_head_results/preregistered_100/splits/scale_100_split.json
```

Held-out count:

```text
10 episodes
```

Held-out quality distribution in the generated preregistered split:

```text
{4: 4, 5: 6}
```

Training pools are nested across N and quality-stratified from Kevin-calibrated scores.

## Seeds And Scales

Frozen training seeds:

```text
42, 137, 256
```

These match Kevin's cached LeWM fullscale A/B/D/E runs. Three seeds is the minimum final run. Five seeds is still better statistically, but it would require two additional LeWM aux-head seeds per CV condition.

Frozen scales:

```text
25, 50, 100
```

Do not increase N beyond 100 without downloading more BridgeData and opening a new data/cost gate.

## Conditions

Planned shared-axis figure conditions:

| condition | paradigm | mechanism |
|---|---|---|
| `baseline` | shared reference | no aux prediction target and no pi0.7 conditioning |
| `cv_B_depth_aux` | LeWM perceptive | depth as aux prediction target |
| `cv_D_tracks_aux` | LeWM perceptive | tracks as aux prediction target |
| `cv_E_depth_tracks_aux` | LeWM perceptive | depth + tracks as aux prediction targets |
| `pi07_rich_text` | BridgeEngine pi0.7 | subtask text conditioning |
| `pi07_rich_text_metadata` | BridgeEngine pi0.7 | subtask text + metadata conditioning |
| `pi07_full_metadata_subgoal` | BridgeEngine pi0.7 | subtask text + metadata + subgoal-frame conditioning |

Current POC does not include an isolated "subgoal only" family. The subgoal condition is the full pi0.7 stack.

## Metric

Primary metric:

```text
held-out next-latent MSE
```

All plotted conditions must be evaluated on the same fixed split. Existing LeWM cached training runs are useful as runtime anchors, but their checkpoints were trained/evaluated with per-seed random 90/10 splits, so their outputs cannot be mixed with BridgeEngine fixed-split outputs on one figure. The handoff runner retrains from the pretrained LeWM checkpoint on split-specific HDF5 datasets and evaluates with `bridgeengine.benchmark.lewm_fixed_eval`.

## Runnable Handoff

Full 100-episode handoff command:

```powershell
.\scripts\run_head_to_head_100.ps1
```

Prepare-only smoke/inspection command:

```powershell
.\scripts\run_head_to_head_100.ps1 -PrepareOnly
```

The handoff script verifies signal files, creates split-specific train and held-out HDF5 files, writes LeWM configs, retrains CV aux conditions from the pretrained LeWM checkpoint, evaluates on the explicit fixed held-out HDF5, then runs the BridgeEngine pi0.7 scale curve with matched `20` epochs, batch `16`, and lr `5e-5`.

Default output is on the SSD:

```text
D:\lewm_runs\bridgeengine_head_to_head\run_100
```

Do not run the full grid into the repo-local `head_to_head_results\run_100` directory on `C:`. The native LeWM trainer writes checkpoints during training and C: does not have enough free space for the grid.

Do not evaluate the old cached random-split LeWM checkpoints on the fixed held-out set. They may overlap the preregistered held-out episodes and are kept only as runtime anchors.

## Runtime Estimate

Generated estimate, from cached historical runtime logs and the current pi0.7 run timings:

```text
LeWM CV fullscale cached N=100 training: 1.660 hours
LeWM CV all scales from scratch estimate: 2.904 hours
LeWM CV incremental estimate reusing cached N=100: 1.245 hours
BridgeEngine pi0.7 3-seed estimate: 0.991 hours
Total from scratch estimate: 3.896 hours
Total incremental estimate reusing cached CV N=100: 2.236 hours
```

Source:

```text
head_to_head_results/preregistered_100/head_to_head_plan.json
head_to_head_results/preregistered_100/runtime_estimate.md
```

The fixed-split evaluator and handoff runner now exist. The estimate still excludes small evaluation overhead and any retry time from CUDA or dependency failures.

## Stop Rules

- Do not change the score rubric after this preregistration unless the final result is explicitly marked as post-rubric.
- Do not change the held-out split after seeing final metrics.
- Do not plot old LeWM CV numbers and BridgeEngine pi0.7 numbers together unless they were evaluated on the same fixed split.
- Do not report masks, centroids, or shape at N=100 until those payloads are extracted or exported for the shared 100.
- Do not claim general video or human-motion validation; the current validation class is robot manipulation video.

## Current Best Pre-Final Result

Human-score and human-boundary corrected BridgeEngine pi0.7 scale curve, two seeds:

| N | baseline | rich_text | rich_text_metadata | rich_text_metadata_subgoal |
|---:|---:|---:|---:|---:|
| 25 | 0.038967 | 0.038332 | 0.036301 | 0.034922 |
| 50 | 0.035255 | 0.035348 | 0.033385 | 0.030203 |
| 100 | 0.021839 | 0.022000 | 0.021849 | 0.020468 |

At N=100, `rich_text_metadata_subgoal` is 6.28% lower mean latent MSE than the current BridgeEngine baseline. This is not yet the final perceptive-vs-pi0.7 head-to-head result.

## Amendments

### 2026-06-09 — Eval-validity analysis (additive, eval-only)

An external review noted that every condition finetunes the full LeWM model, so each
condition's held-out next-latent MSE is measured in its own latent geometry; a condition
can lower raw MSE by contracting target-latent variance rather than predicting better.
This amendment adds analyses to test that confound. It is strictly additive: no split,
seed, rubric, or trained main-grid checkpoint was changed. Only eval-time re-scoring and a
small auxiliary probe (trained on top of frozen world-model latents) were added.

1. **Normalized paired metric.** Both fixed-split evaluators
   (`bridgeengine.benchmark.lewm_fixed_eval` and `bridgeengine.benchmark.pi07_fixed`) now
   also record per run the held-out mean per-dimension target-latent variance
   (`heldout_target_variance`) and mean squared target-latent norm
   (`heldout_target_mean_sq_norm`) in `fixed_eval.json`, and a per-window
   `norm_sq_err = sq_err / heldout_target_variance` column in `fixed_eval_windows.csv`.
   The paired analysis (`bridgeengine.benchmark.leak_power`) reports both raw and
   variance-normalized paired deltas and a per-condition target-variance table. Completed
   scale-25/50 cells were re-evaluated (eval-only) from their existing checkpoints to
   populate these fields; raw `latent_mse` values are unchanged (deterministic re-eval).

2. **Relative subgoal-leak bins.** The boundary-distance leak audit adds
   `near_advantage_rel` / `far_advantage_rel` (bin advantage divided by mean P0 squared
   error in the same bin) and a distance-vs-relative-advantage correlation. Stated
   limitation: absence of near-boundary concentration rules out crude target-copying, not
   privileged-future-information from a same-episode subgoal frame.

3. **IDM action-decoding probe** (`bridgeengine.benchmark.idm`). A 2-layer MLP
   `f(z_t, z_{t+1}) -> a_t` is trained per condition-seed on train-split encoded latents,
   then scored on held-out windows using the model's predicted latents. Per-window action
   MSE is logged in the `fixed_eval_windows.csv` schema (`idm_eval_windows.csv`) so the
   existing paired machinery applies. Because the metric is in physical action units it is
   immune to the latent-variance confound. Scope: scales 25 and 50 (the scales with a
   complete checkpoint set); scale 100 remains excluded until its grid is complete.

Results live in `LEAK_AND_POWER_REPORT.md` and the generated CSVs under the run output
directory. No new training conditions (retrieval/generated subgoal, real text encoder)
were started in this pass.
