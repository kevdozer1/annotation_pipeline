# Technical Report: Annotation Strategies for LeWM World-Model Finetuning on BridgeData V2

Repository: `annotation_pipeline`, branch `fix/head-to-head-npy-dataprep`. This report covers the head-to-head grid through commit `96a3f0f` plus the eval-only diagnostics added in the same branch immediately afterward (`bridgeengine/benchmark/diagnostics.py`). Every number below is sourced to a file path, a function, or a generated artifact under the run directory `D:\lewm_runs\bridgeengine_head_to_head\run_100\` (hereafter `RUN/`). No training was run for this report; no split, seed, rubric, or checkpoint was changed.

---

## 1. Abstract

We compared two families of annotation strategy on the *same* small JEPA world model (LeWM) and the *same* 100 episodes of BridgeData V2, asking whether structured "pi0.7" annotations used as conditioning inputs outperform computer-vision (CV) signals used as auxiliary prediction targets, measured by held-out next-latent MSE on one frozen split, at scales N = 25 and 50 with three seeds. The complete N = 100 cell was not finished and is excluded.

The scoped finding is narrow and mostly negative. Among the CV auxiliary-target conditions, depth and track supervision produce a small but normalization-robust reduction in held-out next-latent MSE relative to the shared native baseline: `E_depth_tracks` separates from zero on both the raw and the variance-normalized paired metric at N = 25 and N = 50, and `B_depth`/`D_tracks` separate at N = 50 (`RUN/paired_window_summary.csv`). Among the pi0.7 conditioning conditions, none survives the validity controls: the largest apparent effect, a −3.4% raw win for the full pi0.7 stack `P4_pi07_full_stack` at N = 50, is **not** a prediction gain — it is manufactured by a 0.97× contraction of that run's target-latent variance and vanishes under normalization (mean normalized delta −0.55%, CI crosses zero).

The protocol caught two artifacts that would otherwise have been reported as wins. First, the variance-contraction artifact just described: because every condition finetunes the full encoder, each condition's MSE lives in its own latent geometry, and a 3% shrinkage of target variance buys a 3% "win" with no improvement in prediction (Section 5–6). Second, and more important for interpreting *all* of these effect sizes, the diagnostics show the underlying task is close to degenerate. On held-out windows a trivial copy baseline `ẑ_{t+1} = z_t` beats the trained predictor for every condition (trained/copy ratio 1.35–1.62, `RUN/diagnostics_trivial_floor.csv`), 60% of held-out windows are near-static (frame-to-frame latent displacement < 0.005, `RUN/diagnostics_motion_bins.csv`), and the entire depth/track advantage is concentrated in the highest-motion quintile. The conditioning channel that the pi0.7 conditions ride on is small (3–10% of the context-embedding norm) and low-cardinality (28 distinct conditioning vectors over 281 held-out windows), and the no-content adapter-null control actually produces a *larger* perturbation than the content adapters — so the annotation content was, in this configuration, near-negligible even in principle (Section 7). Finally, training is severely overfit: train MSE is 8–14× below held-out MSE for every condition (`RUN/diagnostics_training_dynamics.csv`), which is unsurprising at 18.7M trainable parameters on 285–812 training windows.

In one sentence: on this instrument and this task, the depth/track auxiliary signal acts mainly as a mild regularizer that helps in the minority of high-motion windows, the pi0.7 conditioning channel was too small and too low-cardinality to move the metric, and the headline reason effect sizes are small is that one-step latent prediction over 5 Hz kitchen video is nearly solved by copying.

---

## 2. Research question and framing

The primary question, stated in `HEAD_TO_HEAD_PREREGISTRATION.md`: *on the same 100-episode BridgeData subset, does the pi0.7 class of structured annotations outperform the perceptive CV signals from the prior LeWM experiment?* This is framed deliberately as an **annotation-strategy** comparison, not a pure signal-content comparison, because the two paradigms inject information through different mechanisms and each mechanism is the one that is faithful to its paradigm.

The CV signals (depth from Video-Depth-Anything, point tracks from CoTracker3) are used as **auxiliary prediction targets**: a small head decodes the per-frame latent and is trained to predict the signal, adding a loss term that shapes the representation (`C:\Users\Kevin\projects\LeWM_testbed\src\lewm_testbed\auxiliary\heads.py`). The pi0.7 signals (subtask text, episode metadata, subgoal keyframes) are used as **conditioning inputs**: a learned adapter maps an annotation feature vector into the latent space and adds it to the context embedding before the predictor runs (`bridgeengine/benchmark/pi07_fixed.py`, `pi07_native_forward`). The mechanism distinction is a design choice, not a confound to be eliminated — depth is naturally a dense per-pixel target and would be awkward as a conditioning vector, while a subtask label is naturally a conditioning input and would be awkward as a per-frame regression target. The consequence, which Section 8 quantifies, is that the comparison conflates *annotation strategy* with *supervisory bandwidth*: a per-frame depth map carries roughly four orders of magnitude more supervised scalars per episode than a per-segment text label. The report does not pretend this is controlled; it names and measures it.

Both paradigms are evaluated through a single shared evaluator on the identical fixed held-out set so that the metric definition cannot differ between them (Section 5).

---

## 3. Model card

**Architecture (`config.json` in the pretrained snapshot, `D:\hf_cache\models--quentinll--lewm-cube\snapshots\7d05e023b3c1114cc8e803ec23fb0177d688598b\config.json`).** LeWM is a JEPA-style latent world model. The encoder is a from-scratch ViT-tiny (`stable_pretraining.backbone.utils.vit_hf`, size `tiny`, patch 14, image 224, not ImageNet-pretrained), giving a 16×16 = 256-patch grid at hidden width 192. The predictor is a 6-layer transformer (16 heads, MLP dim 2048, dim-head 64, dropout 0.1) over 3 frames at width 192. The action encoder is a linear `Embedder` (config input_dim 25, reinitialized to 7 for BridgeData's 7-D actions, `bridgeengine/benchmark/lewm_fixed_eval.py` and `pi07_fixed.py` both rebuild it). The projector and prediction projector are MLPs 192→2048→192 with BatchNorm1d.

**The CLS bottleneck and its consequence.** The single most important architectural fact for interpreting the results is in `stable_worldmodel/wm/lewm/lewm.py`, `LeWM.encode`: after the ViT runs, `pixels_emb = output.last_hidden_state[:, 0]` — only the CLS token is kept, then projected. All 256 patch tokens collapse to one 192-D vector per frame before anything else happens. The predictor and every auxiliary head see only this 192-D per-frame summary; spatial structure (where the depth gradient is, which points moved) must squeeze through that single vector. A depth head that must reconstruct a 56×56 map from a 192-D CLS vector, and a track head that must predict 400 point displacements from the same 192-D vector, are both fighting the bottleneck. This caps how much spatial supervision can do and is the architectural reason a prior LeWM note flagged the CLS-only design as a limiter.

**Prediction and metric.** `LeWM.predict(emb, act_emb)` runs the predictor then `pred_proj`. With history 3 and one prediction step, a 4-frame window yields context latents `emb[:, :3]` and targets `emb[:, 1:4]`; the per-window error is `mean over (3 timesteps, 192 dims) of (pred − tgt)²` (`bridgeengine/benchmark/lewm_fixed_eval.py:129`, `pi07_fixed.py:738`). SIGReg (`stable_worldmodel.wm.loss.SIGReg`, knots 17, 1024 projections, weight 0.09) is an isotropic-Gaussianization regularizer added during training and reported but not part of the headline metric.

**Pretraining provenance.** The base checkpoint is `quentinll/lewm-cube` (HF), a LeWM pretrained on a cube-manipulation setting, loaded via `stable_worldmodel.wm.utils.load_pretrained`. It is not pretrained on BridgeData; every condition finetunes it.

**Finetuning recipe and what is matched.** All conditions finetune the full model (`freeze: none`) for 20 epochs, batch 16, AdamW lr 5e-5, weight decay 1e-3, LinearWarmupCosineAnnealingLR, bf16-mixed, SIGReg weight 0.09 (`bridgeengine/benchmark/head_to_head_runner.py` `_lewm_config`; `pi07_fixed.py` `train_pi07_cell`). The CV path runs through `LeWM_testbed/scripts/finetune_with_aux.py`; the pi0.7 path runs through `pi07_fixed.py` with the same optimizer/scheduler/precision and the same SIGReg term. Both paths take an internal 90/10 random split of the training windows for Lightning's val loop (`train_samples` 285 / `val_samples` 31 at N = 25, from any `RUN/runs/scale_25/*/metadata.json`); that internal split is for early visibility only and does not touch the frozen held-out set. The recipe is therefore matched across paradigms except for the mechanism itself. Two controls bound the residual: the **P0 native baseline** is the exact same native trainer with no aux head and no conditioning (so CV-vs-P0 and pi0.7-vs-P0 share a baseline), and **P_adapter_null** runs the pi0.7 adapter with its feature vector zeroed, isolating adapter overhead from annotation content. Parameter counts: CV `AuxiliaryLeWM` totals 18,684,833 trainable (e.g. the depth head adds 650,385), all trainable (`RUN/runs/scale_25/B_depth_seed42/metadata.json`); the pi0.7 model is 18,042,642 base + 138,240 conditioner.

---

## 4. Data card

**Episodes.** 100 BridgeData V2 episodes (`manifest_100.json`), kitchen tabletop manipulation. Across the 100 episodes there are 55 distinct language instructions with at most 3 episodes per instruction, almost all of the form "put X on/in Y" (e.g. "put carrot on cutting board", "put broccoli in pot"); extracted from each episode's `metadata.json` `task` field under `D:\bridgedata_v2_subset\episodes\`. Frames are stored 256×256×3 and resized to 224 for the model (`head_to_head_runner.py` `_resize_frames`). Actions and observations are 7-D. Frame rate is not recorded in the local metadata (only `n_frames`); BridgeData V2 is nominally ~5 Hz, which is consistent with the short episode lengths but unverified from these files.

**Lengths and window counts.** Episode lengths (frames) for the 90-episode N=100 training pool: mean 29.0, min 13, max 78; for the 10-episode held-out set: mean 31.1, min 18, max 48 (read from `ep_len` in `RUN/datasets/be_h2h_scale_*_*.h5`). With `num_steps = 4` (history 3 + 1 prediction) and frameskip 1, window counts are: train 316 (N=25), 902 (N=50), 2339 (N=100); held-out **281 windows, constant across N** because the 10 held-out episodes are fixed. Train episode counts are 15 / 40 / 90; held-out is always the same 10. The held-out set is therefore small in two senses simultaneously — 10 episodes and 281 windows — which is the dominant source of CI width (Section 8).

**Annotation provenance.** Subtask segmentation, episode metadata, and subgoal frames originate from Gemini 2.5 Flash in a two-stage pipeline, then human-calibrated; the frozen snapshot is `snap_2026_05_11_1dde3edf5d_human_gold_labels`. The calibration numbers (`HEAD_TO_HEAD_PREREGISTRATION.md`, "Human Calibration State") are not reassuring about the raw VLM labels: of 100 reviewed clips, **58 had their quality score changed** by the human reviewer; quality exact agreement against the human is **0.42**, within-one **0.77**; subtask-boundary temporal IoU mean **0.683**; derived subgoal-frame agreement **0.347**. The snapshot used here applies the human corrections, so the pi0.7 conditions are evaluated on human-calibrated labels — but the 0.347 subgoal-frame agreement is a direct warning that the *automatic* subgoal channel, the most distinctive pi0.7 signal, is barely better than chance at picking the same frame a human would, and the corrected version is doing a lot of the work.

**What the hashed-text representation actually encodes.** The pi0.7 conditioning feature is **not** a semantic embedding. `bridgeengine/benchmark/train_lewm.py` `_hash_text` tokenizes the prompt with a regex (`[a-z0-9_]+`), hashes each token with SHA-256 into one of 128 bins, adds ±1 with a sign bit, and L2-normalizes by √(token count). It is a signed bag-of-hashed-tokens with collisions and no word order, no synonymy, no notion that "carrot" and "vegetable" are related. The metadata vector (`_metadata_vector`, 8-D) holds speed/100, quality/5, a mistake bit, two control-mode one-hots, and three presence flags. The full feature is 136-D (`FEATURE_DIM = 128 + 8`). So "subtask text conditioning" here means "which hash bins the words of a short instruction fall into," a representation that caps how much linguistic structure can possibly transfer (Section 8).

---

## 5. Protocol and the audit chain

**Preregistration and freezing.** Scales 25/50/100, seeds 42/137/256, one fixed held-out split of 10 episodes generated with split-seed 0 and quality-stratified nesting (`head_to_head_results/preregistered_100/splits/scale_{25,50,100}_split.json`; `split_id` e.g. `scale_25_aa40ccba39`). The held-out 10 are identical across N (verified: `scale_25` heldout == `scale_100` heldout) and the training pools are nested. The metric is held-out next-latent MSE through one evaluator (`bridgeengine/benchmark/lewm_fixed_eval.py`, which dispatches pi0.7 runs to `pi07_fixed.evaluate_pi07_run`); both paths compute the identical per-window quantity and write it to `fixed_eval_windows.csv` via `window_eval.write_fixed_eval_windows`.

**Statistics.** Effects are paired per window (same held-out window, condition vs P0) and aggregated with a cluster bootstrap whose resampling unit is the (seed, episode) pair, not the individual window, because adjacent windows within an episode are strongly correlated (`leak_power.py` `_paired_summary`, `_bootstrap_ci`, 2000 reps). This is deliberately conservative and is the main reason most CIs are wide.

**The audit chain, in the order it actually unfolded.** (1) The raw paired table shows `P4_pi07_full_stack` beating P0 at N=50 by −3.4% with a CI clear of zero (`RUN/paired_window_summary.csv`) — an apparent pi0.7 win. (2) The subgoal-leak audit asks whether that win is a same-episode-subgoal oracle effect concentrated near segment boundaries; it is not concentrated near boundaries (Section 6), which rules out crude target-copying but, as stated plainly in `LEAK_AND_POWER_REPORT.md`, does not rule out privileged future information. (3) Variance normalization then dissolves the win: every run's per-window error is divided by that run's own mean per-dimension target-latent variance (`window_eval.TargetLatentMoments`, `heldout_target_variance` in each `fixed_eval.json`), and under this leakage-robust metric P4's delta collapses to −0.55% with a CI crossing zero. The mechanism is visible in `RUN/target_variance_table.csv`: P4 at N=50 has target-variance ratio **0.971** versus P0 — a 2.9% contraction of the latent space that mechanically produces ~2.9% of a "win" with no change in predictive accuracy. A 0.97 variance ratio manufactured a 3.4% win. (4) The IDM action-decoding probe (Section 6) then asks whether *any* condition improves prediction in physical action units, which are immune to latent rescaling; none separates from zero.

The point of recording the chain is that three of the four steps are guards against false positives, and they fired.

---

## 6. Results

**Raw and normalized paired deltas vs P0 (`RUN/paired_window_summary.csv`).** Negative is better; "sig" means the bootstrap CI excludes zero.

| scale | condition | raw Δ | raw % | raw sig | norm Δ | norm % | norm sig |
|---|---|---|---|---|---|---|---|
| 25 | E_depth_tracks | −0.00514 | −10.4% | yes | −0.00482 | −6.9% | **yes** |
| 25 | B_depth | −0.00396 | −8.1% | yes | −0.00334 | −4.8% | no |
| 25 | P4_pi07_full_stack | −0.00044 | −0.9% | no | +0.00026 | +0.4% | no |
| 25 | P3_pi07_subgoal | −0.00039 | −0.8% | no | +0.00028 | +0.4% | no |
| 25 | D_tracks | −0.00001 | −0.02% | no | −0.00003 | −0.04% | no |
| 50 | E_depth_tracks | −0.00572 | −16.5% | yes | −0.00714 | −16.2% | **yes** |
| 50 | B_depth | −0.00533 | −15.4% | yes | −0.00720 | −16.3% | **yes** |
| 50 | D_tracks | −0.00047 | −1.4% | yes | −0.00080 | −1.8% | **yes** |
| 50 | P4_pi07_full_stack | −0.00118 | −3.4% | **yes (raw)** | −0.00024 | −0.55% | **no** |
| 50 | P3_pi07_subgoal | −0.00098 | −2.8% | no | −0.00007 | −0.17% | no |

The depth/track CV effect is real in the sense that it survives variance normalization (E at both scales, B and D at N=50). The single pi0.7 effect that reached raw significance (P4 at N=50) does not survive. Note also that the CV depth wins grow from N=25 to N=50 rather than washing out — the opposite of the originally hypothesized "CV helps small-N then washes out" story — though with N=100 incomplete this cannot be extended.

**Target-variance table (`RUN/target_variance_table.csv`).** No condition trips the >10% contraction flag. The depth conditions contract variance only mildly and inconsistently: B/E variance ratios vs the A_baseline native anchor are 0.964/0.960 at N=25 but 1.006/0.992 at N=50 (B actually expands). The pi0.7 subgoal conditions are the most contracted (P3/P4 ≈ 0.97–0.99), which is exactly why their small raw deltas shrink under normalization. The full table (mean per-dim target-latent variance, and ratio vs the two anchors A_baseline and P0):

| scale | condition | mean target variance | var ratio vs A_baseline | var ratio vs P0 |
|---|---|---|---|---|
| 25 | A_baseline | 0.7156 | 1.000 | 0.999 |
| 25 | B_depth | 0.6897 | 0.964 | 0.963 |
| 25 | D_tracks | 0.7158 | 1.000 | 1.000 |
| 25 | E_depth_tracks | 0.6868 | 0.960 | 0.959 |
| 25 | P0_pi07_baseline | 0.7160 | 1.001 | 1.000 |
| 25 | P3_pi07_subgoal | 0.7078 | 0.989 | 0.989 |
| 25 | P4_pi07_full_stack | 0.7073 | 0.989 | 0.988 |
| 50 | A_baseline | 0.7925 | 1.000 | 1.002 |
| 50 | B_depth | 0.7972 | 1.006 | 1.008 |
| 50 | D_tracks | 0.7942 | 1.002 | 1.004 |
| 50 | E_depth_tracks | 0.7862 | 0.992 | 0.994 |
| 50 | P0_pi07_baseline | 0.7907 | 0.998 | 1.000 |
| 50 | P3_pi07_subgoal | 0.7698 | 0.971 | 0.974 |
| 50 | P4_pi07_full_stack | 0.7682 | 0.969 | 0.972 |

The receipt for the audit is the P4 row at N=50: variance ratio 0.972 vs P0, i.e. a 2.8% contraction, against a raw paired win of 3.4% — almost the entire raw win is the contraction, and indeed the normalized delta is −0.55% and non-significant. By contrast the depth conditions at N=50 have ratios at or *above* 1.0, so their raw wins cannot be variance artifacts and the normalized deltas are as large or larger than the raw ones. The table is the receipt that the depth result is not a variance artifact while the pi0.7 P4 result substantially is.

**Subgoal-leak audit, absolute and relative (`RUN/subgoal_leak_bins.csv`).** For P3 and P4, the subgoal advantage over P0 is *larger far from the segment boundary* than near it: at N=50, `near_minus_far_advantage_rel` is −0.042 (P3) and −0.016 (P4), and the distance-vs-advantage correlation is small and positive (0.10–0.12 absolute, 0.045–0.047 on relative advantage). A target-copying leak would concentrate advantage at the boundary, where the subgoal frame is closest to the predicted frame; it does not. The honest reading, already written into `LEAK_AND_POWER_REPORT.md`, is that this rules out crude target-copying but not privileged-future-information: a same-episode subgoal is drawn from the future of the very trajectory being predicted and can lower error uniformly rather than only at the boundary.

**IDM action-decoding probe (`RUN/idm_paired_summary.csv`).** A 2-layer MLP `f(z_t, z_{t+1}) → a_t` trained per condition-seed on train-split encoded latents and scored on held-out windows using the model's predicted latents (`bridgeengine/benchmark/idm.py`). In action units, **no condition's paired CI separates from zero** at either scale. Direction is informative but underpowered: at N=50 only `E_depth_tracks` (−6.7%) and `D_tracks` (−3.7%) sit below P0 in mean action error while all pi0.7 families sit above it, consistent with motion-style CV targets shaping action-decodable latents and pi0.7 conditioning not doing so. The CIs are much wider than the latent-MSE CIs because each condition-seed trains its own probe in its own geometry; three seeds cannot resolve these gaps.

**Adapter-null control (`RUN/paired_window_summary.csv`).** `P_adapter_null` is statistically indistinguishable from P0 raw and normalized at both scales (raw Δ +4.0e-5 at N=25, −4.7e-5 at N=50; neither significant). Adapter overhead is not creating or masking effects.

---

## 7. Diagnostics

These four eval-only diagnostics (`bridgeengine/benchmark/diagnostics.py`, run over all 60 scale-25/50 cells) are the substance of why the effect sizes are small.

**7.1 Trivial-baseline floor (`RUN/diagnostics_trivial_floor.csv`).** For each run we computed, on the held-out windows, the per-window MSE of three predictors of `z_{t+1}`: copy (`ẑ = z_t`), constant (`ẑ = mean of held-out targets`), and the trained predictor. The result is uniform and damning across all ten conditions:

| scale | condition | trained MSE | copy MSE | constant MSE | trained/copy | R² vs constant |
|---|---|---|---|---|---|---|
| 25 | A_baseline | 0.0491 | 0.0320 | 0.716 | 1.541 | 0.931 |
| 25 | B_depth | 0.0453 | 0.0281 | 0.690 | 1.619 | 0.934 |
| 25 | D_tracks | 0.0492 | 0.0320 | 0.716 | 1.547 | 0.931 |
| 25 | E_depth_tracks | 0.0441 | 0.0272 | 0.687 | 1.625 | 0.935 |
| 25 | P0_pi07_baseline | 0.0492 | 0.0321 | 0.716 | 1.540 | 0.931 |
| 25 | P1_pi07_subtask_text | 0.0493 | 0.0320 | 0.716 | 1.546 | 0.931 |
| 25 | P2_pi07_metadata | 0.0492 | 0.0321 | 0.716 | 1.542 | 0.931 |
| 25 | P3_pi07_subgoal | 0.0489 | 0.0318 | 0.708 | 1.543 | 0.930 |
| 25 | P4_pi07_full_stack | 0.0488 | 0.0319 | 0.707 | 1.537 | 0.930 |
| 25 | P_adapter_null | 0.0493 | 0.0322 | 0.716 | 1.538 | 0.931 |
| 50 | A_baseline | 0.0345 | 0.0259 | 0.793 | 1.365 | 0.956 |
| 50 | B_depth | 0.0294 | 0.0207 | 0.797 | 1.457 | 0.963 |
| 50 | D_tracks | 0.0342 | 0.0259 | 0.794 | 1.352 | 0.957 |
| 50 | E_depth_tracks | 0.0290 | 0.0207 | 0.786 | 1.424 | 0.963 |
| 50 | P0_pi07_baseline | 0.0347 | 0.0263 | 0.791 | 1.354 | 0.956 |
| 50 | P1_pi07_subtask_text | 0.0350 | 0.0263 | 0.791 | 1.362 | 0.955 |
| 50 | P2_pi07_metadata | 0.0347 | 0.0262 | 0.791 | 1.357 | 0.956 |
| 50 | P3_pi07_subgoal | 0.0337 | 0.0255 | 0.770 | 1.355 | 0.956 |
| 50 | P4_pi07_full_stack | 0.0335 | 0.0255 | 0.768 | 1.351 | 0.956 |
| 50 | P_adapter_null | 0.0347 | 0.0263 | 0.791 | 1.353 | 0.956 |

Two facts. First, **the constant-mean baseline already achieves R² ≈ 0.93–0.96** — i.e. the trained predictor explains only the last 4–7% of target variance that simply predicting the average latent does not. Second, and worse, **the copy baseline beats the trained predictor for every one of the ten conditions**: trained/copy ranges 1.35–1.62, so copying the previous latent is 26–62% *more accurate* than the model's learned one-step prediction on held-out windows. The copy baseline's own R² is 0.955 (N=25) and 0.967 (N=50) — above the trained model's 0.93/0.96. Plainly: doing nothing solves ~95% of the one-step latent-prediction problem, and the trained predictor solves slightly less than doing nothing. The condition contrasts of a few percent of held-out MSE are happening *inside the gap between a worse-than-copy predictor and a near-optimal copy*, which is a regime where "which annotation helps" is a question about regularization and overfitting dynamics, not about forward dynamics modeling. Note that the relative ordering of conditions on the trained metric (B/E lowest) is preserved here, but it is an ordering of who loses to copy by the least, not of who predicts best — B_depth at N=50 has the lowest trained MSE (0.0294) yet the *highest* trained/copy ratio at that scale among the strong conditions (1.457), because its own latent trajectory is also the smoothest, so its copy baseline is even better than its trained predictor by a wider margin.

**7.2 Motion-conditioned error (`RUN/diagnostics_motion_bins.csv`).** Binning held-out windows into quintiles by frame-to-frame latent displacement `‖z_{t+1} − z_t‖` (the copy MSE *is* that displacement energy), the held-out is dominated by near-static windows: the bottom quintile has mean motion 0.0006 and the bottom three quintiles (60% of windows) all have motion < 0.005, against 0.14 in the top quintile — a >200× range. In low-motion windows the latent barely changes, so copy is near-perfect and there is nothing for any condition to improve. The effects live almost entirely in the top quintile:

| scale | condition | bin 0 Δ vs P0 (low motion) | bin 4 Δ vs P0 (high motion) |
|---|---|---|---|
| 25 | E_depth_tracks | +0.00025 | −0.0216 |
| 25 | P4_pi07_full_stack | +0.00010 | −0.00255 |
| 50 | E_depth_tracks | +0.00001 | −0.0249 |
| 50 | P4_pi07_full_stack | +0.00051 | −0.00563 |

`E_depth_tracks` is flat-to-slightly-worse than P0 in the low-motion 60% of windows and delivers its entire advantage in the high-motion 20% (−0.022 to −0.025). `P4` follows the same shape at roughly a tenth the magnitude. This is the mechanistic explanation of the small effect sizes: the metric is averaged over a held-out set that is mostly near-static, so even a signal that genuinely helps when the scene moves is diluted ~5× by windows where nothing moves and copy already wins.

The full per-bin table at N=50 (each bin ~168–169 windows; `RUN/diagnostics_motion_bins.csv`) makes the dilution explicit — note how trained MSE itself is an order of magnitude larger in bin 4 than in bin 0, and how the paired advantage tracks the motion almost perfectly:

| scale | condition | motion bin | mean motion | trained MSE | Δ vs P0 |
|---|---|---|---|---|---|
| 50 | P0_pi07_baseline | 0 | 0.00042 | 0.00820 | 0.00000 |
| 50 | P0_pi07_baseline | 1 | 0.00140 | 0.01066 | 0.00000 |
| 50 | P0_pi07_baseline | 2 | 0.00334 | 0.01130 | 0.00000 |
| 50 | P0_pi07_baseline | 3 | 0.00797 | 0.01769 | 0.00000 |
| 50 | P0_pi07_baseline | 4 | 0.11787 | 0.12542 | 0.00000 |
| 50 | E_depth_tracks | 0 | 0.00042 | 0.00821 | +0.00001 |
| 50 | E_depth_tracks | 1 | 0.00140 | 0.01004 | −0.00062 |
| 50 | E_depth_tracks | 2 | 0.00334 | 0.01068 | −0.00063 |
| 50 | E_depth_tracks | 3 | 0.00797 | 0.01532 | −0.00236 |
| 50 | E_depth_tracks | 4 | 0.11787 | 0.10050 | −0.02492 |
| 50 | P4_pi07_full_stack | 0 | 0.00042 | 0.00871 | +0.00051 |
| 50 | P4_pi07_full_stack | 1 | 0.00140 | 0.01070 | +0.00004 |
| 50 | P4_pi07_full_stack | 2 | 0.00334 | 0.01135 | +0.00004 |
| 50 | P4_pi07_full_stack | 3 | 0.00797 | 0.01684 | −0.00084 |
| 50 | P4_pi07_full_stack | 4 | 0.11787 | 0.11979 | −0.00563 |

Read down the Δ column: `E_depth_tracks` is essentially zero or slightly positive (worse) in bins 0–2, turns negative in bin 3, and concentrates 90% of its total advantage in bin 4; `P4` is slightly *positive* (worse than P0) in the three lowest-motion bins and only goes negative in the top two. A single number — the held-out mean — averages a bin-4 advantage of −0.025 against three near-static bins where the condition is neutral-to-harmful, which is exactly how a real high-motion effect becomes a −16% held-out delta for E and a sub-1% delta for P4.

**7.3 Conditioning-channel magnitude (`RUN/diagnostics_conditioning.csv`).** For each pi0.7 checkpoint we measured, on held-out windows, `‖condition‖ / ‖ctx_emb‖`, the number of distinct conditioning feature vectors seen, and subgoal coverage:

| scale | condition | cond/ctx mean | cond/ctx p90 | distinct feature vectors | subgoal coverage |
|---|---|---|---|---|---|
| 25 | P1_pi07_subtask_text | 0.027 | 0.032 | 28 | 1.0 |
| 25 | P2_pi07_metadata | 0.053 | 0.058 | 28 | 1.0 |
| 25 | P3_pi07_subgoal | 0.033 | 0.039 | 28 | 1.0 |
| 25 | P4_pi07_full_stack | 0.059 | 0.067 | 28 | 1.0 |
| 25 | P_adapter_null | 0.098 | 0.106 | 1 | 1.0 |
| 50 | P1_pi07_subtask_text | 0.046 | 0.051 | 28 | 1.0 |
| 50 | P2_pi07_metadata | 0.085 | 0.092 | 28 | 1.0 |
| 50 | P3_pi07_subgoal | 0.056 | 0.062 | 28 | 1.0 |
| 50 | P4_pi07_full_stack | 0.096 | 0.106 | 28 | 1.0 |
| 50 | P_adapter_null | 0.157 | 0.169 | 1 | 1.0 |

The conditioning vector is a small additive perturbation: the ratio is 0.027 (P1 text), 0.053 (P2 metadata), 0.033 (P3 subgoal), 0.059 (P4 full) at N=25, roughly doubling at N=50 (P4 0.096). Three observations make this worse than "small." First, **`P_adapter_null` has the largest ratio of all** (0.098 at N=25, 0.157 at N=50): the zero-feature adapter maps to its own bias and produces a *bigger* perturbation than the content-bearing adapters, so the fraction of the conditioning magnitude actually attributable to annotation content is smaller than the already-small raw ratio. Second, **only 28 distinct conditioning vectors** appear across all 281 held-out windows for every content family — because the conditioning is constant within a subtask segment and the 10 held-out episodes contain exactly 3 segments each (30 segments, collapsing to 28 distinct hashed vectors). The channel carries ~28 effective values, not 281. Third, subgoal coverage is 1.0 (every held-out window's active segment has a subgoal frame), so the subgoal signal is fully present and still does nothing under normalization — the failure is not missing data. Taken together, the conditioning channel was small in magnitude, dominated by content-independent adapter bias, and low in cardinality: it was unlikely to move the metric even in principle, which is the honest framing for the null pi0.7 result.

**7.4 Training-dynamics audit (`RUN/diagnostics_training_dynamics.csv`).** Per-epoch losses were not persisted — both trainers run Lightning with `logger=False` (`finetune_with_aux.py`, `pi07_fixed.py`) — so the convergence-slope question cannot be answered from logs, and that itself is a reproducibility gap worth fixing. The available substitute is the trained predictor's train-split vs held-out MSE, an overfitting measure, computed eval-only. It is severe and uniform: train MSE ≈ 0.0036–0.0040 at N=25 against held-out ≈ 0.044–0.049, a gap of about −92% of held-out; at N=50, train ≈ 0.0034 against held-out ≈ 0.029–0.035, gap about −88 to −90%. The model drives training error to near zero (train MSE is roughly an order of magnitude below even the *copy* floor) and generalizes to a held-out floor that copy beats. This is the expected behavior of 18.7M trainable parameters on 285–812 training windows. One small but consistent signal: at N=50 the depth conditions overfit slightly less (gap −88.4% for B/E vs −90.2% for P0), which is the most likely true mechanism of their advantage — the dense per-frame auxiliary loss is a regularizer that modestly reduces overfitting, not a source of new forward-dynamics skill.

**Synthesis: LeWM-as-instrument and one-step latent prediction as a task.** The diagnostics say the instrument is poorly conditioned for this question. The task is near-degenerate (copy beats trained; 95% solved by the constant baseline), the held-out is motion-poor (60% near-static), the model is badly overfit at these N, the conditioning channel is a ~5% additive nudge with ~28 distinct values dominated by adapter bias, and everything is forced through a 192-D CLS bottleneck. In that regime, the largest honest effect available — depth/track auxiliary supervision — behaves like a regularizer that helps in the high-motion minority, and the pi0.7 conditioning is below the noise floor once latent-geometry confounds are removed. The small effect sizes are not a measurement failure; they are the correct readout of a task that is mostly solved by doing nothing.

---

## 8. Threats to validity

These are stated as findings, not caveats.

**Latent-MSE is fragile across finetuned encoders.** Because every condition finetunes the encoder, the target latents differ per condition, and the raw metric is not in a shared space. We caught one false positive from this (P4 N=50) with variance normalization, but normalization is a partial fix: it controls for first/second moments of the target distribution, not for higher-order geometry changes that could still flatter or penalize a condition. The IDM probe in action units is the more trustworthy instrument and it is underpowered. Any future headline number should be reported normalized and, ideally, in action space.

**Supervision-bandwidth confound, quantified.** The paradigms differ in how many supervised scalars they deliver per episode, by orders of magnitude. Depth (condition B/E) supervises a 56×56 = 3,136-value map per frame; at ~25 frames/episode that is ≈ 78,000 supervised scalars per episode. Tracks (D/E) supervise 400 points × 2 coordinates × ~90% visible (`track_visibility` mean 0.902 in `RUN/datasets/be_h2h_scale_25_heldout.h5`) ≈ 720 values/frame ≈ 18,000 per episode. The pi0.7 text/metadata conditioning delivers, per episode, ~3 subtask phrases and 4 metadata scalars — call it single-digit supervised/conditioning signals per episode. So when "depth as aux target" beats "text as conditioning," the comparison is confounded by a ~4–5 order-of-magnitude difference in supervisory bandwidth. The annotation-strategy framing is honest about the mechanism difference, but a reader should not conclude "CV signals are better annotations than language"; the controlled statement is "a dense per-frame regression target regularizes this overfit model more than a sparse per-segment conditioning vector does," which is nearly tautological.

**Mechanism confound: the missing 2×2.** Each signal is tested in exactly one mechanism (CV as target, pi0.7 as input). The 2×2 — depth as conditioning, text as auxiliary target — is absent, so paradigm and mechanism cannot be separated. This is a deliberate design choice (Section 2) but it means the experiment cannot answer "is the win from the signal or from being a target?"

**Oracle subgoal: target-copying ruled out, privileged future not.** The subgoal frame is sampled from the same episode being predicted (`pi07_fixed.py` `_pi07_subgoal_condition`, `_records_for_h5_order`). The leak audit (Section 6) shows the advantage is not boundary-concentrated, ruling out crude copying, but a same-episode future frame can still leak globally useful future information. Until subgoals are retrieved from other episodes or generated, P3/P4 are oracle-flavored and not a deployable result — and in any case they do not survive normalization here.

**Hashed text is not semantics.** As Section 4 details, the text channel is a 128-bin signed hash of tokens with collisions and no word order. A null pi0.7 text result is partly a null *representation* result; it does not test whether a real language encoder (e.g. a sentence transformer) would help.

**N=100 incomplete; 10 held-out episodes.** The N=100 grid is missing cells and has no window-level CSVs, so all conclusions are N≤50 (`RUN/runs/scale_100/` has only A/B/P0/adapter_null with no `fixed_eval_windows.csv`). The held-out is 10 episodes / 281 windows; the cluster bootstrap correctly resamples 30 (seed, episode) units, which is why even real effects have wide CIs. Effect-size estimates at this held-out size are noisy and the study is underpowered for anything below ~1–2% of held-out MSE.

**CLS bottleneck.** Per the prior LeWM finding and the `encode` source, all spatial information passes through one 192-D CLS token. Auxiliary spatial targets and any spatial conditioning are throttled by this; a patch-token predictor could plausibly change the depth/track result and is untested here.

**Overfitting regime.** With train MSE an order of magnitude below held-out for every condition, the comparison is partly a comparison of *which regularizer overfits least*, not which world model predicts best. Conclusions may not transfer to a regime with enough data to fit the model honestly.

---

## 9. What would change the conclusion

Concrete next runs and the result each would need to produce to overturn the current reading. None were run; they are specified so the bar is explicit.

1. **Real language encoder at N=50.** Replace `_hash_text` with a frozen sentence-transformer embedding of the subtask text, keep everything else fixed, rerun P1/P2/P4 at N=50, three seeds. To matter, the normalized paired delta vs P0 must clear zero where the hashed version did not. This isolates "is the null a representation artifact?"

2. **Dense-semantic auxiliary target.** Convert a pi0.7 signal into a per-frame target (e.g. a per-frame subtask-class logit or a per-frame language-embedding regression) so it enters through the *same* mechanism as depth, at matched supervisory bandwidth. If it then matches depth's normalized win, the depth advantage was mechanism/bandwidth, not signal; if it does not, depth carries genuinely more useful structure.

3. **Multi-step prediction horizon.** Raise the prediction horizon from 1 to e.g. 4–8 steps so that copy is no longer near-optimal and the trained predictor must actually extrapolate. The trivial-floor diagnostic (`diagnostics.py`) should be rerun; the precondition for any annotation to matter is trained/copy < 1. Only in that regime is "which annotation helps prediction" a well-posed question.

4. **Patch-token predictor.** Replace the CLS-only encode with a predictor over patch tokens so spatial supervision is not throttled by the 192-D bottleneck. To matter, the depth/track normalized advantage should grow and/or generalize beyond the high-motion quintile (rerun `diagnostics_motion_bins`).

5. **Held-out and seed scale-up.** Expand held-out beyond 10 episodes and seeds beyond 3 (or complete N=100). The IDM action-space probe is the metric to power up; a real annotation effect should eventually separate from zero there.

---

## 10. Appendix

**Artifacts (all under `RUN/ = D:\lewm_runs\bridgeengine_head_to_head\run_100\`).** Per-run: `runs/scale_{25,50}/<condition>_seed<seed>/` containing `config_snapshot.yaml`, `checkpoints/final/full_weights.pt`, `fixed_eval.json` (+ `heldout_target_variance`, `norm_latent_mse`), `fixed_eval_windows.csv` (+ `norm_sq_err`), `idm_eval.json`, `idm_eval_windows.csv`, `diagnostics.json`, `diagnostics_windows.csv`, `metadata.json`. Aggregates: `paired_window_summary.csv`, `paired_window_deltas.csv`, `target_variance_table.csv`, `subgoal_leak_bins.csv`, `idm_paired_summary.csv`, `idm_paired_deltas.csv`, `diagnostics_trivial_floor.csv`, `diagnostics_motion_bins.csv`, `diagnostics_conditioning.csv`, `diagnostics_training_dynamics.csv`. Figures: `figures/paired_delta_forest.png`, `figures/paired_delta_forest_normalized.png`, `figures/subgoal_advantage_by_boundary_distance.png`. Narrative report: `LEAK_AND_POWER_REPORT.md`.

**Code.** Data prep: `bridgeengine/benchmark/head_to_head_runner.py` (`export_split_h5_from_npy`). Evaluators: `bridgeengine/benchmark/lewm_fixed_eval.py`, `bridgeengine/benchmark/pi07_fixed.py`. Per-window/metric utilities: `bridgeengine/benchmark/window_eval.py` (`TargetLatentMoments`, `normalized_window_errors`, `write_fixed_eval_windows`). Paired analysis: `bridgeengine/benchmark/leak_power.py`. IDM probe: `bridgeengine/benchmark/idm.py`. Diagnostics: `bridgeengine/benchmark/diagnostics.py`. Conditioning features: `bridgeengine/benchmark/train_lewm.py` (`_hash_text`, `_metadata_vector`, `PromptConditioner`). CV trainer / aux heads: `C:\Users\Kevin\projects\LeWM_testbed\scripts\finetune_with_aux.py`, `...\src\lewm_testbed\auxiliary\heads.py`. Model: `C:\Users\Kevin\projects\upstream\stable-worldmodel\stable_worldmodel\wm\lewm\lewm.py`.

**Reproduction commands** (interpreter `.\.venv\Scripts\python.exe`, eval-only, GPU):

```
# Re-score all completed cells with the normalized metric (regenerates fixed_eval*.csv):
python -m bridgeengine.benchmark.leak_power reevaluate --output-dir RUN --scales 25 50 --force
# Raw + normalized paired analysis, variance table, leak bins, figures, report:
python -m bridgeengine.benchmark.leak_power analyze --output-dir RUN --scales 25 50 --report-path LEAK_AND_POWER_REPORT.md
# IDM action-decoding probe + paired analysis (appends IDM section to the report):
python -m bridgeengine.benchmark.idm run --output-dir RUN --scales 25 50
python -m bridgeengine.benchmark.idm analyze --output-dir RUN --scales 25 50 --report-path LEAK_AND_POWER_REPORT.md
# Diagnostics grid + summary tables:
python -m bridgeengine.benchmark.diagnostics run --output-dir RUN --scales 25 50
python -m bridgeengine.benchmark.diagnostics summarize --output-dir RUN --scales 25 50
```

**Preregistration amendments.** `HEAD_TO_HEAD_PREREGISTRATION.md` carries a dated 2026-06-09 amendments section recording the additive, eval-only nature of the normalized metric, relative leak bins, and IDM probe; the diagnostics in this report are a further additive, eval-only pass under the same terms (no split/seed/rubric/checkpoint changed).

**Cost accounting.** This report's diagnostics added three eval-only passes over the 60 completed scale-25/50 cells: the normalized re-evaluation (one held-out forward pass per cell), the IDM probe (one train-split + one held-out forward pass plus a tiny MLP fit per cell), and the diagnostics grid (one held-out + one train-split forward pass per cell). No world-model training was run; total added GPU time was on the order of an hour across the three background grids, dominated by encoder forward passes, not optimization.

**Scope statement.** All quantitative conclusions are restricted to: LeWM-cube finetuned on BridgeData V2 kitchen manipulation, one fixed 10-episode held-out split, N ∈ {25, 50}, three seeds, one-step latent prediction, hashed-text pi0.7 conditioning. They do not generalize to other world models, larger N, multi-step prediction, real language encoders, or non-manipulation video.
