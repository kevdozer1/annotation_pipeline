# Leak And Power Report

Eval-only report. No checkpoints were trained or modified in this pass.

Scales analyzed: 25, 50.
Scale 100 is intentionally excluded here because its checkpoint set is incomplete; this report only analyzes scales with all CV and pi0.7 cells completed.
Paired deltas CSV: `D:\lewm_runs\bridgeengine_head_to_head\run_100\paired_window_deltas.csv`.
Paired summary CSV: `D:\lewm_runs\bridgeengine_head_to_head\run_100\paired_window_summary.csv`.
Boundary leak CSV: `D:\lewm_runs\bridgeengine_head_to_head\run_100\subgoal_leak_bins.csv`.
Target-variance table CSV: `D:\lewm_runs\bridgeengine_head_to_head\run_100\target_variance_table.csv`.

## Eval-Validity Framing

Every condition finetunes the full LeWM model, so each condition's held-out next-latent MSE is measured in its own latent geometry. A condition can therefore lower raw MSE by contracting target-latent variance rather than predicting better. This report reads every paired delta in two ways: the raw per-window MSE delta, and a variance-normalized delta where each run's per-window error is divided by that run's own mean per-dimension target-latent variance (`norm_sq_err = sq_err / heldout_target_variance`). The normalized delta is unitless (fraction of target variance unexplained) and is the leakage-robust read.

## Paired Power Verdict (raw)

CIs are paired bootstraps over episode-seed clusters, not individual adjacent windows. That is intentionally conservative because window errors are temporally correlated.

Conditions with RAW paired CIs separating from zero versus P0: scale 25 E_depth_tracks (better), scale 25 B_depth (better), scale 50 E_depth_tracks (better), scale 50 B_depth (better), scale 50 P4_pi07_full_stack (better), scale 50 D_tracks (better).

| scale_n | condition            | paired_windows | bootstrap_units | seeds | mean_delta_sq_err | mean_delta_pct | ci_low       | ci_high      | separates_zero |
| ------- | -------------------- | -------------- | --------------- | ----- | ----------------- | -------------- | ------------ | ------------ | -------------- |
| 25      | E_depth_tracks       | 843            | 30              | 3     | -0.00514099       | -10.4387       | -0.00685639  | -0.00212954  | True           |
| 25      | B_depth              | 843            | 30              | 3     | -0.00396457       | -8.05004       | -0.00544104  | -0.00136559  | True           |
| 25      | P4_pi07_full_stack   | 843            | 30              | 3     | -0.000441222      | -0.895899      | -0.000962115 | 0.000394743  | False          |
| 25      | P3_pi07_subgoal      | 843            | 30              | 3     | -0.000386111      | -0.783996      | -0.00105548  | 0.000578408  | False          |
| 25      | A_baseline           | 843            | 30              | 3     | -0.000105688      | -0.2146        | -0.000205694 | 3.00054e-05  | False          |
| 25      | P2_pi07_metadata     | 843            | 30              | 3     | -3.02561e-05      | -0.0614348     | -0.000301208 | 0.00022798   | False          |
| 25      | D_tracks             | 843            | 30              | 3     | -1.17754e-05      | -0.0239098     | -0.000392199 | 0.00028086   | False          |
| 25      | P1_pi07_subtask_text | 843            | 30              | 3     | 2.35738e-05       | 0.0478665      | -0.000329783 | 0.000320651  | False          |
| 25      | P_adapter_null       | 843            | 30              | 3     | 3.96122e-05       | 0.0804323      | -0.000345988 | 0.000336032  | False          |
| 50      | E_depth_tracks       | 843            | 30              | 3     | -0.00571531       | -16.4693       | -0.00873532  | -0.00236704  | True           |
| 50      | B_depth              | 843            | 30              | 3     | -0.00532848       | -15.3546       | -0.00746635  | -0.00175997  | True           |
| 50      | P4_pi07_full_stack   | 843            | 30              | 3     | -0.00117641       | -3.38996       | -0.00209131  | -0.000264738 | True           |
| 50      | P3_pi07_subgoal      | 843            | 30              | 3     | -0.000981811      | -2.8292        | -0.00189804  | 7.78402e-05  | False          |
| 50      | D_tracks             | 843            | 30              | 3     | -0.000473017      | -1.36305       | -0.00113022  | -3.0459e-05  | True           |
| 50      | A_baseline           | 843            | 30              | 3     | -0.000208977      | -0.602193      | -0.00061074  | 0.000221588  | False          |
| 50      | P_adapter_null       | 843            | 30              | 3     | -4.69799e-05      | -0.135378      | -0.000422697 | 0.000316863  | False          |
| 50      | P2_pi07_metadata     | 843            | 30              | 3     | 4.30325e-05       | 0.124003       | -0.000334341 | 0.000332571  | False          |
| 50      | P1_pi07_subtask_text | 843            | 30              | 3     | 0.00032784        | 0.944708       | -0.000387418 | 0.000841114  | False          |

## Paired Power Verdict (variance-normalized)

Conditions with NORMALIZED paired CIs separating from zero versus P0: scale 25 E_depth_tracks (better), scale 50 E_depth_tracks (better), scale 50 B_depth (better), scale 50 D_tracks (better).

| scale_n | condition            | paired_windows | mean_delta_norm_sq_err | mean_delta_norm_pct | ci_low_norm  | ci_high_norm | separates_zero_norm |
| ------- | -------------------- | -------------- | ---------------------- | ------------------- | ------------ | ------------ | ------------------- |
| 25      | E_depth_tracks       | 843            | -0.0048203             | -6.94349            | -0.00741144  | -0.000806527 | True                |
| 25      | B_depth              | 843            | -0.00334409            | -4.81706            | -0.0055187   | 6.87737e-05  | False               |
| 25      | P4_pi07_full_stack   | 843            | 0.000257433            | 0.370825            | -0.00043518  | 0.0012562    | False               |
| 25      | P3_pi07_subgoal      | 843            | 0.000278261            | 0.400826            | -0.000635433 | 0.00146941   | False               |
| 25      | A_baseline           | 843            | -9.97101e-05           | -0.143629           | -0.000245319 | 6.31183e-05  | False               |
| 25      | P2_pi07_metadata     | 843            | 4.96523e-07            | 0.000715225         | -0.00033841  | 0.000325047  | False               |
| 25      | D_tracks             | 843            | -3.03404e-05           | -0.0437044          | -0.00052343  | 0.000396098  | False               |
| 25      | P1_pi07_subtask_text | 843            | 5.79078e-05            | 0.0834144           | -0.000415108 | 0.000458331  | False               |
| 25      | P_adapter_null       | 843            | 9.2382e-05             | 0.133073            | -0.000451633 | 0.000482355  | False               |
| 50      | E_depth_tracks       | 843            | -0.00714006            | -16.1628            | -0.0104841   | -0.00288118  | True                |
| 50      | B_depth              | 843            | -0.00720465            | -16.309             | -0.00998895  | -0.00242031  | True                |
| 50      | P4_pi07_full_stack   | 843            | -0.000244597           | -0.553688           | -0.00111695  | 0.000513263  | False               |
| 50      | P3_pi07_subgoal      | 843            | -7.44706e-05           | -0.168577           | -0.00099922  | 0.000925472  | False               |
| 50      | D_tracks             | 843            | -0.000796637           | -1.80333            | -0.00172395  | -0.000102406 | True                |
| 50      | A_baseline           | 843            | -0.000373237           | -0.844886           | -0.000896049 | 0.000181146  | False               |
| 50      | P_adapter_null       | 843            | -7.2272e-05            | -0.1636             | -0.000548295 | 0.000436392  | False               |
| 50      | P2_pi07_metadata     | 843            | -8.99304e-06           | -0.0203573          | -0.000513395 | 0.000373108  | False               |
| 50      | P1_pi07_subtask_text | 843            | 0.000422527            | 0.956463            | -0.000465468 | 0.00109233   | False               |

## Per-Condition Target-Latent Variance

No depth/track aux condition contracts mean per-dim target variance by more than 10% versus the A_baseline native anchor at the same scale, so the raw advantages are not primarily a variance artifact.

| scale_n | condition            | seeds | mean_target_variance | mean_latent_mse | mean_norm_latent_mse | var_ratio_vs_A_baseline | var_ratio_vs_P0 | variance_contracted_flag |
| ------- | -------------------- | ----- | -------------------- | --------------- | -------------------- | ----------------------- | --------------- | ------------------------ |
| 25      | A_baseline           | 3     | 0.715576             | 0.0491434       | 0.0693221            | 1                       | 0.999404        | False                    |
| 25      | B_depth              | 3     | 0.689748             | 0.0452846       | 0.0660778            | 0.963907                | 0.963332        | False                    |
| 25      | D_tracks             | 3     | 0.715816             | 0.0492374       | 0.0693915            | 1.00033                 | 0.999739        | False                    |
| 25      | E_depth_tracks       | 3     | 0.686751             | 0.0441081       | 0.0646015            | 0.959718                | 0.959147        | False                    |
| 25      | P0_pi07_baseline     | 3     | 0.716002             | 0.0492491       | 0.0694218            | 1.0006                  | 1               | False                    |
| 25      | P1_pi07_subtask_text | 3     | 0.716166             | 0.0492727       | 0.0694798            | 1.00083                 | 1.00023         | False                    |
| 25      | P2_pi07_metadata     | 3     | 0.715624             | 0.0492189       | 0.0694223            | 1.00007                 | 0.999471        | False                    |
| 25      | P3_pi07_subgoal      | 3     | 0.70777              | 0.048863        | 0.0697001            | 0.989092                | 0.988502        | False                    |
| 25      | P4_pi07_full_stack   | 3     | 0.707347             | 0.0488079       | 0.0696793            | 0.988501                | 0.987912        | False                    |
| 25      | P_adapter_null       | 3     | 0.715735             | 0.0492887       | 0.0695142            | 1.00022                 | 0.999627        | False                    |
| 50      | A_baseline           | 3     | 0.792532             | 0.0344938       | 0.0438028            | 1                       | 1.00233         | False                    |
| 50      | B_depth              | 3     | 0.797212             | 0.0293743       | 0.0369714            | 1.00591                 | 1.00825         | False                    |
| 50      | D_tracks             | 3     | 0.794228             | 0.0342297       | 0.0433794            | 1.00214                 | 1.00448         | False                    |
| 50      | E_depth_tracks       | 3     | 0.786153             | 0.0289874       | 0.037036             | 0.991951                | 0.994265        | False                    |
| 50      | P0_pi07_baseline     | 3     | 0.790687             | 0.0347027       | 0.044176             | 0.997673                | 1               | False                    |
| 50      | P1_pi07_subtask_text | 3     | 0.790772             | 0.0350306       | 0.0445986            | 0.99778                 | 1.00011         | False                    |
| 50      | P2_pi07_metadata     | 3     | 0.791361             | 0.0347458       | 0.044167             | 0.998523                | 1.00085         | False                    |
| 50      | P3_pi07_subgoal      | 3     | 0.769803             | 0.0337209       | 0.0441016            | 0.971321                | 0.973587        | False                    |
| 50      | P4_pi07_full_stack   | 3     | 0.768239             | 0.0335263       | 0.0439314            | 0.969348                | 0.971609        | False                    |
| 50      | P_adapter_null       | 3     | 0.790463             | 0.0346558       | 0.0441038            | 0.99739                 | 0.999717        | False                    |

## Subgoal-Leak Audit

No clear near-boundary concentration was observed for P3/P4 in the analyzed scales.

| scale_n | condition          | windows | dist_advantage_corr | dist_advantage_rel_corr | near_advantage | far_advantage | near_minus_far_advantage | near_advantage_rel | far_advantage_rel | near_minus_far_advantage_rel |
| ------- | ------------------ | ------- | ------------------- | ----------------------- | -------------- | ------------- | ------------------------ | ------------------ | ----------------- | ---------------------------- |
| 25      | P3_pi07_subgoal    | 843     | 0.0486709           | 0.0168693               | 0.000441501    | 0.00159777    | -0.00115627              | 0.01219            | 0.0167201         | -0.00453006                  |
| 25      | P4_pi07_full_stack | 843     | 0.0396366           | 0.0298675               | 0.000547235    | 0.00124958    | -0.000702342             | 0.0151094          | 0.0130763         | 0.00203306                   |
| 50      | P3_pi07_subgoal    | 843     | 0.124931            | 0.0453763               | 9.49635e-06    | 0.00271728    | -0.00270779              | 0.000473179        | 0.0428089         | -0.0423357                   |
| 50      | P4_pi07_full_stack | 843     | 0.105996            | 0.0470359               | 0.000379423    | 0.00220541    | -0.00182599              | 0.0189057          | 0.0347447         | -0.015839                    |

Relative advantage divides each window's advantage by the P0 baseline error on the same window, so the near-vs-far comparison is not dominated by absolute error scale. Honest reading: the absence of near-boundary concentration rules out crude target-copying of the supplied subgoal frame, but it does **not** rule out privileged-future-information leakage. A same-episode subgoal frame is drawn from the future of the very trajectory being predicted; it can carry globally useful future context that lowers error uniformly across distance bins rather than only at the boundary. Deployability still requires replacing the same-episode subgoal with a retrieved or generated one.

## Figures

- Paired delta forest (raw): `D:\lewm_runs\bridgeengine_head_to_head\run_100\figures\paired_delta_forest.png`
- Paired delta forest (normalized): `D:\lewm_runs\bridgeengine_head_to_head\run_100\figures\paired_delta_forest_normalized.png`
- Boundary-distance audit: `D:\lewm_runs\bridgeengine_head_to_head\run_100\figures\subgoal_advantage_by_boundary_distance.png`

## Interpretation

Negative paired delta means the condition has lower held-out next-latent MSE than P0 on the same windows. Where the raw and normalized verdicts disagree, the normalized verdict governs: a raw win that vanishes after variance normalization is a latent-geometry artifact, not a prediction gain. The boundary-distance audit is a diagnostic, not a proof.

## IDM Action-Decoding Probe

Action-space metric, immune to the latent-variance confound. A 2-layer MLP `f(z_t, z_{t+1}) -> a_t` is trained per condition-seed on train-split encoded latents, then scored on held-out windows using the model's predicted latents; the table reports paired deltas in per-window action MSE versus P0 (negative is better).

Paired IDM summary CSV: `D:\lewm_runs\bridgeengine_head_to_head\run_100\idm_paired_summary.csv`.

In action space, no condition's paired IDM action-error CI separates from zero versus P0 at the analyzed scales: the conditions that win on latent MSE do not measurably improve the decodability of the true action from the predicted latent transition.

Power caveat: each condition-seed trains its own probe in its own latent geometry, so the paired CIs are much wider than the latent-MSE CIs and this probe is underpowered at three seeds. The point estimates are still directionally informative: at scale 50 the track/depth CV conditions (`E_depth_tracks`, `D_tracks`) have the lowest action error and are the only ones below P0, while the pi0.7 conditioning families sit above P0 — consistent with motion-style CV targets shaping action-decodable latents and with pi0.7 conditioning not improving action decodability. None of these gaps clear the noise floor here.

| scale_n | condition            | paired_windows | mean_delta_action_mse | ci_low      | ci_high   | separates_zero |
| ------- | -------------------- | -------------- | --------------------- | ----------- | --------- | -------------- |
| 25      | A_baseline           | 843            | 0.00222752            | -0.0174151  | 0.0101926 | False          |
| 25      | P1_pi07_subtask_text | 843            | 0.0100475             | -0.0688522  | 0.0709537 | False          |
| 25      | D_tracks             | 843            | 0.0137008             | -0.0390576  | 0.059988  | False          |
| 25      | E_depth_tracks       | 843            | 0.0173982             | -0.126873   | 0.142782  | False          |
| 25      | P_adapter_null       | 843            | 0.0269793             | -0.0491125  | 0.0912604 | False          |
| 25      | B_depth              | 843            | 0.0275246             | -0.130121   | 0.153661  | False          |
| 25      | P2_pi07_metadata     | 843            | 0.0404663             | -0.0531014  | 0.098049  | False          |
| 25      | P4_pi07_full_stack   | 843            | 0.0456484             | -0.0385243  | 0.108722  | False          |
| 25      | P3_pi07_subgoal      | 843            | 0.0534425             | -0.0356106  | 0.123028  | False          |
| 50      | E_depth_tracks       | 843            | -0.0269144            | -0.148533   | 0.0332253 | False          |
| 50      | D_tracks             | 843            | -0.0146388            | -0.0920874  | 0.0275343 | False          |
| 50      | B_depth              | 843            | 0.00855177            | -0.232816   | 0.148408  | False          |
| 50      | P1_pi07_subtask_text | 843            | 0.0158964             | -0.182452   | 0.238893  | False          |
| 50      | A_baseline           | 843            | 0.0230394             | -0.00906542 | 0.0713784 | False          |
| 50      | P_adapter_null       | 843            | 0.0404797             | -0.0538323  | 0.187715  | False          |
| 50      | P4_pi07_full_stack   | 843            | 0.0567716             | -0.0688425  | 0.281247  | False          |
| 50      | P3_pi07_subgoal      | 843            | 0.0624186             | -0.0588811  | 0.280883  | False          |
| 50      | P2_pi07_metadata     | 843            | 0.0843547             | -0.0253245  | 0.289747  | False          |

