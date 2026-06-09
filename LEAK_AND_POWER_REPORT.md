# Leak And Power Report

Eval-only report. No checkpoints were trained or modified in this pass.

Scales analyzed: 25, 50.
Scale 100 is intentionally excluded here because its checkpoint set is incomplete; this report only analyzes scales with all CV and pi0.7 cells completed.
Full paired deltas CSV: `D:\lewm_runs\bridgeengine_head_to_head\run_100\paired_window_deltas.csv` (not committed; generated table is 23,604 rows).
Paired summary CSV: `head_to_head_results/preregistered_100/leak_power/paired_window_summary.csv`.
Boundary leak CSV: `head_to_head_results/preregistered_100/leak_power/subgoal_leak_bins.csv`.

## Paired Power Verdict

CIs are paired bootstraps over episode-seed clusters, not individual adjacent windows. That is intentionally conservative because window errors are temporally correlated.

Conditions with paired CIs separating from zero versus P0: scale 25 E_depth_tracks (better), scale 25 B_depth (better), scale 50 E_depth_tracks (better), scale 50 B_depth (better), scale 50 P4_pi07_full_stack (better), scale 50 D_tracks (better).

| scale_n | condition            | paired_windows | bootstrap_units | seeds | mean_delta_sq_err | mean_delta_pct | ci_low       | ci_high      | separates_zero |
| ------- | -------------------- | -------------- | --------------- | ----- | ----------------- | -------------- | ------------ | ------------ | -------------- |
| 25      | E_depth_tracks       | 843            | 30              | 3     | -0.00514099       | -10.4387       | -0.00684599  | -0.00209727  | True           |
| 25      | B_depth              | 843            | 30              | 3     | -0.00396457       | -8.05004       | -0.00555243  | -0.00133741  | True           |
| 25      | P4_pi07_full_stack   | 843            | 30              | 3     | -0.000441222      | -0.895899      | -0.000962849 | 0.000388165  | False          |
| 25      | P3_pi07_subgoal      | 843            | 30              | 3     | -0.000386111      | -0.783996      | -0.0010874   | 0.000580817  | False          |
| 25      | A_baseline           | 843            | 30              | 3     | -0.000105688      | -0.2146        | -0.000205694 | 3.00054e-05  | False          |
| 25      | P2_pi07_metadata     | 843            | 30              | 3     | -3.02561e-05      | -0.0614348     | -0.000275123 | 0.000226762  | False          |
| 25      | D_tracks             | 843            | 30              | 3     | -1.17754e-05      | -0.0239098     | -0.000381204 | 0.000300058  | False          |
| 25      | P1_pi07_subtask_text | 843            | 30              | 3     | 2.35738e-05       | 0.0478665      | -0.000312017 | 0.000322029  | False          |
| 25      | P_adapter_null       | 843            | 30              | 3     | 3.96122e-05       | 0.0804323      | -0.000301729 | 0.000358341  | False          |
| 50      | E_depth_tracks       | 843            | 30              | 3     | -0.00571531       | -16.4693       | -0.00845056  | -0.00230508  | True           |
| 50      | B_depth              | 843            | 30              | 3     | -0.00532848       | -15.3546       | -0.00760086  | -0.00175974  | True           |
| 50      | P4_pi07_full_stack   | 843            | 30              | 3     | -0.00117641       | -3.38996       | -0.0020428   | -0.000238489 | True           |
| 50      | P3_pi07_subgoal      | 843            | 30              | 3     | -0.000981811      | -2.8292        | -0.00195999  | 7.15576e-05  | False          |
| 50      | D_tracks             | 843            | 30              | 3     | -0.000473017      | -1.36305       | -0.00112026  | -3.714e-05   | True           |
| 50      | A_baseline           | 843            | 30              | 3     | -0.000208977      | -0.602193      | -0.000615983 | 0.000240705  | False          |
| 50      | P_adapter_null       | 843            | 30              | 3     | -4.69799e-05      | -0.135378      | -0.000406947 | 0.000324898  | False          |
| 50      | P2_pi07_metadata     | 843            | 30              | 3     | 4.30325e-05       | 0.124003       | -0.000326273 | 0.000304083  | False          |
| 50      | P1_pi07_subtask_text | 843            | 30              | 3     | 0.00032784        | 0.944708       | -0.000391157 | 0.000824436  | False          |

## Subgoal-Leak Audit

No clear near-boundary concentration was observed for P3/P4 in the analyzed scales.

| scale_n | condition          | windows | dist_advantage_corr | near_advantage | far_advantage | near_minus_far_advantage |
| ------- | ------------------ | ------- | ------------------- | -------------- | ------------- | ------------------------ |
| 25      | P3_pi07_subgoal    | 843     | 0.0486709           | 0.000441501    | 0.00159777    | -0.00115627              |
| 25      | P4_pi07_full_stack | 843     | 0.0396366           | 0.000547235    | 0.00124958    | -0.000702342             |
| 50      | P3_pi07_subgoal    | 843     | 0.124931            | 9.49635e-06    | 0.00271728    | -0.00270779              |
| 50      | P4_pi07_full_stack | 843     | 0.105996            | 0.000379423    | 0.00220541    | -0.00182599              |

## Figures

- Paired delta forest: `figures/leak_power/paired_delta_forest.png`
- Boundary-distance audit: `figures/leak_power/subgoal_advantage_by_boundary_distance.png`

## Interpretation

Negative paired delta means the condition has lower held-out next-latent MSE than P0 on the same windows. The boundary-distance audit is a diagnostic, not a proof: if subgoal advantage is concentrated near segment ends and decays with distance while CV controls do not, the same-episode subgoal frame is acting like an oracle cue rather than a deployable conditioning signal.
