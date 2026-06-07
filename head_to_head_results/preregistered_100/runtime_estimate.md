# Head-To-Head Runtime Estimate

Snapshot: `snap_2026_05_11_1dde3edf5d_human_gold_labels`

## Shared Episode Check

- same 100-episode set: `True`
- BridgeEngine episodes: 100
- LeWM manifest episodes: 100

## Planned Conditions

| condition | paradigm | mechanism |
|---|---|---|
| `baseline` | shared_reference | no auxiliary prediction target and no pi0.7 conditioning |
| `cv_B_depth_aux` | LeWM perceptive | Video-Depth-Anything depth as an auxiliary prediction target |
| `cv_D_tracks_aux` | LeWM perceptive | CoTracker3 point displacements as an auxiliary prediction target |
| `cv_E_depth_tracks_aux` | LeWM perceptive | VDA depth and CoTracker3 tracks as auxiliary prediction targets |
| `pi07_rich_text` | BridgeEngine pi0.7 | subtask text as a conditioning input |
| `pi07_rich_text_metadata` | BridgeEngine pi0.7 | subtask text plus speed, quality, mistake, and control metadata as conditioning inputs |
| `pi07_full_metadata_subgoal` | BridgeEngine pi0.7 | subtask text, metadata, and end-of-segment subgoal frame as conditioning inputs |

## Estimate

- requested sizes: `[25, 50, 100]`
- requested train seeds: `[42, 137, 256]`
- LeWM CV fullscale cached N=100 training: 1.660 hours
- LeWM CV all scales from scratch estimate: 2.904 hours
- LeWM CV incremental estimate reusing cached N=100: 1.245 hours
- BridgeEngine pi0.7 3-seed estimate: 0.991 hours
- total from scratch estimate: 3.896 hours
- total incremental estimate reusing cached CV N=100: 2.236 hours

## Stop Reason

Do not launch the unified grid until the fixed-split evaluator path is chosen. The cached LeWM evaluator currently uses a per-seed random 90/10 split, while the preregistered comparison requires one shared fixed split.
