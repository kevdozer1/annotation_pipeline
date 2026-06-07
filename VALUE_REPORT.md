# BridgeEngine Value Report

**TOOL VALUE:** A user cloning this today gets a working local BridgeData-to-Parquet-to-VLM-labels-to-DuckDB/export/viewer POC; it can run end to end with the mock backend without an API key, but useful semantic labels require a real VLM backend.

**MEASURED EFFECT:** A human-score and human-boundary corrected 100-episode LeWM scale curve now exists; the rich-text + metadata + subgoal family beats baseline at N=25/50/100, including a 6.28% lower mean latent MSE at N=100.

**BOTTOM LINE:** This is worth shipping as a research/demo scaffold with a real positive smoke result, but not as final proof that pi0.7-style conditioning robustly improves robot policy learning.

## Section A - Tool Value

### 1. Capability Audit

| Capability | Status | Evidence | Honest read |
|---|---|---|---|
| Venv/install | WORKS | `python -m pip install -e .` completed in the existing `.venv`. | Basic Python package setup is fine on this machine. |
| BridgeData ingest | WORKS | Ingest produced deterministic snapshots at 13, 50, remaining-50, and all-100 local episodes from `D:\bridgedata_v2_subset`. | Ingest is deterministic and can operate on Kevin's local BridgeData subset, which currently exposes 100 episodes. |
| Snapshot storage | WORKS | Snapshot writes `manifest.json`, `episodes.parquet`, `steps.parquet`, `sensors.parquet`, and `labels.parquet`. | Plain Parquet snapshot layer works for the POC. It is not a production dataset versioning system. |
| Two-stage semantic labeling | PARTIAL | Gemini-backed all-100 snapshot has 493 merged label rows: 100 metadata rows, 100 subtask rows, and 293 subgoal rows. Kevin then reviewed all 100 clips and changed 58 scores. Mock backend still works for CI. | The labeler pipeline works. The mock backend is scaffolding only. Scores are now human-calibrated, but subtask boundaries and subgoal selections are not human-validated. |
| Subgoal images | WORKS | Current all-100 snapshot has 293 subgoal image rows, one per segment. | This is deterministic end-of-segment frame extraction, not generated future subgoal images like pi0.7. |
| Quality gate | WORKS, BUT HEURISTIC | Human-gold-label all-100 snapshot passes: `Episode pass rate: 1.000`, quality counts `{2: 6, 3: 15, 4: 30, 5: 49}`. Mock snapshot fails due score collapse. | Useful as a benchmark blocker. It catches obvious bad labels, but it is not a substitute for reliability measurement. |
| DuckDB query layer | WORKS | Five demo queries on the live snapshot returned in 21-28 ms. | This is one of the stronger parts of the project. Queryability is real. |
| Export cut | WORKS | Export produced a 13-episode cut manifest and label path map. | Good enough for deterministic downstream wiring. It does not yet export full LeRobot/RLDS artifacts. |
| Streamlit viewer | WORKS AT SMOKE LEVEL | Started on port 8765 and returned HTTP 200. | The app launches and can display snapshots. This was not a full UX QA pass. |
| Figures/status artifacts | WORKS | `figures/quality_summary.png`, `figures/snapshot_overview.png`, and `figures/scale_curve_human_gold_labels_100.png` exist. | Good for a demo status page. |
| Gold-set reliability scaffold | PARTIAL | `bridgeengine.goldset init/report` exists and tests pass. Kevin filled score labels for 100/100 clips and boundary timestamps for the 50-episode subset. | Score calibration is real. Boundary/subgoal reliability is now measured on the subset, but raw Gemini does not hit the preregistered reliability thresholds. |
| Perception comparison labelers | PARTIAL/BROKEN IN THIS VENV | `system_check` found local artifacts/checkpoints, but Python imports for `torch`, `sam2`, `video_depth_anything`, and `cotracker` are missing. | The wrappers exist, but this environment cannot run live perception extraction right now. This is not blocking the pi0.7-style main pipeline. |
| Benchmark runner | WORKS AS SMOKE-SCALE SCIENCE | `bridgeengine.benchmark.run_grid` and `bridgeengine.benchmark.scale_curve` run real LeWM frozen-adapter train/eval paths; the human-calibrated 100 scale curve writes CSV/plot/splits. | This is a real learned positive smoke result, but still only a small conditioning adapter with 100 local episodes and two seeds, not a robust robotics conclusion. |
| Head-to-head planner | WORKS AS PREREGISTRATION | `bridgeengine.benchmark.head_to_head` verifies BridgeEngine's 100 episodes match LeWM's `manifest_100.json`, writes fixed split files, and estimates runtime. | The final perceptive-vs-pi0.7 run is not launched yet because LeWM's cached evaluator must be adapted to the shared fixed split. |

### 2. Mock Quickstart End-to-End

I ran the no-API-key path in an isolated temp data root so it would not overwrite the current OpenAI labels:

```powershell
$RunRoot = "$env:TEMP\bridgeengine_value_report_20260604_002311"
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m bridgeengine.ingest --source bridge_v2 --episodes 13 --data-root $RunRoot
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot snap_2026_05_11_68c8cb784d --data-root $RunRoot --vlm-backend mock
.\.venv\Scripts\python.exe -m bridgeengine.inspect_labels --snapshot snap_2026_05_11_68c8cb784d --data-root $RunRoot
.\.venv\Scripts\python.exe -m bridgeengine.query --snapshot snap_2026_05_11_68c8cb784d --data-root $RunRoot
.\.venv\Scripts\python.exe -m bridgeengine.export --snapshot snap_2026_05_11_68c8cb784d --data-root $RunRoot --output-path "$RunRoot\training_cuts" --cut-name cut_mode_a_all_labels
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot snap_2026_05_11_68c8cb784d --data-root $RunRoot
```

Observed outputs:

```text
ingest: episode_count=13, snapshot_id=snap_2026_05_11_68c8cb784d
label: label_rows=65, labelers=[episode_metadata, subgoal_images, subtask_segmenter]
query: subtask_coverage=13 rows, metadata_quality=13 rows, subgoal_paths=39 rows, labeler_success_counts=3 rows, pi07_prompt_trace=1 row, all around 20-23 ms
export: cut_mode_a_all_labels, episode_count=13
quality_report: FAIL, quality counts={4: 13}, issue=score_dispersion collapse
```

Verdict: a new user can get from clone to labeled, queryable, exportable data without an API key. The catch is that mock labels are not meaningful labels. They are enough to validate plumbing and UI, not enough for benchmarking or demo claims.

### 3. Forge Comparison

Reference: Forge public docs describe it as a robotics data normalization toolkit that converts, inspects, visualizes, scores, filters, segments, and discovers datasets across formats including RLDS, LeRobot, Zarr, HDF5, MCAP, Rosbag, and RoboDM: https://arpitg1304.github.io/forge/

What BridgeEngine does that Forge does not appear to focus on:

- Generates pi0.7-style semantic conditioning fields: task/subtask prompt strings, episode quality, mistake, control mode, and subgoal keyframes.
- Uses a two-stage VLM observe-then-label flow with raw VLM provenance receipts.
- Has a semantic label quality gate that refuses benchmark runs when labels are templated, collapsed, or contradictory.
- Presents a direct VLA-conditioning research story: "do richer labels help the model?"

What Forge does better or more broadly:

- Real multi-format robotics dataset conversion. BridgeEngine does not currently write LeRobot/RLDS; it writes a local training-cut manifest.
- Dataset-scale inspection, filtering, scoring, and segmentation across many public formats.
- Proprioception-based quality metrics that do not require VLM calls.
- Dataset registry and likely a much better general user onboarding story.
- Broader visualization for robotics data formats, not just this BridgeData POC.

Plain comparison: BridgeEngine is not a Forge replacement. Forge looks like the better general robotics data toolkit. BridgeEngine's only defensible differentiation is semantic VLA-conditioning annotation and label-gated benchmark plumbing. If Forge adds VLM-derived pi0.7-style prompt metadata and provenance, BridgeEngine's standalone tool value drops sharply.

### 4. Real User Stories

1. Independent VLA researcher using BridgeData V2.
They would use BridgeEngine to generate subtask/metadata/subgoal conditioning rows, inspect them, query them, and export a deterministic cut before trying a real LEWM or OpenVLA-style ablation.

2. Robotics data/annotation team evaluating VLM label QA.
They would use the two-stage raw-output provenance, quality gate, and gold-set scaffold to test whether VLM-generated subtask labels are good enough to enter a training pipeline.

3. Kevin as an interview/demo artifact.
The strongest demo is not "this beats baseline." The strongest demo is "here is a contained data curation layer that turns raw robot episodes into queryable pi0.7-style training context, catches bad labels, and shows the exact label provenance."

I would not pitch this today to a production robotics team as a drop-in data platform. It is a credible research scaffold, not a production system.

## Section B - Scientific Value

### 5. Benchmark Result Attempt

Before calibration, I ran the Gemini-only curve:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.scale_curve --snapshot snap_2026_05_11_1dde3edf5d --sizes 25 50 100 --heldout-count 10 --quality-stratified --benchmark-seeds 0 1 --output-dir scale_results\gemini_100 --run
```

Then Kevin reviewed all 100 clips and changed the curation scores. He also reviewed timestamp boundaries on the 50-episode reliability subset. Subgoal gold labels were derived from those reviewed subtask end boundaries. I applied the human scores, reviewed boundaries, and derived subgoals to:

```text
snap_2026_05_11_1dde3edf5d_human_gold_labels
```

The original Gemini distribution was `{2: 5, 4: 9, 5: 86}`. Kevin's score calibration changed it to `{2: 6, 3: 15, 4: 30, 5: 49}`. Source Gemini reliability against Kevin's reviewed labels is: exact score agreement `0.42`, within-one score agreement `0.77`, mean subtask-boundary temporal IoU `0.683`, and derived subgoal-frame agreement `0.347`.

I then ran:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.scale_curve --snapshot snap_2026_05_11_1dde3edf5d_human_gold_labels --sizes 25 50 100 --heldout-count 10 --quality-stratified --benchmark-seeds 0 1 --output-dir scale_results\human_gold_labels_100 --run
```

It produced a real learned smoke-scale result using CUDA, Kevin's local LeWM checkpoint, a fixed mixed-quality held-out split per size, quality-stratified training mixtures, and 2 seeds per family:

```text
N=25:
baseline:                    0.038967
rich_text:                   0.038332
rich_text_metadata:          0.036301
rich_text_metadata_subgoal:  0.034922

N=50:
baseline:                    0.035255
rich_text:                   0.035348
rich_text_metadata:          0.033385
rich_text_metadata_subgoal:  0.030203

N=100:
baseline:                    0.021839
rich_text:                   0.022000
rich_text_metadata:          0.021849
rich_text_metadata_subgoal:  0.020468
```

Scientific verdict: POSITIVE SMOKE RESULT, STILL INCONCLUSIVE AS A GENERAL CLAIM.

The real benchmark path now works, and the human-score/boundary corrected scale curve shows the rich-text + metadata + subgoal family beating baseline in mean latent MSE at all three tested sizes. The deltas versus baseline are `-10.38%` at N=25, `-14.33%` at N=50, and `-6.28%` at N=100. Text-only and metadata-only do not hold a stable advantage at N=100.

This is not proof that pi0.7-style labels help in general. It is a small, two-seed trend probe with human-calibrated scores and partial human boundary correction. The biggest remaining weakness is that the raw VLM labels miss the preregistered reliability thresholds, so the public product needs calibration tooling rather than claims of out-of-the-box label quality.

### Minimal Steps To Make The Number Stronger

1. Run at least 3-5 seeds per family on the human-gold-label snapshot.
2. Adapt or wrap the cached LeWM A/B/D/E aux-head evaluator to the shared fixed split, then run the preregistered pi0.7-vs-perceptive head-to-head.
3. Download or expose more BridgeData V2 episodes and run N > 100 only after a fresh cost gate.
4. Replace the hashed text adapter with the actual language-conditioning path used by the downstream VLA or world-model stack, if available.

### 6. Proven vs Unproven

Proven right now:

- The local data-engine pipeline works on the 100-episode local BridgeData subset.
- Live OpenAI and Gemini labels can pass the current quality gate.
- Mock labels fail the quality gate for the right broad reason: collapsed metadata quality.
- Query, export, viewer startup, and tests work.
- Raw VLM provenance and label payload paths are recorded.
- The fake CPU-proxy benchmark has been removed from the normal path.
- The current benchmark feeds the four family contents into a real LeWM train/eval path and writes learned held-out latent MSE.
- Kevin reviewed all 100 clips and calibrated the score distribution.
- The human-score/boundary corrected 100 scale curve is positive for rich-text + metadata + subgoal conditioning in mean latent MSE.

Unproven right now:

- Rich-text conditioning robustly improves LEWM latent MSE beyond seed noise.
- Metadata robustly improves beyond subtask text.
- Subgoal keyframes robustly help beyond metadata.
- VLM-derived segmentation is accurate enough against human boundary labels.
- VLM-derived subgoal selections match human-selected subgoals.
- The current quality gate correlates with downstream model usefulness.
- The pipeline scales beyond this local 100-episode demo without cost, latency, or QA issues.
- The positive two-seed result survives more seeds and larger N.
- The LeWM aux-head CV conditions and BridgeEngine pi0.7 conditions have been evaluated on the same fixed split and plotted on one valid shared-axis figure.

## Section C - The Honest Critique

### 7. Strongest Fair Case Against Value

The harsh case is that BridgeEngine's positive 100-episode trend is still fragile. The label quality gate is heuristic and can pass labels that still contain semantic errors. Kevin calibrated scores and reviewed a boundary subset, but raw Gemini reliability misses the preregistered targets. The mock backend gives generic labels, and the real useful path depends on paid hosted VLM calls. Forge already covers much of the general robotics data-toolkit surface, including conversion, inspection, quality scoring, filtering, segmentation, visualization, and dataset discovery. BridgeEngine does not yet export standard training formats, the perceptive head-to-head still needs a fixed-split LeWM evaluator, and its LeWM benchmark is a frozen-adapter smoke test rather than a full downstream VLA training run. If someone asks "does this make robot models better?", the honest answer is "the corrected 100-episode smoke result is positive, but not proven beyond this setup."

### Honest Rebuttal

The rebuttal is that the project does have real tool value if judged as a focused VLA data-conditioning POC rather than a general toolkit or completed paper. It turns raw BridgeData episodes into queryable, versioned, pi0.7-shaped label artifacts with provenance. It catches bad labels before benchmark entry. It exposes labels visually enough for a human to judge and correct them. It now has a real human-corrected LeWM scale-curve result, and the result is materially stronger than the earlier 13-episode null and Gemini-only 100-episode curve. The Gemini labels are cheap enough to support scale probes, and the raw provenance plus gold tooling makes disagreement auditing possible. For an interview or research scaffold, that is meaningful. The missing pieces are the fixed-split perceptive head-to-head, larger scale, and a stronger downstream conditioning interface.

### 8. Highest-Leverage Next Steps

1. Human-score boundary and subgoal reliability.
Kevin calibrated scores. The next reliability gap is temporal IoU for subtask boundaries and agreement on subgoal selections.

2. Run the calibrated scale curve with more seeds and a larger N.
The current result is positive, but two seeds and 100 episodes are still smoke-scale.

3. Add real standard-format export or explicitly narrow the scope.
Either implement LeRobot/RLDS export or stop implying this is a general robotics data pipeline. Right now Forge wins the general-tooling comparison; BridgeEngine should either interoperate with that ecosystem or stay sharply framed as semantic-conditioning infrastructure.

## Final Assessment

The project is worth shipping as a public POC if the README and demo are honest: "This is a pi0.7-style annotation and curation layer for BridgeData with quality-gated benchmark plumbing, and the first human-corrected 100-episode scale curve shows a positive but unproven metadata+subgoal result." It is not worth shipping as a settled scientific result or as a general robotics data platform. The next serious milestone is adapting the cached LeWM aux-head evaluator to the shared fixed split, then running the 3-seed annotation-strategy head-to-head.
