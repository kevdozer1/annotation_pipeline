# BridgeEngine Value Report

**TOOL VALUE:** A user cloning this today gets a working local BridgeData-to-Parquet-to-VLM-labels-to-DuckDB/export/viewer POC; it can run end to end with the mock backend without an API key, but useful semantic labels require a real VLM backend.

**MEASURED EFFECT:** A real 100-episode Gemini-labeled LeWM scale curve now exists; richer conditioning trends better than baseline at 25/50/100 episodes, but the 100-episode gap is small, two-seed, and not yet human-gold validated.

**BOTTOM LINE:** This is worth shipping as a research/demo scaffold for robot-dataset conditioning, label QA, and curation workflow discussion, but it should still be framed as a smoke-scale trend probe rather than proof that pi0.7-style conditioning improves policy-world-model learning.

## Section A - Tool Value

### 1. Capability Audit

| Capability | Status | Evidence | Honest read |
|---|---|---|---|
| Venv/install | WORKS | `python -m pip install -e .` completed in the existing `.venv`. | Basic Python package setup is fine on this machine. |
| BridgeData ingest | WORKS | Ingest produced deterministic snapshots at 13, 50, remaining-50, and all-100 local episodes from `D:\bridgedata_v2_subset`. | Ingest is deterministic and can operate on Kevin's local BridgeData subset, which currently exposes 100 episodes. |
| Snapshot storage | WORKS | Snapshot writes `manifest.json`, `episodes.parquet`, `steps.parquet`, `sensors.parquet`, and `labels.parquet`. | Plain Parquet snapshot layer works for the POC. It is not a production dataset versioning system. |
| Two-stage semantic labeling | PARTIAL | Gemini-backed all-100 snapshot has 493 merged label rows: 100 metadata rows, 100 subtask rows, and 293 subgoal rows. Mock backend still works for CI. | The labeler pipeline works. The mock backend is scaffolding only. Gemini labels pass the gate but are top-heavy and not human-gold validated. |
| Subgoal images | WORKS | Current snapshot has 39 subgoal image rows, one per segment. | This is deterministic end-of-segment frame extraction, not generated future subgoal images like pi0.7. |
| Quality gate | WORKS, BUT HEURISTIC | Gemini all-100 snapshot passes: `Episode pass rate: 1.000`, quality counts `{2: 5, 4: 9, 5: 86}`. Mock snapshot fails due score collapse. | Useful as a benchmark blocker. It catches obvious bad labels, but it is not a substitute for human validation or score calibration. |
| DuckDB query layer | WORKS | Five demo queries on the live snapshot returned in 21-28 ms. | This is one of the stronger parts of the project. Queryability is real. |
| Export cut | WORKS | Export produced a 13-episode cut manifest and label path map. | Good enough for deterministic downstream wiring. It does not yet export full LeRobot/RLDS artifacts. |
| Streamlit viewer | WORKS AT SMOKE LEVEL | Started on port 8765 and returned HTTP 200. | The app launches and can display snapshots. This was not a full UX QA pass. |
| Figures/status artifacts | WORKS | `figures/quality_summary.png`, `figures/snapshot_overview.png`, and `figures/benchmark_placeholder.png` exist and regenerate. | Good for a demo status page. |
| Gold-set reliability scaffold | PARTIAL | `bridgeengine.goldset init/report` exists and tests pass. | The tool is wired, but there are no real human gold labels yet. |
| Perception comparison labelers | PARTIAL/BROKEN IN THIS VENV | `system_check` found local artifacts/checkpoints, but Python imports for `torch`, `sam2`, `video_depth_anything`, and `cotracker` are missing. | The wrappers exist, but this environment cannot run live perception extraction right now. This is not blocking the pi0.7-style main pipeline. |
| Benchmark runner | WORKS AS SMOKE-SCALE SCIENCE | `bridgeengine.benchmark.run_grid` and `bridgeengine.benchmark.scale_curve` run real LeWM frozen-adapter train/eval paths; the Gemini 100 scale curve writes CSV/plot/splits. | This is a real learned result, but still only a small conditioning adapter with 100 local episodes and two seeds, not a robust robotics conclusion. |

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

I ran:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.scale_curve --snapshot snap_2026_05_11_1dde3edf5d --sizes 25 50 100 --heldout-count 10 --quality-stratified --benchmark-seeds 0 1 --output-dir scale_results\gemini_100 --run
```

It produced a real learned smoke-scale result using CUDA, Kevin's local LeWM checkpoint, a fixed held-out split per size, quality-stratified training mixtures, and 2 seeds per family:

```text
N=25:
baseline:                    0.044892
rich_text:                   0.045292
rich_text_metadata:          0.041925
rich_text_metadata_subgoal:  0.041327

N=50:
baseline:                    0.022307
rich_text:                   0.022268
rich_text_metadata:          0.020579
rich_text_metadata_subgoal:  0.021179

N=100:
baseline:                    0.016242
rich_text:                   0.016079
rich_text_metadata:          0.016096
rich_text_metadata_subgoal:  0.015647
```

Scientific verdict: ENCOURAGING BUT STILL INCONCLUSIVE.

The real benchmark path now works, and the 100-episode Gemini scale curve shows richer conditioning beating baseline in mean latent MSE at all three tested sizes. The strongest mean gaps are at 25 and 50 episodes. At 100 episodes the subgoal family is 3.67% better than baseline, while metadata-only is 0.90% better than baseline.

This is not proof that pi0.7-style labels help in general. It is a small, two-seed, VLM-derived-label trend probe. The biggest weakness is that Gemini's score distribution is top-heavy: 86 of 100 episodes are `5/5`, and the held-out split contains only Gemini `5/5` episodes. That means the benchmark does not yet test whether metadata helps with mixed-quality held-out data.

### Minimal Steps To Make The Number Stronger

1. Fill a small gold set using `bridgeengine.goldset init`, then run `bridgeengine.goldset report` so the VLM-derived labels have measured reliability instead of just heuristic gate approval.
2. Calibrate Gemini scoring against the human gold set, especially the overuse of `5/5`.
3. Re-run the scale curve with a held-out split that includes low- and medium-quality episodes.
4. Download or expose more BridgeData V2 episodes and run N > 100 only after a fresh cost gate.
5. Replace the hashed text adapter with the actual language-conditioning path used by the downstream VLA or world-model stack, if available.

### 6. Proven vs Unproven

Proven right now:

- The local data-engine pipeline works on the 100-episode local BridgeData subset.
- Live OpenAI and Gemini labels can pass the current quality gate.
- Mock labels fail the quality gate for the right broad reason: collapsed metadata quality.
- Query, export, viewer startup, and tests work.
- Raw VLM provenance and label payload paths are recorded.
- The fake CPU-proxy benchmark has been removed from the normal path.
- The current benchmark feeds the four family contents into a real LeWM train/eval path and writes learned held-out latent MSE.
- The Gemini 100 scale curve trends positive for richer conditioning in mean latent MSE.

Unproven right now:

- Rich-text conditioning robustly improves LEWM latent MSE beyond seed noise.
- Metadata robustly improves beyond subtask text.
- Subgoal keyframes robustly help beyond metadata.
- VLM-derived segmentation is accurate enough against human labels.
- The current quality gate correlates with downstream model usefulness.
- The pipeline scales beyond this local 100-episode demo without cost, latency, or QA issues.
- Gemini's `5/5`-heavy scoring is calibrated enough for strong curation decisions.
- The perception comparison modules are runnable end to end in this venv; torch now imports, but SAM/VDA/CoTracker live dependencies still need their own check.

## Section C - The Honest Critique

### 7. Strongest Fair Case Against Value

The harsh case is that BridgeEngine's positive 100-episode trend is still fragile. The label quality gate is heuristic and can pass labels that still contain semantic errors. Gemini is cheap, but its scoring is top-heavy enough that the metadata signal may be blunted or miscalibrated. The mock backend gives generic labels, and the real useful path depends on paid hosted VLM calls. Forge already covers much of the general robotics data-toolkit surface, including conversion, inspection, quality scoring, filtering, segmentation, visualization, and dataset discovery. BridgeEngine does not yet export standard training formats, does not have human-gold reliability numbers, and its LeWM benchmark is a frozen-adapter smoke test rather than a full downstream VLA training run. If someone asks "does this make robot models better?", the honest answer is "the first 100-episode smoke trend is positive, but not proven."

### Honest Rebuttal

The rebuttal is that the project does have real tool value if judged as a focused VLA data-conditioning POC rather than a general toolkit or completed paper. It turns raw BridgeData episodes into queryable, versioned, pi0.7-shaped label artifacts with provenance. It catches bad labels before benchmark entry. It exposes labels visually enough for a human to judge them. It now has a real LeWM scale-curve result, and the result is directionally more interesting than the earlier 13-episode null. The Gemini labels are cheap enough to support scale probes, and the raw provenance makes disagreement auditing possible. For an interview or research scaffold, that is meaningful. The missing pieces are human-gold reliability measurement, larger scale, and a stronger downstream conditioning interface.

### 8. Highest-Leverage Next Steps

1. Build and score a human gold set for 13-50 episodes.
Use the existing gold-set scaffold to measure temporal IoU for subtask boundaries, quality-score agreement, and subgoal agreement. This makes label reliability discussable instead of impressionistic.

2. Calibrate Gemini scoring and rerun the 100-episode scale curve.
The biggest immediate risk is not cost; it is that Gemini overuses `5/5`. Fix the rubric against human gold labels and rerun before treating the trend as stable.

3. Add real standard-format export or explicitly narrow the scope.
Either implement LeRobot/RLDS export or stop implying this is a general robotics data pipeline. Right now Forge wins the general-tooling comparison; BridgeEngine should either interoperate with that ecosystem or stay sharply framed as semantic-conditioning infrastructure.

## Final Assessment

The project is worth shipping as a public POC if the README and demo are honest: "This is a pi0.7-style annotation and curation layer for BridgeData with quality-gated benchmark plumbing, and the first 100-episode Gemini scale curve shows an encouraging but unproven conditioning trend." It is not worth shipping as a settled scientific result or as a general robotics data platform. The next serious milestone is measured label reliability plus calibrated scoring.
