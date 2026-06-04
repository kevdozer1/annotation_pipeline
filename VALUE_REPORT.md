# BridgeEngine Value Report

**TOOL VALUE:** A user cloning this today gets a working local BridgeData-to-Parquet-to-VLM-labels-to-DuckDB/export/viewer POC; it can run end to end with the mock backend without an API key, but useful semantic labels require a real VLM backend.

**MEASURED EFFECT:** A real 13-episode LeWM frozen-adapter smoke ablation now exists; baseline has the best mean latent MSE, and rich-text + metadata is +1.4% worse than baseline, within seed noise.

**BOTTOM LINE:** This is worth shipping as a research/demo scaffold for robot-dataset conditioning, label QA, and curation workflow discussion, but the current smoke result does not support claiming pi0.7-style conditioning improves policy-world-model learning.

## Section A - Tool Value

### 1. Capability Audit

| Capability | Status | Evidence | Honest read |
|---|---|---|---|
| Venv/install | WORKS | `python -m pip install -e .` completed in the existing `.venv`. | Basic Python package setup is fine on this machine. |
| BridgeData ingest | WORKS | Mock quickstart ingest on `D:\bridgedata_v2_subset` produced `snap_2026_05_11_68c8cb784d` with 13 episodes and 334 steps. | Ingest is deterministic and can operate on Kevin's local BridgeData subset. |
| Snapshot storage | WORKS | Snapshot writes `manifest.json`, `episodes.parquet`, `steps.parquet`, `sensors.parquet`, and `labels.parquet`. | Plain Parquet snapshot layer works for the POC. It is not a production dataset versioning system. |
| Two-stage semantic labeling | PARTIAL | OpenAI-backed snapshot has 65 label rows: 13 metadata rows, 13 subtask rows, 39 subgoal rows. Mock backend also writes 65 rows. | The labeler pipeline works. The mock backend is scaffolding only. OpenAI labels look materially better and pass the gate, but are not human-gold validated. |
| Subgoal images | WORKS | Current snapshot has 39 subgoal image rows, one per segment. | This is deterministic end-of-segment frame extraction, not generated future subgoal images like pi0.7. |
| Quality gate | WORKS, BUT HEURISTIC | Live OpenAI snapshot passes: `Episode pass rate: 1.000`, quality counts `{1: 1, 3: 3, 4: 3, 5: 6}`. Mock snapshot fails due score collapse `{4: 13}`. | Useful as a benchmark blocker. It catches obvious bad labels, but it is not a substitute for human validation. |
| DuckDB query layer | WORKS | Five demo queries on the live snapshot returned in 21-28 ms. | This is one of the stronger parts of the project. Queryability is real. |
| Export cut | WORKS | Export produced a 13-episode cut manifest and label path map. | Good enough for deterministic downstream wiring. It does not yet export full LeRobot/RLDS artifacts. |
| Streamlit viewer | WORKS AT SMOKE LEVEL | Started on port 8765 and returned HTTP 200. | The app launches and can display snapshots. This was not a full UX QA pass. |
| Figures/status artifacts | WORKS | `figures/quality_summary.png`, `figures/snapshot_overview.png`, and `figures/benchmark_placeholder.png` exist and regenerate. | Good for a demo status page. |
| Gold-set reliability scaffold | PARTIAL | `bridgeengine.goldset init/report` exists and tests pass. | The tool is wired, but there are no real human gold labels yet. |
| Perception comparison labelers | PARTIAL/BROKEN IN THIS VENV | `system_check` found local artifacts/checkpoints, but Python imports for `torch`, `sam2`, `video_depth_anything`, and `cotracker` are missing. | The wrappers exist, but this environment cannot run live perception extraction right now. This is not blocking the pi0.7-style main pipeline. |
| Benchmark runner | WORKS AS SMOKE-SCALE SCIENCE | `bridgeengine.benchmark.run_grid` now runs 4 families x 3 seeds through a real LeWM frozen-adapter train/eval path and writes CSV/plot/summary. | This is a real learned result, but still only a 13-episode smoke ablation with a small conditioning adapter, not a robust robotics conclusion. |

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
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.run_grid --snapshot snap_2026_05_11_68c8cb784d --output-dir bench_results
```

It produced a real learned smoke-scale result using CUDA, Kevin's local LeWM checkpoint, a fixed 10 train / 3 held-out episode split, and 3 seeds per family:

```text
baseline mean:                    0.039541 +/- 0.004444
rich_text mean:                   0.042126 +/- 0.004554
rich_text_metadata mean:          0.040083 +/- 0.003849
rich_text_metadata_subgoal mean:  0.039926 +/- 0.004309
```

Scientific verdict: INCONCLUSIVE-TO-NEGATIVE.

The real benchmark path now works, but the current result does not show the desired pi0.7-style effect. Baseline has the best mean. Rich-text + metadata is 1.4% worse than baseline and well within seed noise by the simple std-sum check. The subgoal family is also worse than baseline by about 1.0%.

This is not proof that pi0.7-style labels do not help in general. It is evidence that this particular 13-episode, VLM-derived-label, frozen-LeWM-adapter setup does not yet produce a positive metadata result.

### Minimal Steps To Make The Number Stronger

1. Fill a small gold set using `bridgeengine.goldset init`, then run `bridgeengine.goldset report` so the VLM-derived labels have measured reliability instead of just heuristic gate approval.
2. Run the same benchmark on 50-100 episodes, still with a fixed split and at least 3 seeds.
3. Replace the hashed text adapter with the actual language-conditioning path used by the downstream VLA or world-model stack, if available.
4. Test whether human-corrected subtask boundaries change the result. That isolates VLM label quality from the conditioning idea itself.

### 6. Proven vs Unproven

Proven right now:

- The local data-engine pipeline works on the 13-episode BridgeData subset.
- Live OpenAI labels can pass the current quality gate.
- Mock labels fail the quality gate for the right broad reason: collapsed metadata quality.
- Query, export, viewer startup, and tests work.
- Raw VLM provenance and label payload paths are recorded.
- The fake CPU-proxy benchmark has been removed from the normal path.
- The current benchmark feeds the four family contents into a real LeWM train/eval path and writes learned held-out latent MSE.

Unproven right now:

- Rich-text conditioning improves LEWM latent MSE.
- Metadata improves beyond subtask text.
- Subgoal keyframes help beyond metadata.
- VLM-derived segmentation is accurate enough against human labels.
- The current quality gate correlates with downstream model usefulness.
- The pipeline scales beyond this local 13-episode demo without cost, latency, or QA issues.
- The perception comparison modules are runnable end to end in this venv; torch now imports, but SAM/VDA/CoTracker live dependencies still need their own check.

## Section C - The Honest Critique

### 7. Strongest Fair Case Against Value

The harsh case is that BridgeEngine's first real benchmark result is negative for the headline conditioning claim. The label quality gate is heuristic and can pass labels that still contain semantic errors. The mock backend gives generic labels, and the real useful path depends on paid hosted VLM calls. Forge already covers much of the general robotics data-toolkit surface, including conversion, inspection, quality scoring, filtering, segmentation, visualization, and dataset discovery. BridgeEngine does not yet export standard training formats, does not have human-gold reliability numbers, and its LeWM benchmark is a frozen-adapter smoke test rather than a full downstream VLA training run. If someone asks "does this make robot models better?", the honest answer is "not shown yet."

### Honest Rebuttal

The rebuttal is that the project does have real tool value if judged as a focused VLA data-conditioning POC rather than a general toolkit or completed paper. It turns raw BridgeData episodes into queryable, versioned, pi0.7-shaped label artifacts with provenance. It catches bad labels before benchmark entry. It exposes labels visually enough for a human to judge them. It has a real LeWM smoke benchmark now, and the result being negative is still useful because it prevents a weak demo claim. The OpenAI-backed labels are visibly different from the bad templated fallback and the current quality distribution is non-collapsed. For an interview or research scaffold, that is meaningful. The missing pieces are human-gold reliability measurement, larger scale, and a stronger downstream conditioning interface.

### 8. Highest-Leverage Next Steps

1. Build and score a human gold set for 13-50 episodes.
Use the existing gold-set scaffold to measure temporal IoU for subtask boundaries, quality-score agreement, and subgoal agreement. This makes label reliability discussable instead of impressionistic.

2. Scale the real benchmark to 50-100 episodes.
The current 13-episode split is useful as a smoke test, but seed noise is large enough that small conditioning effects are hard to interpret.

3. Add real standard-format export or explicitly narrow the scope.
Either implement LeRobot/RLDS export or stop implying this is a general robotics data pipeline. Right now Forge wins the general-tooling comparison; BridgeEngine should either interoperate with that ecosystem or stay sharply framed as semantic-conditioning infrastructure.

## Final Assessment

The project is worth shipping as a public POC if the README and demo are honest: "This is a pi0.7-style annotation and curation layer for BridgeData with quality-gated benchmark plumbing, and the first real smoke ablation did not show a metadata win." It is not worth shipping as a positive scientific result or as a general robotics data platform. The next serious milestone is measured label reliability plus a larger real LeWM ablation.
