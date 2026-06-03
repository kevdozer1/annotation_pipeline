# BridgeEngine Moondream Pivot Session Report

Date: 2026-05-12

## Current Handoff

The Streamlit viewer is running here:

```powershell
http://localhost:8502
```

To restart it yourself:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\python.exe -m streamlit run bridgeengine/viewer/app.py --server.port 8502
```

The current live-labeled snapshot is:

```text
snap_2026_05_11_a8256b172c
```

Useful validation commands:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.inspect_labels --snapshot snap_2026_05_11_a8256b172c
.\.venv\Scripts\python.exe -m bridgeengine.query --snapshot snap_2026_05_11_a8256b172c
.\.venv\Scripts\python.exe -m bridgeengine.export --snapshot snap_2026_05_11_a8256b172c --output-path training_cuts --cut-name cut_mode_a_all_labels
.\.venv\Scripts\python.exe -m pytest
```

Do not run the benchmark yet. The live labels are real and inspectable, but the semantic quality is not strong enough to support a benchmark claim. The benchmark runner now enforces this and fails on the current snapshot with repeated-subtask quality-gate errors.

## What We Built In This Session

We fixed the Moondream integration path and moved the project off the deterministic fallback labels that overseer correctly rejected.

Specific work completed:

- Added a live `MoondreamClient` for `https://api.moondream.ai/v1/query`.
- Fixed the PowerShell UTF-8 BOM bug that caused `UnicodeEncodeError: 'latin-1' codec can't encode character '\ufeff'` in the auth header.
- Added `scripts/set_moondream_key.ps1`, which saves the API key to `.secrets/moondream_api_key.txt`; `.secrets/` is gitignored.
- Updated the client to read secrets from `MOONDREAM_API_KEY`, `.secrets/moondream_api_key.txt`, or `.env`, stripping BOM and quotes safely.
- Re-ran live Moondream labeling on all 13 BridgeData V2 subset episodes.
- Saved raw VLM responses under `bridgeengine_data/snapshots/<snapshot_id>/raw_vlm_outputs/<episode_id>/`.
- Added `bridgeengine.inspect_labels`, a compact CLI for eyeballing 3-4 representative episodes before spending compute.
- Added a benchmark guard: `bridgeengine.benchmark.run_grid` now refuses fallback/scaffolding labels unless explicitly overridden.
- Added a second benchmark quality gate that refuses current live labels with repeated subtask text.
- Pivoted the label schema toward pi0.7-style data: subtask segments, episode metadata, and subgoal image paths.
- Preserved perception labelers under `bridgeengine.labelers.perceptive` as comparison modules, not the main benchmark path.
- Updated README and deviations so the project no longer claims deterministic fallback labels are acceptable demo labels.
- Archived the pre-pivot benchmark CSV at `bench_results/pre_pivot/bench_results.csv`.

## Current Artifacts

The current snapshot contains:

```text
episodes: 13
steps: 334
labels: 65
subtask_segmenter rows: 13
episode_metadata rows: 13
subgoal_images rows: 39
```

The live labeler runtime in the latest full run was:

```text
subtask_segmenter: 22.347318 seconds
episode_metadata: 8.601785 seconds
subgoal_images: 0.373957 seconds
```

The DuckDB queries completed successfully, all under roughly 25 ms locally:

```text
subtask_coverage
metadata_quality
subgoal_paths
labeler_success_counts
pi07_prompt_trace
```

The training cut export completed successfully:

```text
training_cuts/cut_mode_a_all_labels
```

## Important Implementation Detail

The first live Moondream run produced task-specific words, but the temporal boundaries were not acceptable. Examples included gappy ranges and repeated phrases. To avoid pretending those frame-index guesses were good, the segmenter now uses robot gripper/action transitions for temporal boundaries and asks Moondream to name the subtask for each interval.

This is recorded in provenance as:

```text
boundary_source: gripper_transition_vlm_text
```

That means the current system should be described as action-transition segmentation with live VLM semantic labeling, not as pure VLM temporal segmentation.

## Quality Audit

The current live labels are substantially better than the rejected deterministic fallback, but they are not benchmark-grade yet.

Good signs:

- No fallback/scaffolding labels were detected.
- Every subtask segmentation is contiguous from step 0 to the final step.
- Boundaries vary by episode because they are derived from gripper/action transitions.
- Subgoal images were extracted for every segment.
- Raw VLM responses are saved, so provenance is inspectable.
- Metadata is not constant: quality and mistake values vary across the 13 episodes.

Current audit numbers:

```text
segment_count_distribution: {3: 13}
contiguous_all: True
episodes_with_duplicate_text: 6 / 13
quality_counts: {1: 4, 4: 9}
mistake_counts: {False: 9, True: 4}
```

Concerning signs:

- Six episodes still have repeated subtask text, for example repeated "place cup in sink" across multiple intervals.
- Some labels are object-confused, for example corn/pot cases can produce phrases about grasping the pot instead of the corn.
- The metadata judge is too coarse: it mostly emits quality 1 or 4.
- Some metadata explanations contradict the score, for example a successful reason paired with `quality=1`.
- Moondream sometimes returns placeholder-ish reasoning, such as "One short sentence."

## How To Appraise The Results

The project infrastructure is good. The live pipeline works end to end, produces provenance, exposes the labels in Streamlit, supports SQL querying, and exports a deterministic training cut. This is enough to show the current capabilities of the system as an annotation engine.

The annotation quality is mixed. It is good enough to demonstrate the data substrate, snapshotting, provenance, viewer, and pi0.7-style prompt format. It is not good enough to claim that rich-text annotation improves LEWM latent MSE. The current Moondream labels would make the benchmark hard to interpret because text repetition and metadata contradictions could dominate the result.

The honest demo framing is:

```text
BridgeEngine now produces pi0.7-shaped annotations on BridgeData V2 with live VLM provenance. The infrastructure is ready; the current Moondream semantic labels reveal exactly why provenance and pre-benchmark inspection matter. The next research step is improving or validating subtask/metadata label quality before running the 12-run label-value benchmark.
```

## What Not To Claim Yet

Do not claim:

- That rich-text labels beat baseline.
- That Moondream alone provides reliable temporal segmentation.
- That the current metadata judge is calibrated.
- That the current subgoal condition has been evaluated.

Do claim:

- The data engine is working.
- Live Moondream labels are integrated.
- Fallback labels and current repeated-subtask live labels are blocked from benchmark runs.
- Raw VLM outputs and label provenance are stored.
- The viewer makes label quality auditable before compute is spent.

## Recommended Next Steps

1. Improve the segmenter prompt with a two-stage VLM flow:
   - First ask Moondream for frame-by-frame observations.
   - Then ask it to name each action-transition interval using those observations.

2. Add automatic quality gates:
   - Reject repeated `subtask_text` within an episode.
   - Reject metadata rows where `quality <= 2` but the reason says the task succeeded.
   - Reject placeholder reasons.
   - Flag object mismatch between task instruction and subtask text.

3. Consider a stronger hosted VLM for the semantic layer:
   - Moondream is fast and cheap, but the current outputs are not reliable enough for the final benchmark claim.
   - A stronger multimodal model could be used only for the 13-episode POC label generation.

4. Add a small human validation pass:
   - Correct 13 episodes manually or semi-manually in a JSON file.
   - Keep the original Moondream output as provenance.
   - Benchmark both `vlm_raw` and `human_validated` labels if time permits.

5. Only after label quality passes inspection, run:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.run_grid --snapshot snap_2026_05_11_a8256b172c --output-dir bench_results
```

## Demo Path

Start with the Streamlit viewer:

```powershell
.\.venv\Scripts\python.exe -m streamlit run bridgeengine/viewer/app.py --server.port 8502
```

In the demo, open an episode and show:

- the task instruction,
- the pi0.7 prompt preview,
- subtask segments,
- metadata payload,
- subgoal images,
- raw VLM provenance paths.

Then run:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.inspect_labels --snapshot snap_2026_05_11_a8256b172c
```

That command is the strongest credibility move: it shows the system does not hide bad labels, and it blocks benchmark runs until the labels are good enough.
