# BridgeEngine Final-State Assessment

Last updated: 2026-06-05

## Latest GUI Update

The preferred human-review interface is now the dedicated browser GUI:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\python.exe -m bridgeengine.review_gui --snapshot snap_2026_05_11_1dde3edf5d --port 8787
```

Open:

```text
http://127.0.0.1:8787
```

This GUI is better than the Streamlit tab for calibration review. It presents a persistent video player, a scrollable episode queue, auto metadata, subtask timeline, subgoal frames, score controls, and a `Save review and next` button that immediately advances to the next unreviewed episode.

## What Changed In The Calibration Pass

The Streamlit viewer now has a `Calibration` tab for human score review. It lets Kevin move through the local 100-episode snapshot, watch each episode video or frame sequence, inspect the auto subtask timeline, see the model-generated metadata, assign a calibrated 1-5 curation score, mark visible mistakes, write notes, and optionally accept auto subtask boundaries or subgoal frames for reliability scoring.

Reviews are saved to:

```text
bridgeengine_data/snapshots/<snapshot_id>/gold/calibration_gold.json
```

This is intentionally the same gold-set JSON format used by `bridgeengine.goldset report`, so the GUI is not just a viewer. It creates measurable calibration data.

## How To Use The Streamlit Calibration GUI

Start the viewer:

```powershell
cd C:\Users\Kevin\projects\annotation_pipeline
.\.venv\Scripts\streamlit.exe run bridgeengine\viewer\app.py --server.port 8765
```

Open:

```text
http://localhost:8765
```

Select the all-100 Gemini snapshot:

```text
snap_2026_05_11_1dde3edf5d
```

Open the `Calibration` tab. Use `Next`, `Previous`, and `Next unreviewed` to move through the set. Save a calibrated score for each episode. Keep the score semantics simple:

- `1 - clear reject`: not usable as training data for visible manipulation boundaries
- `2 - reject`: weak or wrong interaction, but not totally empty
- `3 - near reject`: some useful evidence, but too occluded, incomplete, or ambiguous for the main keep set
- `4 - near keep`: usable, with some imperfection
- `5 - clear keep`: clean visible cause-effect interaction, clear boundaries, useful training signal

Run reliability after saving reviews:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.goldset report `
  --snapshot snap_2026_05_11_1dde3edf5d `
  --gold-file bridgeengine_data\snapshots\snap_2026_05_11_1dde3edf5d\gold\calibration_gold.json
```

## Current Capability Level

BridgeEngine is now a serious internal maximum-slice POC. It can:

- ingest local BridgeData-style episodes into deterministic Parquet snapshots
- label episodes with hosted VLM backends, currently OpenAI and Gemini
- produce pi0.7-style subtask, metadata, mistake, control-mode, and subgoal-frame annotations
- preserve raw VLM provenance
- quality-gate labels before benchmark entry
- compare model graders
- estimate labeling cost
- visualize episodes and annotations
- create human gold calibration data through the GUI
- run real LeWM frozen-adapter smoke benchmarks and scale curves
- score value/anomaly and run tiered-compression probes

That is enough to be useful to a foundation-model-adjacent researcher evaluating annotation strategy, curation heuristics, and label-value experiments on small robot datasets.

It is not yet a polished public robotics data product.

## Completed Human Score Calibration

Kevin reviewed all 100 clips in the dedicated GUI. The review pass changed scores only:

- reviewed clips: `100 / 100`
- score changes versus Gemini auto curation: `58 / 100`
- review notes intentionally entered: `0`
- calibration reasons intentionally entered: `0`
- subtask-boundary accept toggles used: `0`
- subgoal accept toggles used: `0`

The original Gemini score distribution was `{2: 5, 4: 9, 5: 86}`. Kevin's calibrated score distribution is `{2: 6, 3: 15, 4: 30, 5: 49}`.

Those scores were applied to a cloned snapshot:

```text
snap_2026_05_11_1dde3edf5d_human_calibrated
```

The calibrated snapshot passes the quality gate:

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {2: 6, 3: 15, 4: 30, 5: 49}
```

Important caveat: this is score calibration, not full gold annotation. Boundary IoU and subgoal-selection agreement are still unmeasured.

## How Close This Is To The Final Useful Project

For an internal demo and research scaffold: about 90 percent there. The score calibration and calibrated scale curve are now done; the remaining work is mostly boundary/subgoal reliability, documentation polish, and one more seed/scale pass.

For a genuinely useful open-source tool that other foundation-model/data teams could clone and adapt: about 60 percent there. The core ideas and local implementation exist, but the repo still assumes Kevin's local paths, BridgeData-style files, and a specific LeWM evaluation path.

For a publishable scientific claim that pi0.7-style labels improve robot model learning: about 50 percent there. There is now a real score-calibrated 100-episode positive trend, but it is two-seed and still lacks boundary/subgoal gold validation.

## What Still Needs To Happen

1. Human-review boundaries and subgoal selections.
   The score calibration is complete. The remaining reliability gap is whether VLM-derived segment boundaries and subgoal frames match human judgment.

2. Run more seeds on the calibrated scale curve.
   The current trend is positive but still two-seed. A three- to five-seed rerun would make the variance more defensible.

3. Scale beyond 100 only after a new cost gate.
   The current local source exposes 100 episodes. Larger N requires more BridgeData V2 locally available.

4. Make the repo modular.
   The future public repo should separate data adapters, VLM backends, scoring policies, exporters, and evaluators. This repo has most of those pieces, but they are still too tied to BridgeData and local snapshot conventions.

5. Add standard export.
   A public version should export LeRobot and/or RLDS if it wants to be adopted by robot-learning people. The current export is deterministic and useful, but BridgeEngine-native.

6. Scale beyond 100 only after a new cost gate.
   Gemini makes labeling cheap enough to consider larger runs, but the local SSD source only exposes 100 episodes. More scale requires more BridgeData V2 locally available.

## Honest Value Judgment

The strongest real value is not the model result yet. The strongest value is the workflow: a robot dataset can be ingested, semantically labeled, inspected, human-calibrated, query-filtered, and benchmarked without pretending the labels are automatically trustworthy.

That is a credible foundation-model data-engineering story. It is the kind of tool someone entering robot foundation-model work could learn from because it makes the hidden annotation and curation loop concrete.

The weak point is now narrower: boundary and subgoal reliability. The project can say "we built the pipeline, score-calibrated all 100 clips, and found a positive metadata+subgoal smoke trend." It cannot yet say "all label fields are reliable" or "this conditioning reliably improves robot learning."
