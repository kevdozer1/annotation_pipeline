# BridgeEngine Final-State Assessment

Last updated: 2026-06-05

## What Changed In This Pass

The Streamlit viewer now has a `Calibration` tab for human score review. It lets Kevin move through the local 100-episode snapshot, watch each episode video or frame sequence, inspect the auto subtask timeline, see the model-generated metadata, assign a calibrated 1-5 curation score, mark visible mistakes, write notes, and optionally accept auto subtask boundaries or subgoal frames for reliability scoring.

Reviews are saved to:

```text
bridgeengine_data/snapshots/<snapshot_id>/gold/calibration_gold.json
```

This is intentionally the same gold-set JSON format used by `bridgeengine.goldset report`, so the GUI is not just a viewer. It creates measurable calibration data.

## How To Use The Calibration GUI

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

## How Close This Is To The Final Useful Project

For an internal demo and research scaffold: about 80 percent there. The remaining work is mostly calibration, documentation, and one more evidence pass.

For a genuinely useful open-source tool that other foundation-model/data teams could clone and adapt: about 55 percent there. The core ideas and local implementation exist, but the repo still assumes Kevin's local paths, BridgeData-style files, and a specific LeWM evaluation path.

For a publishable scientific claim that pi0.7-style labels improve robot model learning: about 35 percent there. There is now a real 100-episode positive trend, but it is two-seed, Gemini-score-heavy, and not human-gold calibrated.

## What Still Needs To Happen

1. Human-calibrate the 100-episode set.
   The immediate priority is not more API calls. It is reviewing enough episodes to measure whether Gemini's `5/5`-heavy grading matches Kevin's judgment. This GUI is built for exactly that.

2. Recompute reliability and calibrate the rubric.
   After review, run `bridgeengine.goldset report`. If Gemini is over-scoring, update the scoring rubric or prompt and rerun only the necessary labels.

3. Rerun the 100-episode scale curve after calibration.
   The current trend is encouraging but weak. The next result should report whether calibrated score thresholds change the conditioning gap.

4. Make the repo modular.
   The future public repo should separate data adapters, VLM backends, scoring policies, exporters, and evaluators. This repo has most of those pieces, but they are still too tied to BridgeData and local snapshot conventions.

5. Add standard export.
   A public version should export LeRobot and/or RLDS if it wants to be adopted by robot-learning people. The current export is deterministic and useful, but BridgeEngine-native.

6. Scale beyond 100 only after a new cost gate.
   Gemini makes labeling cheap enough to consider larger runs, but the local SSD source only exposes 100 episodes. More scale requires more BridgeData V2 locally available.

## Honest Value Judgment

The strongest real value is not the model result yet. The strongest value is the workflow: a robot dataset can be ingested, semantically labeled, inspected, human-calibrated, query-filtered, and benchmarked without pretending the labels are automatically trustworthy.

That is a credible foundation-model data-engineering story. It is the kind of tool someone entering robot foundation-model work could learn from because it makes the hidden annotation and curation loop concrete.

The weak point is still calibration. Until the human gold set exists, the project can say "we built the pipeline and found an encouraging smoke trend." It cannot yet say "these labels are reliable" or "this conditioning reliably improves robot learning."
