# BridgeEngine Status

Current snapshot visualized: `snap_2026_05_11_a8256b172c`

## What Is Built

- Deterministic BridgeData V2 ingest into Parquet snapshots.
- Two-stage semantic labelers with swappable VLM backends (`openai`, `moondream`, `mock`).
- pi0.7-shaped labels: subtask segments, episode metadata, and end-of-segment subgoal images.
- Full label provenance, including raw VLM output paths.
- DuckDB query layer and deterministic training-cut export.
- Streamlit viewer with episode frames, labels, provenance, query outputs, and generated status figures.
- Quality-gated benchmark scaffold that refuses bad labels.
- Gold-set scaffold and reliability report command.

## What Is Blocked

- Stronger OpenAI-backend relabeling is blocked until `OPENAI_API_KEY` is available in the process or `.secrets/openai_api_key.txt`.
- The current Moondream-era labels still fail the quality gate.
- The real LEWM benchmark loop has not started.

## Current Quality State

The current labeled snapshot fails quality gates for repeated subtask text, placeholder metadata reasons, object-grounding gaps, and quality-score collapse.

![Quality Summary](figures/quality_summary.png)

## Snapshot Overview

![Snapshot Overview](figures/snapshot_overview.png)

## Benchmark State

The benchmark chart is intentionally a placeholder until labels pass the gate.

![Benchmark Placeholder](figures/benchmark_placeholder.png)

## Next Step

Set the OpenAI key without committing it:

```powershell
.\scripts\set_openai_key.ps1
```

Then rerun:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot snap_2026_05_11_68c8cb784d --vlm-backend openai --vlm-model gpt-5.5
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot snap_2026_05_11_68c8cb784d
.\.venv\Scripts\python.exe -m bridgeengine.figures --snapshot snap_2026_05_11_68c8cb784d --compare-snapshot snap_2026_05_11_a8256b172c
```
