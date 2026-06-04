# BridgeEngine Status

Current OpenAI-backed snapshot visualized: `snap_2026_05_11_68c8cb784d`

Comparison snapshot: `snap_2026_05_11_a8256b172c`

## What Is Built

- Deterministic BridgeData V2 ingest into Parquet snapshots.
- Two-stage semantic labelers with swappable VLM backends (`openai`, `moondream`, `mock`).
- pi0.7-shaped labels: subtask segments, episode metadata, and end-of-segment subgoal images.
- Full label provenance, including raw VLM output paths stored outside git.
- DuckDB query layer and deterministic training-cut export.
- Streamlit viewer with episode frames, labels, provenance, query outputs, and generated status figures.
- Quality-gated benchmark runner that refuses bad labels, then runs a real LeWM frozen-adapter smoke ablation.
- Gold-set scaffold and reliability report command.

## Current Quality State

The OpenAI-backed relabel passes the quality gate on all 13 episodes.

```text
Quality gate: PASS
Episode pass rate: 1.000
Quality counts: {1: 1, 3: 3, 4: 3, 5: 6}
```

The previous failure modes were:

- Repeated templated subtask text: resolved by using live OpenAI two-stage observe-then-label output instead of deterministic fallback labels.
- Metadata score/reason contradictions: resolved in the labels and by making the gate distinguish negated phrases such as "no wrong destination" from real failure claims.
- Object-grounding gaps: resolved in the label set; the gate now ignores action/time words such as `pickup`, `before`, and `withdraw` as non-object tokens.
- Quality-score collapse: resolved. The current distribution spans 1, 3, 4, and 5.

![Quality Summary](figures/quality_summary.png)

## Snapshot Overview

![Snapshot Overview](figures/snapshot_overview.png)

## Benchmark State

The fake CPU-proxy benchmark has been replaced. The current chart is a real LeWM frozen-adapter smoke ablation over 13 episodes, with a fixed 10 train / 3 held-out episode split and 3 seeds per family.

Mean held-out latent MSE:

| Family | Mean | Std | Delta vs baseline |
|---|---:|---:|---:|
| baseline | 0.039541 | 0.004444 | 0.0% |
| rich_text | 0.042126 | 0.004554 | +6.5% |
| rich_text_metadata | 0.040083 | 0.003849 | +1.4% |
| rich_text_metadata_subgoal | 0.039926 | 0.004309 | +1.0% |

Interpretation: baseline is the best mean. The metadata family does not beat baseline beyond seed noise on this smoke split. This is evidence against making a positive conditioning claim at 13 episodes, not a final result about pi0.7-style annotation in general.

![Benchmark Results](figures/benchmark_placeholder.png)

## Human Inspection

Three full example payloads are committed at:

```text
examples/openai_label_samples_snap_2026_05_11_68c8cb784d.json
```

The real subgoal frames and raw VLM outputs remain local-only and ignored by git.

## Validation Run

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.quality_report --snapshot snap_2026_05_11_68c8cb784d
.\.venv\Scripts\python.exe -m bridgeengine.figures --snapshot snap_2026_05_11_68c8cb784d --compare-snapshot snap_2026_05_11_a8256b172c
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.run_grid --snapshot snap_2026_05_11_68c8cb784d --output-dir bench_results
.\.venv\Scripts\python.exe -m pytest
```

Latest local result: real grid completed on CUDA, then pytest passed locally.

## What Is Still Blocked

- Human gold labels are still missing, so label reliability is heuristic-gated rather than measured against human review.
- The reliability report is wired, but Kevin still needs to fill the gold labels; the example file is intentionally not a real gold set.
- No strong scientific claim should be made from this grid. It is a smoke-scale ablation and the positive pi0.7-style metadata effect did not appear on this split.
