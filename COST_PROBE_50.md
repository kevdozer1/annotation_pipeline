# BridgeEngine 50-Episode Cost Probe

Snapshot: `snap_2026_05_11_de43f7bf0b`

This is the spend gate before any larger BridgeData V2 labeling run. The mounted SSD source currently exposes 100 local episodes at `D:\bridgedata_v2_subset`, not the full approximately 60k BridgeData V2 corpus. This probe labels the first 50 deterministic episodes from that mounted subset and projects larger runs from observed token usage and serial wall-clock.

## Labeling Run

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.ingest --source D:\bridgedata_v2_subset --episodes 50 --episode-offset 0
.\.venv\Scripts\python.exe -m bridgeengine.label --snapshot snap_2026_05_11_de43f7bf0b --vlm-backend openai --vlm-model gpt-5.5
.\.venv\Scripts\python.exe -m bridgeengine.cost_probe --snapshot snap_2026_05_11_de43f7bf0b --projection 200 --projection 1000 --projection 60000
```

Observed label rows:

- `episode_metadata`: 50 rows
- `subtask_segmenter`: 50 rows
- `subgoal_images`: 147 rows
- total: 247 rows

Observed runtime:

- `subtask_segmenter`: 1163.916551 seconds
- `episode_metadata`: 804.767220 seconds
- `subgoal_images`: 1.737524 seconds
- total serial wall-clock: 1970.421295 seconds
- per episode: 39.408426 seconds

## Cost Estimate

The probe sums token usage from raw OpenAI Responses receipts under the snapshot's ignored `raw_vlm_outputs` directory.

Pricing assumptions:

- model: `gpt-5.5`
- input: `$5.00 / 1M tokens`
- cached input: `$0.50 / 1M tokens`
- output: `$30.00 / 1M tokens`
- source: `https://developers.openai.com/api/docs/models/gpt-5.5`

Observed token totals:

- requests: 212
- input tokens: 188,256
- cached input tokens: 0
- output tokens: 74,150
- reasoning tokens: 40,843
- total tokens: 262,406

Observed estimate:

- total cost for 50: `$3.165780`
- cost per episode: `$0.063316`

Projected serial labeling:

| Episodes | Estimated cost | Estimated serial wall-clock |
|---:|---:|---:|
| 200 | `$12.66` | 2.19 hours |
| 1,000 | `$63.32` | 10.95 hours |
| 60,000 | `$3,798.94` | 656.81 hours |

Do not start a larger run until Kevin chooses the target N.

## Quality Distribution

Quality gate result: `FAIL`

Episode pass rate: `0.920`

Quality counts:

| Quality | Count |
|---:|---:|
| 1 | 5 |
| 2 | 4 |
| 3 | 3 |
| 4 | 14 |
| 5 | 24 |

Mistake counts:

| Mistake | Count |
|---|---:|
| false | 37 |
| true | 13 |

Gate issues:

- `episode_012632`: object grounding flagged `empty`
- `episode_017546`: object grounding flagged `beside`
- `episode_034676`: object grounding flagged `across`
- `episode_034676`: object grounding flagged `settle`
- `episode_036781`: object grounding flagged `finish`

These failures look like the object-grounding heuristic is over-counting relation/action words as missing object nouns. The labels should not be used for scale-curve training until Kevin inspects these episodes or the gate is tightened.

## Value-Curation Probe On The 50-Episode Snapshot

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.value report --snapshot snap_2026_05_11_de43f7bf0b --method embedding-distance --top-n 10 --high-value-percentile 0.9
```

Top outliers by embedding-distance score:

| Rank | Episode | Score | Task |
|---:|---|---:|---|
| 1 | `episode_048956` | 16.392514 | put knife in pot or pan |
| 2 | `episode_001860` | 14.810907 | Pick up the spatula and move it to the lower right corner of the table |
| 3 | `episode_014040` | 14.188038 | put the blue cube to the blue cube in the right |
| 4 | `episode_001146` | 13.229725 | put banana in pot or pan |
| 5 | `episode_013597` | 11.690706 | put lemon on plate |
| 6 | `episode_010365` | 11.552950 | Place the broccoli into the pot |
| 7 | `episode_006686` | 11.520190 | put banana in pot or pan |
| 8 | `episode_013182` | 9.883628 | put knife in pot or pan |
| 9 | `episode_052920` | 9.571606 | put cup from counter or drying rack into sink |
| 10 | `episode_015256` | 9.452965 | put cup into pot or pan |

Tiered compression comparison:

- high-value percentile: `0.900`
- high-value episodes: 5
- source parquet bytes: 215,247
- uniform zstd bytes: 156,548
- tiered bytes: 193,081
- tiered vs uniform savings: `-23.34%`

At this 50-episode scale, tiered split-file overhead still dominates, so the value-aware layout is larger than uniform zstd. That is an honest negative result for tiny snapshots, not evidence against the idea at corpus scale.

## Scale-Curve Plan

The scale planner was run in planning mode only:

```powershell
.\.venv\Scripts\python.exe -m bridgeengine.benchmark.scale_curve --snapshot snap_2026_05_11_de43f7bf0b --sizes 50 200 800 --heldout-count 10 --quality-stratified --output-dir scale_results\plan_50_probe
```

Available:

- 50 episodes: 40 train / 10 held-out, quality-stratified training pool

Unavailable from the mounted snapshot:

- 200 episodes: only 50 labeled episodes present
- 800 episodes: only 50 labeled episodes present

No scale-curve LeWM training was launched.
