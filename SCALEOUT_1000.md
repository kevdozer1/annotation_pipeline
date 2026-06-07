# BridgeEngine 1000-Episode Scale-Out Plan

Last updated: 2026-06-07

## Local Data Reality

The SSD currently exposes:

```text
episode folders: 100
mp4 files: 100
manifests: {'manifest.json': 20, 'manifest_100.json': 100, 'manifest_new80.json': 80}
```

I do not see a hidden full BridgeData V2 mirror on `D:`. The local source is a curated 100-episode subset, not the full approximately 60k-episode corpus.

## Target

Scale from 100 local episodes to 1000 local episodes by downloading about `900` more BridgeData V2 episodes from Hugging Face dataset `Qu3tzal/bridgev2`.

The first 100 are human-gold calibrated. The added episodes would be Gemini-calibrated, anchored by the measured 100-episode reliability numbers:

```text
quality exact agreement: 0.42
quality within-one agreement: 0.77
subtask-boundary temporal IoU mean: 0.683
derived subgoal frame agreement: 0.347
```

That means the 1000 run is not human-gold. It is a scale probe using the calibrated rubric and VLM prompt.

## Gemini Labeling Estimate

Measured on `snap_2026_05_11_1dde3edf5d` with Gemini 2.5 Flash:

```text
cost per episode: $0.011886
serial wall-clock per episode: 16.69s
```

Projection:

| scope | episodes | cost | serial labeling time |
|---|---:|---:|---:|
| incremental new set | 900 | $10.70 | 4.17 hours |
| full 1000 set | 1000 | $11.89 | 4.64 hours |

## Depth And Track Extraction Estimate

The existing 100 episodes already have Video-Depth-Anything depth and CoTracker3 track files. The repo does not contain a reliable measured per-episode extraction timing summary for those models, so the honest next step is a 10-episode extraction probe before launching all 900 new episodes.

Conservative handoff command:

```powershell
.\scripts\scaleout_1000_extract_depth_tracks.ps1 -Manifest D:\bridgedata_v2_subset\manifest_1000.json -Device cuda -GridSize 20
```

Do not run this across 900 new episodes until the downloader and a 10-episode extractor probe confirm wall-clock.

## Training-Time Estimate

The preregistered 100-episode head-to-head estimate is:

```text
total from scratch at 100 episodes: 3.896 hours
```

A simple frame-count linear projection gives:

```text
full-grid 1000-episode training estimate: 39.0 hours
```

This is an estimate, not a promise. Larger batches, fewer scales, fewer conditions, or running only N=1000 would change it.

## Gated Commands

Download more episodes:

```powershell
.\scripts\scaleout_1000_download.ps1 -TargetEpisodes 1000 -ScanSample 5000 -OutputRoot D:\bridgedata_v2_subset
```

Label with Gemini:

```powershell
.\scripts\scaleout_1000_label.ps1 -SourceRoot D:\bridgedata_v2_subset -Episodes 1000 -Backend gemini -Model gemini-2.5-flash
```

Extract depth and tracks:

```powershell
.\scripts\scaleout_1000_extract_depth_tracks.ps1 -Manifest D:\bridgedata_v2_subset\manifest_1000.json -Device cuda -GridSize 20
```

## Stop Rule

Do not download, label, extract, or train at 1000 scale until Kevin explicitly approves the disk, API spend, and GPU time.
