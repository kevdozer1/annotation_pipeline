from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ID = "Qu3tzal/bridgev2"
DEFAULT_OUTPUT_ROOT = Path("D:/bridgedata_v2_subset")
INCLUDE_KEYWORDS = ("push", "slide", "pick", "place", "put", "move", "lift", "stack")
EXCLUDE_KEYWORDS = ("fold", "cloth", "towel", "wipe", "sweep", "pour", "drawer", "door", "faucet")
RIGID_OBJECT_KEYWORDS = (
    "block",
    "cube",
    "can",
    "bottle",
    "cup",
    "bowl",
    "spoon",
    "pan",
    "pot",
    "plate",
    "box",
    "container",
    "tray",
    "sponge",
    "carrot",
    "banana",
    "pepper",
    "mug",
    "jar",
    "knife",
    "fork",
    "apple",
    "lemon",
    "tomato",
)


def inventory(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output_root)
    episodes_dir = root / "episodes"
    episode_dirs = sorted(p for p in episodes_dir.glob("episode_*") if p.is_dir()) if episodes_dir.exists() else []
    mp4_count = len(list(episodes_dir.glob("episode_*/video.mp4"))) if episodes_dir.exists() else 0
    manifests = {path.name: _manifest_len(path) for path in root.glob("manifest*.json")}
    return {
        "output_root": str(root),
        "episode_dir_count": len(episode_dirs),
        "mp4_count": mp4_count,
        "manifests": manifests,
        "first_episode_dirs": [p.name for p in episode_dirs[:5]],
        "last_episode_dirs": [p.name for p in episode_dirs[-5:]],
    }


def write_estimate(output_path: str | Path = "SCALEOUT_1000.md", output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    inv = inventory(output_root)
    current = int(inv["episode_dir_count"])
    target = 1000
    incremental = max(0, target - current)
    gemini_cost_per_episode = 0.011886
    gemini_seconds_per_episode = 16.6944
    full_grid_100_hours = 3.896
    # Conservative scaling anchor: 1000 has roughly 10x frames of the 100-episode run.
    full_grid_1000_hours = full_grid_100_hours * 10.0
    text = f"""# BridgeEngine 1000-Episode Scale-Out Plan

Last updated: 2026-06-07

## Local Data Reality

The SSD currently exposes:

```text
episode folders: {inv['episode_dir_count']}
mp4 files: {inv['mp4_count']}
manifests: {inv['manifests']}
```

I do not see a hidden full BridgeData V2 mirror on `D:`. The local source is a curated 100-episode subset, not the full approximately 60k-episode corpus.

## Target

Scale from 100 local episodes to 1000 local episodes by downloading about `{incremental}` more BridgeData V2 episodes from Hugging Face dataset `{REPO_ID}`.

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
cost per episode: ${gemini_cost_per_episode:.6f}
serial wall-clock per episode: {gemini_seconds_per_episode:.2f}s
```

Projection:

| scope | episodes | cost | serial labeling time |
|---|---:|---:|---:|
| incremental new set | {incremental} | ${incremental * gemini_cost_per_episode:.2f} | {incremental * gemini_seconds_per_episode / 3600.0:.2f} hours |
| full 1000 set | 1000 | ${1000 * gemini_cost_per_episode:.2f} | {1000 * gemini_seconds_per_episode / 3600.0:.2f} hours |

## Depth And Track Extraction Estimate

The existing 100 episodes already have Video-Depth-Anything depth and CoTracker3 track files. The repo does not contain a reliable measured per-episode extraction timing summary for those models, so the honest next step is a 10-episode extraction probe before launching all 900 new episodes.

Conservative handoff command:

```powershell
.\\scripts\\scaleout_1000_extract_depth_tracks.ps1 -Manifest D:\\bridgedata_v2_subset\\manifest_1000.json -Device cuda -GridSize 20
```

Do not run this across 900 new episodes until the downloader and a 10-episode extractor probe confirm wall-clock.

## Training-Time Estimate

The preregistered 100-episode head-to-head estimate is:

```text
total from scratch at 100 episodes: {full_grid_100_hours:.3f} hours
```

A simple frame-count linear projection gives:

```text
full-grid 1000-episode training estimate: {full_grid_1000_hours:.1f} hours
```

This is an estimate, not a promise. Larger batches, fewer scales, fewer conditions, or running only N=1000 would change it.

## Gated Commands

Download more episodes:

```powershell
.\\scripts\\scaleout_1000_download.ps1 -TargetEpisodes 1000 -ScanSample 5000 -OutputRoot D:\\bridgedata_v2_subset
```

Label with Gemini:

```powershell
.\\scripts\\scaleout_1000_label.ps1 -SourceRoot D:\\bridgedata_v2_subset -Episodes 1000 -Backend gemini -Model gemini-2.5-flash
```

Extract depth and tracks:

```powershell
.\\scripts\\scaleout_1000_extract_depth_tracks.ps1 -Manifest D:\\bridgedata_v2_subset\\manifest_1000.json -Device cuda -GridSize 20
```

## Stop Rule

Do not download, label, extract, or train at 1000 scale until Kevin explicitly approves the disk, API spend, and GPU time.
"""
    output = Path(output_path)
    output.write_text(text, encoding="utf-8")
    return output


def download(target_episodes: int, scan_sample: int, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Download a selected BridgeData scale-out set.

    This is intentionally gated behind the explicit `download` command because
    it can use network, disk, and time.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    episodes_dir = root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = {int(path.name.split("_")[1]) for path in episodes_dir.glob("episode_*") if path.is_dir()}
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    parquet_files = sorted(path for path in all_files if path.endswith(".parquet"))
    random.seed(20260607)
    sample = random.sample(parquet_files, min(scan_sample, len(parquet_files)))
    candidates = _candidate_episode_rows(sample, existing_ids, target_episodes, hf_hub_download)
    selected = candidates[: max(0, target_episodes - len(existing_ids))]
    all_selected_ids = sorted(existing_ids | {row["episode_index"] for row in selected})
    downloaded = []
    for row in selected:
        ep_idx = int(row["episode_index"])
        ep_dir = episodes_dir / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(exist_ok=True)
        chunk_idx = ep_idx // 1000
        parquet_file = f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        video_file = f"videos/chunk-{chunk_idx:03d}/observation.images.image_0/episode_{ep_idx:06d}.mp4"
        parquet_local = hf_hub_download(REPO_ID, parquet_file, repo_type="dataset")
        df = pd.read_parquet(parquet_local).sort_values("frame_index").reset_index(drop=True)
        actions = np.array(df["action"].tolist()) if "action" in df.columns else np.zeros((len(df), 7), dtype=np.float32)
        states = (
            np.array(df["observation.state"].tolist())
            if "observation.state" in df.columns
            else np.zeros((len(df), 7), dtype=np.float32)
        )
        np.save(ep_dir / "actions.npy", actions)
        np.save(ep_dir / "states.npy", states)
        video_local = hf_hub_download(REPO_ID, video_file, repo_type="dataset")
        shutil.copy2(video_local, ep_dir / "video.mp4")
        frames = _extract_frames_cv2(ep_dir / "video.mp4")
        np.save(ep_dir / "frames.npy", frames)
        metadata = {
            "episode_index": ep_idx,
            "task": row["task"],
            "n_frames": int(len(df)),
            "action_dim": int(actions.shape[-1]),
            "state_dim": int(states.shape[-1]),
            "has_video": True,
            "frame_shape": list(frames.shape),
        }
        (ep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        downloaded.append(metadata)
    manifest_rows = []
    for ep_idx in all_selected_ids:
        meta_path = episodes_dir / f"episode_{ep_idx:06d}" / "metadata.json"
        if meta_path.exists():
            manifest_rows.append(json.loads(meta_path.read_text(encoding="utf-8")))
    manifest_path = root / f"manifest_{len(manifest_rows)}.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    if len(manifest_rows) >= 1000:
        (root / "manifest_1000.json").write_text(json.dumps(manifest_rows[:1000], indent=2), encoding="utf-8")
    return {
        "target_episodes": int(target_episodes),
        "existing_episode_count": len(existing_ids),
        "downloaded_episode_count": len(downloaded),
        "manifest_path": str(manifest_path),
        "final_manifest_episode_count": len(manifest_rows),
    }


def _candidate_episode_rows(sample_files: list[str], existing_ids: set[int], target: int, hf_hub_download) -> list[dict[str, Any]]:
    rows = []
    for parquet_file in sample_files:
        if len(rows) >= max(0, target - len(existing_ids)) * 3:
            break
        try:
            local = hf_hub_download(REPO_ID, parquet_file, repo_type="dataset")
            df = pd.read_parquet(local)
            ep_idx = int(df["episode_index"].iloc[0])
            if ep_idx in existing_ids:
                continue
            task = str(df["language_instruction"].iloc[0]) if "language_instruction" in df.columns else ""
            task_lower = task.lower()
            if not any(word in task_lower for word in INCLUDE_KEYWORDS):
                continue
            if any(word in task_lower for word in EXCLUDE_KEYWORDS):
                continue
            score = sum(1 for word in RIGID_OBJECT_KEYWORDS if word in task_lower)
            rows.append({"episode_index": ep_idx, "task": task, "n_frames": int(len(df)), "score": score})
        except Exception:
            continue
    rows.sort(key=lambda row: (-int(row["score"]), abs(int(row["n_frames"]) - 32), int(row["episode_index"])))
    return rows


def _extract_frames_cv2(video_path: Path) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    ok, frame = capture.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = capture.read()
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode frames from {video_path}")
    return np.stack(frames, axis=0)


def _manifest_len(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return -1
    return len(data) if isinstance(data, list) else len(data.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="BridgeEngine 1000-episode scale-out gated utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    est = sub.add_parser("estimate")
    est.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    est.add_argument("--output", default="SCALEOUT_1000.md")
    dl = sub.add_parser("download")
    dl.add_argument("--target-episodes", type=int, default=1000)
    dl.add_argument("--scan-sample", type=int, default=5000)
    dl.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    if args.cmd == "inventory":
        print(json.dumps(inventory(args.output_root), indent=2, sort_keys=True))
    elif args.cmd == "estimate":
        path = write_estimate(args.output, args.output_root)
        print(path)
    elif args.cmd == "download":
        print(json.dumps(download(args.target_episodes, args.scan_sample, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
