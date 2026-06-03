from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bridgeengine.ingest.snapshot import LABELER_VERSIONS

from .base import LabelResult, episode_id_from_path, sha256_file, write_json


class SubgoalImageLabeler:
    """Extract end-of-segment frames as pi0.7-style subgoal image targets."""

    name = "subgoal_images"
    version = LABELER_VERSIONS[name]

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def label_episode(self, episode_path: Path, snapshot_id: str) -> list[LabelResult]:
        t0 = time.perf_counter()
        episode_id = episode_id_from_path(episode_path)
        segment_path = self.output_root / "labels" / "subtask_segments" / snapshot_id / f"{episode_id}.json"
        if not segment_path.exists():
            raise FileNotFoundError(f"Subtask segment payload missing: {segment_path}")
        segments = json.loads(segment_path.read_text(encoding="utf-8"))["segments"]
        frames_path = episode_path / "frames.npy"
        if not frames_path.exists():
            raise FileNotFoundError(f"frames.npy missing for subgoal extraction: {episode_path}")
        frames = np.load(frames_path, mmap_mode="r")
        out_dir = self.output_root / "snapshots" / snapshot_id / "subgoals" / episode_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[LabelResult] = []
        for segment in segments:
            segment_idx = int(segment["segment_idx"])
            frame_idx = int(min(max(segment["end_step"], 0), frames.shape[0] - 1))
            image_path = out_dir / f"{segment_idx:02d}.jpg"
            Image.fromarray(np.asarray(frames[frame_idx], dtype=np.uint8)).save(image_path, quality=92)
            payload_path = out_dir / f"{segment_idx:02d}.json"
            payload = {
                "episode_id": episode_id,
                "segment_idx": segment_idx,
                "frame_idx": frame_idx,
                "subtask_text": segment["subtask_text"],
                "subgoal_image_path": str(image_path.resolve()),
                "source": "actual_end_of_segment_frame",
            }
            write_json(payload_path, payload)
            provenance = {
                "input_sha256": sha256_file(frames_path),
                "segment_payload_sha256": sha256_file(segment_path),
                "labeler_version": self.version,
                "source": "actual_end_of_segment_frame",
                "wall_clock_seconds": 0.0,
                "frame_idx": frame_idx,
            }
            rows.append(
                LabelResult(
                    self.name,
                    self.version,
                    episode_id,
                    payload_path,
                    1.0,
                    provenance,
                    segment_idx=segment_idx,
                    subgoal_image_path=image_path,
                )
            )
        total = time.perf_counter() - t0
        if rows:
            per_row = total / len(rows)
            rows = [
                LabelResult(
                    r.labeler_name,
                    r.labeler_version,
                    r.episode_id,
                    r.payload_path,
                    r.confidence,
                    {**r.provenance, "wall_clock_seconds": per_row},
                    segment_idx=r.segment_idx,
                    metadata_payload_json=r.metadata_payload_json,
                    subgoal_image_path=r.subgoal_image_path,
                )
                for r in rows
            ]
        return rows
