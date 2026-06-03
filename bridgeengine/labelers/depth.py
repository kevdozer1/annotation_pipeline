from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from bridgeengine.ingest.snapshot import LABELER_VERSIONS

from .base import LabelResult, episode_id_from_path, sha256_file


class DepthLabeler:
    name = "perceptive_depth"
    version = LABELER_VERSIONS["perceptive_depth"]

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def label_episode(self, episode_path: Path, snapshot_id: str) -> LabelResult:
        t0 = time.perf_counter()
        episode_id = episode_id_from_path(episode_path)
        source = episode_path / "depth.npy"
        payload_path = self.output_root / "labels" / "depth" / snapshot_id / episode_id / "depth.npy"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, payload_path)
            depths = np.load(payload_path, mmap_mode="r")
            adapter = "lewm_video_depth_anything_artifact"
        else:
            depths_arr = _synthetic_depth(episode_path)
            np.save(payload_path, depths_arr)
            depths = depths_arr
            adapter = "synthetic_gradient_depth"
        dt = time.perf_counter() - t0
        provenance = {
            "input_sha256": sha256_file(episode_path / "frames.npy"),
            "source_depth_sha256": sha256_file(source),
            "labeler_version": self.version,
            "labeler_config": {"model": "Video-Depth-Anything-Small", "adapter": adapter},
            "wall_clock_seconds": dt,
            "depth_min": float(np.asarray(depths).min()),
            "depth_max": float(np.asarray(depths).max()),
            "depth_shape": [int(v) for v in np.asarray(depths).shape],
        }
        return LabelResult(self.name, self.version, episode_id, payload_path, None, provenance)


def _synthetic_depth(episode_path: Path) -> np.ndarray:
    frames_path = episode_path / "frames.npy"
    if frames_path.exists():
        frames = np.load(frames_path, mmap_mode="r")
        t, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    else:
        t, h, w = 8, 96, 96
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    base = x + y
    return np.stack([(base + i / max(t, 1)).astype(np.float32) for i in range(t)], axis=0)
