from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from bridgeengine.ingest.snapshot import LABELER_VERSIONS

from .base import LabelResult, episode_id_from_path, sha256_file


class TrackLabeler:
    name = "tracks"
    version = LABELER_VERSIONS["tracks"]

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def label_episode(self, episode_path: Path, snapshot_id: str) -> LabelResult:
        t0 = time.perf_counter()
        episode_id = episode_id_from_path(episode_path)
        tracks_source = episode_path / "tracks.npy"
        vis_source = episode_path / "visibility.npy"
        payload_path = self.output_root / "labels" / "tracks" / snapshot_id / episode_id / "tracks.npz"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        if tracks_source.exists() and vis_source.exists():
            tracks = np.load(tracks_source, allow_pickle=False)
            visibility = np.load(vis_source, allow_pickle=False).astype(bool)
            adapter = "lewm_cotracker3_artifact"
        else:
            tracks, visibility = _synthetic_tracks(episode_path)
            adapter = "synthetic_grid_tracks"
        np.savez_compressed(payload_path, tracks=tracks.astype(np.float32), visibility=visibility)
        confidence = float(np.asarray(visibility).mean())
        dt = time.perf_counter() - t0
        provenance = {
            "input_sha256": sha256_file(episode_path / "frames.npy"),
            "source_tracks_sha256": sha256_file(tracks_source),
            "source_visibility_sha256": sha256_file(vis_source),
            "labeler_version": self.version,
            "labeler_config": {"model": "CoTracker3 offline", "adapter": adapter, "foreground_fraction": 0.6},
            "wall_clock_seconds": dt,
            "n_tracks": int(tracks.shape[1]),
            "avg_visibility": confidence,
            "track_shape": [int(v) for v in tracks.shape],
        }
        return LabelResult(self.name, self.version, episode_id, payload_path, confidence, provenance)


def _synthetic_tracks(episode_path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames_path = episode_path / "frames.npy"
    if frames_path.exists():
        frames = np.load(frames_path, mmap_mode="r")
        t, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    else:
        t, h, w = 8, 96, 96
    nx, ny = 20, 20
    xs = np.linspace(5, w - 6, nx, dtype=np.float32)
    ys = np.linspace(5, h - 6, ny, dtype=np.float32)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    tracks = np.zeros((t, len(grid), 2), dtype=np.float32)
    for i in range(t):
        offset = np.array([0.5 * i, 0.2 * i], dtype=np.float32)
        tracks[i] = grid + offset
    visibility = np.ones((t, len(grid)), dtype=bool)
    return tracks, visibility

