from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from bridgeengine.ingest.snapshot import LABELER_VERSIONS

from .base import LabelResult, episode_id_from_path, read_metadata, sha256_file, write_json


class CaptionLabeler:
    """Lightweight Moondream-compatible caption adapter.

    The POC keeps the labeler contract and provenance shape Moondream would use,
    but avoids downloading model weights during quickstart. If a real Moondream
    wrapper is added later, only this class needs to change.
    """

    name = "captions"
    version = LABELER_VERSIONS["captions"]

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def label_episode(self, episode_path: Path, snapshot_id: str) -> LabelResult:
        t0 = time.perf_counter()
        episode_id = episode_id_from_path(episode_path)
        meta = read_metadata(episode_path)
        task = meta.get("task") or meta.get("language_instruction") or episode_id
        frames_path = episode_path / "frames.npy"
        frame_stats = _frame_stats(frames_path)
        caption_text = _caption_from_task(task, frame_stats)
        payload = {
            "episode_id": episode_id,
            "caption": caption_text,
            "frame_captions": {
                "first": f"Robot workspace before manipulation: {task}.",
                "middle": f"Robot manipulates the relevant object for: {task}.",
                "last": f"Robot approaches the terminal state for: {task}.",
            },
            "confidence": 0.86,
            "adapter": "deterministic_poc_captioner",
            "intended_model": "vikhyatk/moondream2",
        }
        payload_path = self.output_root / "labels" / "captions" / snapshot_id / f"{episode_id}.json"
        write_json(payload_path, payload)
        dt = time.perf_counter() - t0
        provenance = {
            "input_sha256": sha256_file(frames_path),
            "labeler_version": self.version,
            "labeler_config": {"frames": ["first", "middle", "last"], "mode": "poc_adapter"},
            "wall_clock_seconds": dt,
            "caption_text": caption_text,
            "frame_stats": frame_stats,
        }
        return LabelResult(self.name, self.version, episode_id, payload_path, 0.86, provenance)


def _frame_stats(frames_path: Path) -> dict:
    if not frames_path.exists():
        return {"available": False}
    frames = np.load(frames_path, mmap_mode="r")
    sample = frames[[0, len(frames) // 2, len(frames) - 1]]
    mean_rgb = sample.reshape(-1, 3).mean(axis=0)
    return {
        "available": True,
        "n_frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "mean_rgb": [round(float(v), 3) for v in mean_rgb],
    }


def _caption_from_task(task: str, frame_stats: dict) -> str:
    stats = ""
    if frame_stats.get("available"):
        stats = f" The clip has {frame_stats['n_frames']} RGB frames."
    return f"A robot manipulation episode where the instruction is to {task}.{stats}"

