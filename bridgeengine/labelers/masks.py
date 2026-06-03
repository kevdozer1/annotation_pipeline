from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from bridgeengine.ingest.snapshot import LABELER_VERSIONS

from .base import LabelResult, episode_id_from_path, sha256_file


class MaskLabeler:
    name = "perceptive_masks"
    version = LABELER_VERSIONS["perceptive_masks"]

    def __init__(self, output_root: Path):
        self.output_root = output_root

    def label_episode(self, episode_path: Path, snapshot_id: str) -> LabelResult:
        t0 = time.perf_counter()
        episode_id = episode_id_from_path(episode_path)
        source = episode_path / "object_mask.npy"
        payload_path = self.output_root / "labels" / "masks" / snapshot_id / episode_id / "masks.npz"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            tight = np.load(source, allow_pickle=False).astype(bool)
            adapter = "lewm_sam_artifact"
        else:
            tight = _synthetic_masks(episode_path)
            adapter = "synthetic_center_mask"
        dilate8 = _dilate_stack(tight, radius=8)
        dilate20 = _dilate_stack(tight, radius=20)
        np.savez_compressed(payload_path, tight=tight, dilate8=dilate8, dilate20=dilate20)
        coverage = float(tight.mean())
        confidence = float(np.clip(0.72 + (1.0 - abs(coverage - 0.12)) * 0.18, 0.0, 0.97))
        dt = time.perf_counter() - t0
        provenance = {
            "input_sha256": sha256_file(episode_path / "frames.npy"),
            "source_mask_sha256": sha256_file(source),
            "labeler_version": self.version,
            "labeler_config": {"model": "SAM3/SAM2-style LEWM port", "adapter": adapter, "dilations_px": [8, 20]},
            "wall_clock_seconds": dt,
            "mask_coverage": coverage,
            "mask_shape": [int(v) for v in tight.shape],
        }
        return LabelResult(self.name, self.version, episode_id, payload_path, confidence, provenance)


def _synthetic_masks(episode_path: Path) -> np.ndarray:
    frames_path = episode_path / "frames.npy"
    if frames_path.exists():
        frames = np.load(frames_path, mmap_mode="r")
        t, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    else:
        t, h, w = 8, 96, 96
    yy, xx = np.mgrid[:h, :w]
    masks = []
    for i in range(t):
        cx = w * (0.35 + 0.25 * i / max(t - 1, 1))
        cy = h * 0.52
        r = min(h, w) * 0.16
        masks.append(((xx - cx) ** 2 + (yy - cy) ** 2) <= r**2)
    return np.stack(masks, axis=0)


def _dilate_stack(mask_stack: np.ndarray, radius: int) -> np.ndarray:
    size = max(3, radius * 2 + 1)
    if size % 2 == 0:
        size += 1
    out = []
    for mask in mask_stack:
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        dilated = image.filter(ImageFilter.MaxFilter(size=size))
        out.append(np.asarray(dilated) > 0)
    return np.stack(out, axis=0)
