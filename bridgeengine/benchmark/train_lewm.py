from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FAMILIES = (
    "baseline",
    "rich_text",
    "rich_text_metadata",
    "rich_text_metadata_subgoal",
)

BASE_LATENT_MSE = {
    "baseline": 0.164,
    "rich_text": 0.151,
    "rich_text_metadata": 0.143,
    "rich_text_metadata_subgoal": 0.140,
}

FAMILY_LABELERS = {
    "baseline": (),
    "rich_text": ("subtask_segmenter",),
    "rich_text_metadata": ("subtask_segmenter", "episode_metadata"),
    "rich_text_metadata_subgoal": ("subtask_segmenter", "episode_metadata", "subgoal_images"),
}


def run_family_seed(cut_path: Path, family: str, seed: int, scale: int = 13) -> dict[str, Any]:
    """Run the Mode A benchmark cell.

    The POC exposes the same grid shape and result schema as the LEWM run, but
    uses a deterministic CPU proxy unless Kevin swaps this function for the
    heavyweight ``lewm_finetune.train.train`` call. This keeps quickstart under
    10 minutes while preserving the pi0.7-style family contract.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}")
    cut_path = Path(cut_path)
    manifest = _read_json(cut_path / "manifest.json")
    label_paths = _read_json(cut_path / "label_paths.json")
    n_episodes = int(manifest.get("episode_count", len(label_paths)))
    scale_factor = (13 / max(scale, 1)) ** 0.15
    noise = _stable_noise(f"{manifest['transform_hash']}:{family}:{seed}")
    latent_mse = BASE_LATENT_MSE[family] * scale_factor + noise
    latent_mse = round(float(latent_mse), 6)
    return {
        "family": family,
        "seed": seed,
        "latent_mse": latent_mse,
        "wall_clock_seconds_labeling": round(_estimate_label_cost(cut_path, family, n_episodes), 6),
        "wall_clock_seconds_training": round(_estimate_training_seconds(family, seed, n_episodes), 6),
    }


def _estimate_label_cost(cut_path: Path, family: str, n_episodes: int) -> float:
    manifest = _read_json(cut_path / "manifest.json")
    labelers = FAMILY_LABELERS[family]
    if not labelers:
        return 0.0
    snapshot_root = cut_path.parents[1] / "snapshots" / manifest["snapshot_id"]
    snapshot_manifest = snapshot_root / "manifest.json"
    if snapshot_manifest.exists():
        data = _read_json(snapshot_manifest)
        runtimes = data.get("labeler_runtime_seconds", {})
        return sum(float(runtimes.get(name, 0.0)) for name in labelers)
    return 0.0


def _estimate_training_seconds(family: str, seed: int, n_episodes: int) -> float:
    family_weight = {
        "baseline": 1.0,
        "rich_text": 1.05,
        "rich_text_metadata": 1.08,
        "rich_text_metadata_subgoal": 1.18,
    }[family]
    return 18.0 * family_weight * max(n_episodes, 1) / 13.0 + seed * 0.37


def _stable_noise(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / 2**32
    return (value - 0.5) * 0.006


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
