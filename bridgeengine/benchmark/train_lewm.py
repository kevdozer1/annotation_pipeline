from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FAMILIES = ("baseline", "textual", "perceptive", "hybrid")

BASE_LATENT_MSE = {
    "baseline": 0.164,
    "textual": 0.154,
    "perceptive": 0.139,
    "hybrid": 0.132,
}

FAMILY_LABELERS = {
    "baseline": (),
    "textual": ("captions",),
    "perceptive": ("masks", "depth", "tracks"),
    "hybrid": ("captions", "masks", "depth", "tracks"),
}


def run_family_seed(cut_path: Path, family: str, seed: int, scale: int = 13) -> dict[str, Any]:
    """Run the Mode A benchmark cell.

    The POC exposes the same grid shape and result schema as the LEWM run, but
    uses a deterministic CPU proxy unless Kevin swaps this function for the
    heavyweight ``lewm_finetune.train.train`` call. This keeps quickstart under
    10 minutes while preserving the experiment contract.
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
        "scale": scale,
        "seed": seed,
        "latent_mse": latent_mse,
        "idm_accuracy": None,
        "total_label_cost_seconds": round(_estimate_label_cost(cut_path, family, n_episodes), 6),
        "run_mode": "deterministic_cpu_proxy",
        "episode_count": n_episodes,
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
        return sum(float(runtimes.get(name, 0.0)) for name in labelers) / max(n_episodes, 1)
    return 0.0


def _stable_noise(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / 2**32
    return (value - 0.5) * 0.006


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

