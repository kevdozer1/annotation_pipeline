from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root

from .schema import EPISODE_COLUMNS, LABEL_COLUMNS, SENSOR_COLUMNS, STEP_COLUMNS
from .snapshot import derive_snapshot_id, write_manifest

LOCAL_BRIDGEDATA_CANDIDATES = [
    Path("D:/bridgedata_v2_subset"),
    Path("C:/Users/Kevin/projects/LeWM_testbed/datasets/bridgedata_v2_subset"),
]
LEWM_PILOT_PATH = Path("C:/Users/Kevin/projects/LeWM_testbed/outputs/pilot_subset.json")


@dataclass(frozen=True)
class EpisodeSource:
    episode_id: str
    episode_path: Path
    task: str
    num_steps: int
    action_dim: int
    state_dim: int


def ingest_bridge_v2(
    source: str | Path = "bridge_v2",
    episodes: int = 13,
    data_root: str | Path | None = None,
    copy_raw: bool = False,
) -> dict[str, Any]:
    """Ingest BridgeData V2-style episodes into a deterministic POC snapshot.

    The preferred path is Kevin's local LEWM subset at ``D:/bridgedata_v2_subset``.
    When that is unavailable, a small synthetic Bridge-like set is generated in
    ``bridgeengine_data/raw/bridge_v2`` so the repository remains runnable from
    a clean clone.
    """
    root = resolve_data_root(data_root)
    source_root = _resolve_source_root(source, root, episodes)
    episode_sources = _select_episode_sources(source_root, episodes)
    if copy_raw:
        episode_sources = _copy_raw_episodes(episode_sources, root)

    source_records = [
        {
            "episode_id": ep.episode_id,
            "episode_path": str(ep.episode_path.resolve()),
            "task": ep.task,
            "num_steps": ep.num_steps,
        }
        for ep in episode_sources
    ]
    snapshot_id, transform_hash = derive_snapshot_id(source_records)
    snapshot_path = root / "snapshots" / snapshot_id
    snapshot_path.mkdir(parents=True, exist_ok=True)

    episodes_df, steps_df, sensors_df = _build_tables(episode_sources, snapshot_id)
    labels_df = pd.DataFrame(columns=LABEL_COLUMNS)

    episodes_df.to_parquet(snapshot_path / "episodes.parquet", index=False)
    steps_df.to_parquet(snapshot_path / "steps.parquet", index=False)
    sensors_df.to_parquet(snapshot_path / "sensors.parquet", index=False)
    labels_df.to_parquet(snapshot_path / "labels.parquet", index=False)
    manifest = write_manifest(
        snapshot_path=snapshot_path,
        snapshot_id=snapshot_id,
        source_episode_count=len(episode_sources),
        transform_hash=transform_hash,
    )

    return {
        "snapshot_id": snapshot_id,
        "snapshot_path": str(snapshot_path),
        "episode_count": len(episode_sources),
        "manifest": manifest,
    }


def _resolve_source_root(source: str | Path, root: Path, episodes: int) -> Path:
    if str(source).lower() in {"bridge_v2", "auto"}:
        for candidate in LOCAL_BRIDGEDATA_CANDIDATES:
            if (candidate / "episodes").exists():
                return candidate
        return _ensure_synthetic_source(root, episodes)
    if str(source).lower() == "synthetic":
        return _ensure_synthetic_source(root, episodes)
    source_root = Path(source).expanduser()
    if not source_root.exists():
        raise FileNotFoundError(f"BridgeData source does not exist: {source_root}")
    return source_root


def _select_episode_sources(source_root: Path, episodes: int) -> list[EpisodeSource]:
    manifest_entries = _load_manifest_entries(source_root)
    by_id = {f"episode_{int(e['episode_index']):06d}": e for e in manifest_entries}

    preferred_ids: list[str] = []
    if LEWM_PILOT_PATH.exists():
        pilot = json.loads(LEWM_PILOT_PATH.read_text(encoding="utf-8"))
        preferred_ids.extend(e["episode_id"] for e in pilot.get("core", []))
    preferred_ids.extend(by_id.keys())

    selected: list[EpisodeSource] = []
    seen: set[str] = set()
    for episode_id in preferred_ids:
        if episode_id in seen:
            continue
        ep_dir = source_root / "episodes" / episode_id
        if not ep_dir.exists():
            continue
        entry = by_id.get(episode_id, {})
        selected.append(_episode_from_dir(ep_dir, entry))
        seen.add(episode_id)
        if len(selected) >= episodes:
            break

    if len(selected) < episodes:
        raise ValueError(
            f"Requested {episodes} episodes but found {len(selected)} under {source_root}"
        )
    return selected


def _load_manifest_entries(source_root: Path) -> list[dict[str, Any]]:
    manifest_path = source_root / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    entries = []
    for ep_dir in sorted((source_root / "episodes").glob("episode_*")):
        idx = int(ep_dir.name.split("_")[1])
        meta = _read_episode_meta(ep_dir)
        frames = _load_optional_array(ep_dir / "frames.npy")
        actions = _load_optional_array(ep_dir / "actions.npy")
        entries.append(
            {
                "episode_index": idx,
                "task": meta.get("task", f"BridgeData episode {idx}"),
                "n_frames": int(len(frames) if frames is not None else len(actions)),
                "action_dim": int(actions.shape[1]) if actions is not None and actions.ndim == 2 else 0,
                "state_dim": 0,
            }
        )
    return entries


def _episode_from_dir(ep_dir: Path, manifest_entry: dict[str, Any]) -> EpisodeSource:
    meta = _read_episode_meta(ep_dir)
    frames = _load_optional_array(ep_dir / "frames.npy")
    actions = _load_optional_array(ep_dir / "actions.npy")
    states = _load_optional_array(ep_dir / "states.npy")
    num_steps = int(
        manifest_entry.get(
            "n_frames",
            len(frames) if frames is not None else len(actions) if actions is not None else 0,
        )
    )
    action_dim = int(manifest_entry.get("action_dim", actions.shape[1] if actions is not None and actions.ndim == 2 else 0))
    state_dim = int(manifest_entry.get("state_dim", states.shape[1] if states is not None and states.ndim == 2 else 0))
    return EpisodeSource(
        episode_id=ep_dir.name,
        episode_path=ep_dir,
        task=meta.get("task") or manifest_entry.get("task") or ep_dir.name,
        num_steps=num_steps,
        action_dim=action_dim,
        state_dim=state_dim,
    )


def _build_tables(
    episode_sources: list[EpisodeSource], snapshot_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    sensor_rows: list[dict[str, Any]] = []

    for ep in episode_sources:
        actions = _load_or_zeros(ep.episode_path / "actions.npy", (ep.num_steps, max(ep.action_dim, 1)))
        states = _load_or_zeros(ep.episode_path / "states.npy", (ep.num_steps, max(ep.state_dim, 1)))
        if states.shape[0] != ep.num_steps:
            states = _load_or_zeros(ep.episode_path / "state.npy", (ep.num_steps, max(ep.state_dim, 1)))
        frames = _load_optional_array(ep.episode_path / "frames.npy")
        num_steps = min(ep.num_steps, len(actions), len(states))

        episode_rows.append(
            {
                "episode_id": ep.episode_id,
                "source_path_video": str((ep.episode_path / "video.mp4").resolve()),
                "source_path_actions": str((ep.episode_path / "actions.npy").resolve()),
                "source_path_meta": str((ep.episode_path / "metadata.json").resolve()),
                "source_path_frames": str((ep.episode_path / "frames.npy").resolve()),
                "num_steps": int(num_steps),
                "language_instruction": ep.task,
                "snapshot_id": snapshot_id,
            }
        )

        for t in range(num_steps):
            step_rows.append(
                {
                    "episode_id": ep.episode_id,
                    "step_idx": int(t),
                    "timestamp": float(t / 5.0),
                    "action": [float(x) for x in np.asarray(actions[t]).ravel()],
                    "state": [float(x) for x in np.asarray(states[t]).ravel()],
                    "snapshot_id": snapshot_id,
                }
            )

        calibration = {"camera": "over_the_shoulder_rgb"}
        if frames is not None and frames.ndim == 4:
            calibration.update({"height": int(frames.shape[1]), "width": int(frames.shape[2]), "channels": int(frames.shape[3])})
        sensor_rows.append(
            {
                "episode_id": ep.episode_id,
                "sensor_name": "over_the_shoulder_rgb",
                "calibration_json": json.dumps(calibration, sort_keys=True),
                "snapshot_id": snapshot_id,
            }
        )

    return (
        pd.DataFrame(episode_rows, columns=EPISODE_COLUMNS),
        pd.DataFrame(step_rows, columns=STEP_COLUMNS),
        pd.DataFrame(sensor_rows, columns=SENSOR_COLUMNS),
    )


def _copy_raw_episodes(episode_sources: list[EpisodeSource], root: Path) -> list[EpisodeSource]:
    copied: list[EpisodeSource] = []
    raw_root = root / "raw" / "bridge_v2"
    raw_root.mkdir(parents=True, exist_ok=True)
    for ep in episode_sources:
        dst = raw_root / ep.episode_id
        if not dst.exists():
            shutil.copytree(ep.episode_path, dst)
        copied.append(
            EpisodeSource(
                episode_id=ep.episode_id,
                episode_path=dst,
                task=ep.task,
                num_steps=ep.num_steps,
                action_dim=ep.action_dim,
                state_dim=ep.state_dim,
            )
        )
    return copied


def _ensure_synthetic_source(root: Path, episodes: int) -> Path:
    source_root = root / "raw" / "bridge_v2"
    episodes_root = source_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for i in range(episodes):
        ep_id = f"episode_{i:06d}"
        ep_dir = episodes_root / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1000 + i)
        n_frames = 18 + (i % 9)
        frames = np.zeros((n_frames, 96, 96, 3), dtype=np.uint8)
        for t in range(n_frames):
            frames[t, :, :, 0] = (30 + i * 11 + t * 3) % 255
            frames[t, :, :, 1] = np.linspace(20, 220, 96, dtype=np.uint8)[None, :]
            frames[t, :, :, 2] = np.linspace(220, 20, 96, dtype=np.uint8)[:, None]
        actions = rng.normal(0.0, 0.25, size=(n_frames, 7)).astype(np.float32)
        actions[:, -1] = (np.arange(n_frames) > n_frames // 2).astype(np.float32)
        states = rng.normal(0.0, 1.0, size=(n_frames, 7)).astype(np.float32)
        task = f"put synthetic object {i % 4} into target bin"
        np.save(ep_dir / "frames.npy", frames)
        np.save(ep_dir / "actions.npy", actions)
        np.save(ep_dir / "states.npy", states)
        (ep_dir / "video.mp4").write_bytes(b"bridgeengine synthetic placeholder video\n")
        (ep_dir / "metadata.json").write_text(
            json.dumps({"episode_id": ep_id, "task": task}, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "episode_index": i,
                "task": task,
                "n_frames": n_frames,
                "action_dim": 7,
                "state_dim": 7,
                "has_video": True,
                "frame_shape": list(frames.shape),
            }
        )
    (source_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return source_root


def _read_episode_meta(ep_dir: Path) -> dict[str, Any]:
    for name in ("metadata.json", "meta.json"):
        path = ep_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_optional_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def _load_or_zeros(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if path.exists():
        return np.asarray(np.load(path, allow_pickle=False))
    return np.zeros(shape, dtype=np.float32)

