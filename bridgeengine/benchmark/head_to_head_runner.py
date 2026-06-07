from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import yaml


DEFAULT_LOCAL_ROOT = Path("D:/bridgedata_v2_subset")
DEFAULT_EPISODES_ROOT = DEFAULT_LOCAL_ROOT / "episodes"
DEFAULT_SOURCE_H5 = DEFAULT_LOCAL_ROOT / "datasets" / "bridgedata_v2_100ep.h5"
DEFAULT_MANIFEST = DEFAULT_LOCAL_ROOT / "manifest_100.json"

# Target spatial resolution for the LeWM world model (matches the original
# LeWM_testbed/scripts/export_bridgedata_100ep_h5.py export pipeline).
TARGET_SIZE = 224
DEFAULT_N_POINTS = 400
DEFAULT_PLAN_DIR = Path("head_to_head_results/preregistered_100")
DEFAULT_LEWM_ROOT = Path("C:/Users/Kevin/projects/LeWM_testbed")
DEFAULT_PRETRAINED = (
    "D:/hf_cache/models--quentinll--lewm-cube/snapshots/"
    "7d05e023b3c1114cc8e803ec23fb0177d688598b"
)
DEFAULT_SEEDS = (42, 137, 256)
DEFAULT_SIZES = (25, 50, 100)

CV_CONDITIONS: dict[str, dict[str, Any]] = {
    "A_baseline": {
        "condition_name": "A_baseline",
        "auxiliary_heads": {},
    },
    "B_depth": {
        "condition_name": "B_depth",
        "auxiliary_heads": {
            "depth": {"enabled": True, "weight": 0.1, "output_size": 56, "hidden_channels": 64}
        },
    },
    "D_tracks": {
        "condition_name": "D_tracks",
        "auxiliary_heads": {
            "tracks": {"enabled": True, "weight": 0.1, "n_points": 400, "hidden_dim": 256}
        },
    },
    "E_depth_tracks": {
        "condition_name": "E_depth_tracks",
        "auxiliary_heads": {
            "depth": {"enabled": True, "weight": 0.1, "output_size": 56, "hidden_channels": 64},
            "tracks": {"enabled": True, "weight": 0.1, "n_points": 400, "hidden_dim": 256},
        },
    },
}

PI07_FAMILIES = (
    "baseline",
    "rich_text",
    "rich_text_metadata",
    "rich_text_metadata_subgoal",
)


def verify_signal_files(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    plan_dir: str | Path = DEFAULT_PLAN_DIR,
) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    root = Path(local_root)
    required = ("frames.npy", "actions.npy", "states.npy", "depth.npy", "tracks.npy", "visibility.npy", "video.mp4")
    rows = []
    for episode in manifest:
        episode_id = _episode_id(episode)
        ep_dir = root / "episodes" / episode_id
        missing = [name for name in required if not (ep_dir / name).exists()]
        rows.append({"episode_id": episode_id, "missing": missing, "ok": not missing})
    splits = _split_membership(plan_dir)
    report = {
        "manifest_episode_count": len(manifest),
        "checked_required_files": list(required),
        "episode_ok_count": sum(1 for row in rows if row["ok"]),
        "episode_missing_count": sum(1 for row in rows if not row["ok"]),
        "missing": [row for row in rows if not row["ok"]],
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "all_splits_covered_by_manifest": all(
            episode_id in {row["episode_id"] for row in rows} for ids in splits.values() for episode_id in ids
        ),
    }
    return report


def prepare_handoff(
    plan_dir: str | Path = DEFAULT_PLAN_DIR,
    source_h5: str | Path = DEFAULT_SOURCE_H5,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = "head_to_head_results/run_100",
    lewm_root: str | Path = DEFAULT_LEWM_ROOT,
    pretrained_path: str = DEFAULT_PRETRAINED,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    max_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 5e-5,
    dry_run: bool = False,
    episodes_root: str | Path = DEFAULT_EPISODES_ROOT,
) -> dict[str, Any]:
    # Resolve every path to an absolute location so that downstream training and
    # evaluation work regardless of the process working directory.
    out = Path(output_dir).resolve()
    plan_dir = Path(plan_dir).resolve()
    lewm_root = Path(lewm_root).resolve()
    episodes_root = Path(episodes_root).resolve()
    datasets_dir = out / "datasets"
    configs_dir = out / "configs"
    logs_dir = out / "logs"
    runs_dir = out / "runs"
    for path in (datasets_dir, configs_dir, logs_dir, runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(manifest_path)
    manifest_ids = [_episode_id(row) for row in manifest]
    split_payloads = _load_split_payloads(plan_dir, sizes)
    # The consolidated source HDF5 is intentionally NOT read: it stores pixels and
    # depth with a Blosc2 plugin codec that does not resolve in every interpreter,
    # which is what produced the original "can't open directory" OSError. Instead we
    # assemble each split HDF5 directly from the per-episode .npy arrays into a
    # self-contained, gzip-compressed file with no external/plugin links.
    source_h5 = Path(source_h5)
    if not episodes_root.is_dir():
        raise FileNotFoundError(f"Per-episode arrays directory not found: {episodes_root}")

    h5_exports = []
    config_paths = []
    commands: list[dict[str, Any]] = []
    sanity_summary: list[dict[str, Any]] = []
    for size, split in split_payloads.items():
        train_name = f"be_h2h_scale_{size}_train"
        heldout_name = f"be_h2h_scale_{size}_heldout"
        train_h5 = datasets_dir / f"{train_name}.h5"
        heldout_h5 = datasets_dir / f"{heldout_name}.h5"
        if not dry_run:
            export_split_h5_from_npy(
                episodes_root, manifest_ids, split["train_episode_ids"], train_name, train_h5
            )
            export_split_h5_from_npy(
                episodes_root, manifest_ids, split["heldout_episode_ids"], heldout_name, heldout_h5
            )
            sanity_summary.append(_summarize_split_h5(size, split, train_h5, heldout_h5))
        h5_exports.extend([str(train_h5), str(heldout_h5)])

        for condition_name, condition in CV_CONDITIONS.items():
            cfg = _lewm_config(
                condition=condition,
                dataset_name=train_name,
                data_cache_dir=out,
                pretrained_path=pretrained_path,
                max_epochs=max_epochs,
                batch_size=batch_size,
                lr=lr,
            )
            cfg_path = configs_dir / f"scale_{size}_{condition_name}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            config_paths.append(str(cfg_path))
            for seed in seeds:
                run_dir = runs_dir / f"scale_{size}" / f"{condition_name}_seed{seed}"
                train_cmd = [
                    sys.executable,
                    str(Path(lewm_root) / "scripts" / "finetune_with_aux.py"),
                    "--config",
                    str(cfg_path),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(run_dir),
                ]
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "bridgeengine.benchmark.lewm_fixed_eval",
                    "--run-dir",
                    str(run_dir),
                    "--dataset-name",
                    heldout_name,
                    "--data-cache-dir",
                    str(out),
                    "--split-file",
                    str(Path(plan_dir) / "splits" / f"scale_{size}_split.json"),
                    "--output-json",
                    str(run_dir / "fixed_eval.json"),
                ]
                commands.append(
                    {
                        "scale_n": int(size),
                        "condition": condition_name,
                        "paradigm": "lewm_cv_aux",
                        "seed": int(seed),
                        "train_cmd": train_cmd,
                        "eval_cmd": eval_cmd,
                        "run_dir": str(run_dir),
                    }
                )

        for family in PI07_FAMILIES:
            for seed in seeds:
                commands.append(
                    {
                        "scale_n": int(size),
                        "condition": family,
                        "paradigm": "bridgeengine_pi07",
                        "seed": int(seed),
                        "note": (
                            "Run through bridgeengine.benchmark.scale_curve. The PowerShell handoff "
                            "groups pi0.7 families by scale using the same split file."
                        ),
                    }
                )

    command_manifest = {
        "output_dir": str(out),
        "plan_dir": str(plan_dir),
        "data_source": "per_episode_npy",
        "episodes_root": str(episodes_root),
        "source_h5_reference": str(source_h5),
        "interpreter": sys.executable,
        "seeds": [int(x) for x in seeds],
        "sizes": [int(x) for x in sizes],
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "h5_exports": h5_exports,
        "config_paths": config_paths,
        "commands": commands,
        "sanity_summary": sanity_summary,
        "stop_rule": "Prepared only. Run scripts/run_head_to_head_100.ps1 to launch the long grid.",
    }
    manifest_path_out = out / "command_manifest.json"
    manifest_path_out.write_text(json.dumps(command_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not dry_run and sanity_summary:
        print("Prepared split HDF5 sanity summary:")
        for row in sanity_summary:
            print(json.dumps(row, sort_keys=True))
    return command_manifest


def export_h5_subset(
    source_h5: str | Path,
    manifest_episode_ids: list[str],
    selected_episode_ids: list[str],
    dataset_name: str,
    output_h5: str | Path,
) -> Path:
    selected = [str(x) for x in selected_episode_ids]
    episode_to_index = {episode_id: idx for idx, episode_id in enumerate(manifest_episode_ids)}
    missing = [episode_id for episode_id in selected if episode_id not in episode_to_index]
    if missing:
        raise ValueError(f"Selected episodes are not in source manifest: {missing[:5]}")
    indices = [episode_to_index[episode_id] for episode_id in selected]
    output = Path(output_h5)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_h5, "r") as src, h5py.File(output, "w") as dst:
        lengths = src["ep_len"][:]
        offsets = src["ep_offset"][:]
        selected_lengths = lengths[indices]
        selected_offsets = []
        total = 0
        for length in selected_lengths:
            selected_offsets.append(total)
            total += int(length)
        dst.create_dataset("ep_len", data=selected_lengths)
        dst.create_dataset("ep_offset", data=selected_offsets)
        for key in src.keys():
            if key in {"ep_len", "ep_offset"}:
                continue
            chunks = []
            for idx in indices:
                start = int(offsets[idx])
                end = start + int(lengths[idx])
                chunks.append(src[key][start:end])
            dst.create_dataset(key, data=_concat(chunks), compression="gzip", compression_opts=4)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["dataset_name"] = dataset_name
        dst.attrs["n_episodes"] = len(indices)
        dst.attrs["total_frames"] = int(total)
        dst.attrs["bridgeengine_selected_episode_ids_json"] = json.dumps(selected)
    return output


def export_split_h5_from_npy(
    episodes_root: str | Path,
    manifest_episode_ids: list[str],
    selected_episode_ids: list[str],
    dataset_name: str,
    output_h5: str | Path,
    n_points: int = DEFAULT_N_POINTS,
    target_size: int = TARGET_SIZE,
) -> Path:
    """Assemble one self-contained split HDF5 from per-episode .npy arrays.

    This is the data-prep path for the head-to-head grid. It reads each selected
    episode's raw .npy arrays, applies the exact same preprocessing as the
    original LeWM export (resize frames/depth to ``target_size``, rescale track
    coordinates to ``target_size``, pad/trim tracks to ``n_points``), and writes a
    single gzip-compressed HDF5 with no external or plugin-codec links. The schema
    matches what ``stable_worldmodel.data.dataset.HDF5Dataset`` and the LeWM
    auxiliary heads expect:

        ep_len           (E,)                     int32
        ep_offset        (E,)                     int64
        pixels           (T, 224, 224, 3)         uint8
        action           (T, 7)                   float32
        observation      (T, 7)                   float32
        depth            (T, 224, 224)            float32   (if available)
        contact          (T,)                     bool      (if available)
        tracks           (T, n_points, 2)         float32   (if available)
        track_visibility (T, n_points)            bool      (if available)
        object_mask      (T, 224, 224)            bool      (if available)

    Ordering is deterministic: episodes are emitted in ``manifest_episode_ids``
    order (restricted to the selected ids) so that train/held-out files built for
    different N are mutually consistent.
    """
    episodes_root = Path(episodes_root)
    selected = [str(x) for x in selected_episode_ids]
    selected_set = set(selected)
    # Deterministic order: follow the manifest, keep only selected ids.
    ordered_ids = [eid for eid in manifest_episode_ids if eid in selected_set]
    missing_from_manifest = [eid for eid in selected if eid not in set(manifest_episode_ids)]
    if missing_from_manifest:
        raise ValueError(f"Selected episodes are not in source manifest: {missing_from_manifest[:5]}")

    episodes: list[dict[str, Any]] = []
    for episode_id in ordered_ids:
        ep_dir = episodes_root / episode_id
        if not ep_dir.is_dir():
            raise FileNotFoundError(f"Episode directory not found: {ep_dir}")
        required = ("frames.npy", "actions.npy", "states.npy")
        missing = [name for name in required if not (ep_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Episode {episode_id} is missing required arrays: {missing}")

        frames = np.load(ep_dir / "frames.npy")
        actions = np.load(ep_dir / "actions.npy").astype(np.float32)
        states = np.load(ep_dir / "states.npy").astype(np.float32)
        steps = min(len(frames), len(actions), len(states))
        orig_h = int(frames.shape[1])
        ep: dict[str, Any] = {
            "episode_id": episode_id,
            "ep_len": int(steps),
            "pixels": _resize_frames(frames[:steps], target_size),
            "action": actions[:steps],
            "observation": states[:steps],
        }

        depth_path = ep_dir / "depth.npy"
        if depth_path.exists():
            ep["depth"] = _resize_depth(np.load(depth_path)[:steps], target_size)

        contact_path = ep_dir / "contact.npy"
        if contact_path.exists():
            ep["contact"] = np.load(contact_path)[:steps].astype(bool)

        tracks_path = ep_dir / "tracks.npy"
        vis_path = ep_dir / "visibility.npy"
        if tracks_path.exists() and vis_path.exists():
            tracks = np.load(tracks_path)[:steps].astype(np.float32)
            visibility = np.load(vis_path)[:steps].astype(bool)
            tracks = _rescale_tracks(tracks, orig_h, target_size)
            tracks, visibility = _pad_or_trim_tracks(tracks, visibility, n_points)
            ep["tracks"] = tracks
            ep["track_visibility"] = visibility

        mask_path = ep_dir / "object_mask.npy"
        if mask_path.exists():
            ep["object_mask"] = _resize_mask(np.load(mask_path)[:steps], target_size)

        episodes.append(ep)

    if not episodes:
        raise ValueError(f"No episodes selected for {dataset_name}")

    has_depth = all("depth" in ep for ep in episodes)
    has_contact = all("contact" in ep for ep in episodes)
    has_tracks = all("tracks" in ep for ep in episodes)
    has_mask = all("object_mask" in ep for ep in episodes)

    ep_lens = np.array([ep["ep_len"] for ep in episodes], dtype=np.int32)
    ep_offsets = np.zeros(len(episodes), dtype=np.int64)
    if len(episodes) > 1:
        ep_offsets[1:] = np.cumsum(ep_lens[:-1])
    total = int(ep_offsets[-1] + ep_lens[-1])

    output = Path(output_h5)
    output.parent.mkdir(parents=True, exist_ok=True)

    def _gzip(name: str, data: np.ndarray) -> None:
        dst.create_dataset(name, data=data, compression="gzip", compression_opts=4)

    with h5py.File(output, "w") as dst:
        dst.create_dataset("ep_len", data=ep_lens)
        dst.create_dataset("ep_offset", data=ep_offsets)
        _gzip("pixels", np.concatenate([ep["pixels"] for ep in episodes], axis=0))
        _gzip("action", np.concatenate([ep["action"] for ep in episodes], axis=0))
        _gzip("observation", np.concatenate([ep["observation"] for ep in episodes], axis=0))
        if has_depth:
            _gzip("depth", np.concatenate([ep["depth"] for ep in episodes], axis=0))
        if has_contact:
            _gzip("contact", np.concatenate([ep["contact"] for ep in episodes], axis=0))
        if has_tracks:
            _gzip("tracks", np.concatenate([ep["tracks"] for ep in episodes], axis=0))
            _gzip("track_visibility", np.concatenate([ep["track_visibility"] for ep in episodes], axis=0))
        if has_mask:
            _gzip("object_mask", np.concatenate([ep["object_mask"] for ep in episodes], axis=0))
        dst.attrs["dataset_name"] = dataset_name
        dst.attrs["n_episodes"] = len(episodes)
        dst.attrs["total_frames"] = int(total)
        dst.attrs["image_size"] = int(target_size)
        dst.attrs["n_track_points"] = int(n_points)
        dst.attrs["data_source"] = "per_episode_npy"
        dst.attrs["bridgeengine_selected_episode_ids_json"] = json.dumps(ordered_ids)
    return output


def _resize_frames(frames: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    T, H, W, C = frames.shape
    if H == size and W == size:
        return np.ascontiguousarray(frames)
    out = np.empty((T, size, size, C), dtype=frames.dtype)
    for i in range(T):
        out[i] = cv2.resize(frames[i], (size, size), interpolation=cv2.INTER_AREA)
    return out


def _resize_depth(depths: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    T, H, W = depths.shape
    depths = depths.astype(np.float32)
    if H == size and W == size:
        return np.ascontiguousarray(depths)
    out = np.empty((T, size, size), dtype=np.float32)
    for i in range(T):
        out[i] = cv2.resize(depths[i], (size, size), interpolation=cv2.INTER_AREA)
    return out


def _resize_mask(masks: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    T, H, W = masks.shape
    if H == size and W == size:
        return np.ascontiguousarray(masks.astype(bool))
    out = np.empty((T, size, size), dtype=bool)
    for i in range(T):
        resized = cv2.resize(masks[i].astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        out[i] = resized.astype(bool)
    return out


def _rescale_tracks(tracks: np.ndarray, orig_size: int, new_size: int = TARGET_SIZE) -> np.ndarray:
    if orig_size == new_size:
        return tracks
    return tracks * (new_size / orig_size)


def _pad_or_trim_tracks(tracks: np.ndarray, visibility: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    T, N, _ = tracks.shape
    if N == n_points:
        return tracks, visibility
    if N > n_points:
        return tracks[:, :n_points], visibility[:, :n_points]
    pad_tracks = np.zeros((T, n_points, 2), dtype=tracks.dtype)
    pad_vis = np.zeros((T, n_points), dtype=visibility.dtype)
    pad_tracks[:, :N] = tracks
    pad_vis[:, :N] = visibility
    return pad_tracks, pad_vis


def _summarize_split_h5(
    size: int, split: dict[str, Any], train_h5: Path, heldout_h5: Path
) -> dict[str, Any]:
    def _describe(path: Path) -> dict[str, Any]:
        with h5py.File(path, "r") as f:
            info = {
                "path": str(path),
                "n_episodes": int(f.attrs.get("n_episodes", len(f["ep_len"]))),
                "total_frames": int(f.attrs.get("total_frames", 0)),
                "keys": sorted(k for k in f.keys() if k not in ("ep_len", "ep_offset")),
            }
            for key in ("pixels", "depth", "tracks", "track_visibility"):
                if key in f:
                    info[f"{key}_shape"] = list(f[key].shape)
            # Force-read a depth/tracks slice to prove the file is self-contained
            # and decodable by this interpreter (no plugin-codec dependency).
            if "depth" in f:
                info["depth_min"] = float(np.asarray(f["depth"][0]).min())
            if "tracks" in f:
                info["tracks_sample_max"] = float(np.asarray(f["tracks"][0]).max())
        return info

    return {
        "scale_n": int(size),
        "split_id": split.get("split_id"),
        "expected_train_episodes": len(split["train_episode_ids"]),
        "expected_heldout_episodes": len(split["heldout_episode_ids"]),
        "train": _describe(train_h5),
        "heldout": _describe(heldout_h5),
    }


def run_command_manifest(
    manifest_path: str | Path,
    *,
    skip_existing: bool = True,
    max_cells: int | None = None,
    cleanup_epoch_checkpoints: bool = True,
    clean_stale_runs: bool = True,
) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cells = [cmd for cmd in manifest["commands"] if cmd.get("paradigm") == "lewm_cv_aux"]
    if max_cells is not None:
        cells = cells[: int(max_cells)]
    for cell in cells:
        run_dir = Path(cell["run_dir"])
        eval_json = run_dir / "fixed_eval.json"
        if skip_existing and eval_json.exists():
            print(f"[skip] {eval_json}")
            continue
        if clean_stale_runs and run_dir.exists() and not eval_json.exists():
            print(f"[clean stale] {run_dir}")
            _safe_rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] scale={cell['scale_n']} condition={cell['condition']} seed={cell['seed']}")
        subprocess.run(cell["train_cmd"], check=True)
        print(f"[eval] {run_dir}")
        subprocess.run(cell["eval_cmd"], check=True)
        if cleanup_epoch_checkpoints:
            _cleanup_epoch_checkpoints(run_dir)


def _cleanup_epoch_checkpoints(run_dir: Path) -> None:
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.exists():
        return
    final = checkpoints / "final" / "full_weights.pt"
    eval_json = run_dir / "fixed_eval.json"
    if not final.exists() or not eval_json.exists():
        return
    for epoch_dir in checkpoints.glob("epoch_*"):
        if epoch_dir.is_dir():
            _safe_rmtree(epoch_dir)


def _safe_rmtree(path: Path) -> None:
    import tempfile
    import shutil

    target = path.resolve()
    allowed_roots = [
        Path.cwd().resolve(),
        Path("D:/lewm_runs").resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]
    if not any(str(target).lower().startswith(str(root).lower()) for root in allowed_roots):
        raise ValueError(f"Refusing to remove unexpected path: {target}")
    shutil.rmtree(target)


def _lewm_config(
    condition: dict[str, Any],
    dataset_name: str,
    data_cache_dir: Path,
    pretrained_path: str,
    max_epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    return {
        "condition_name": condition["condition_name"],
        "pretrained_path": pretrained_path,
        "data_cache_dir": str(data_cache_dir),
        "dataset_name": dataset_name,
        "action_dim": 7,
        "frameskip": 1,
        "history_size": 3,
        "num_preds": 1,
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "sigreg_weight": 0.09,
        "gradient_clip": 1.0,
        "precision": "bf16-mixed",
        "num_workers": 0,
        "seed": 42,
        "freeze": "none",
        "auxiliary_heads": condition["auxiliary_heads"],
    }


def _load_split_payloads(plan_dir: str | Path, sizes: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    result = {}
    for size in sizes:
        path = Path(plan_dir) / "splits" / f"scale_{int(size)}_split.json"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        result[int(size)] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _split_membership(plan_dir: str | Path) -> dict[str, list[str]]:
    result = {}
    split_dir = Path(plan_dir) / "splits"
    if not split_dir.exists():
        return result
    for path in split_dir.glob("scale_*_split.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        result[path.stem + "_train"] = [str(x) for x in data.get("train_episode_ids", [])]
        result[path.stem + "_heldout"] = [str(x) for x in data.get("heldout_episode_ids", [])]
    return result


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Expected manifest list at {path}")
    return data


def _episode_id(row: dict[str, Any]) -> str:
    if "episode_id" in row:
        raw = str(row["episode_id"])
        return raw if raw.startswith("episode_") else f"episode_{int(raw):06d}"
    return f"episode_{int(row['episode_index']):06d}"


def _concat(chunks: list[Any]) -> Any:
    import numpy as np

    if not chunks:
        return np.asarray([])
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and run the preregistered 100-episode head-to-head.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    verify = sub.add_parser("verify-signals")
    verify.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    verify.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    verify.add_argument("--plan-dir", default=str(DEFAULT_PLAN_DIR))

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--plan-dir", default=str(DEFAULT_PLAN_DIR))
    prepare.add_argument("--source-h5", default=str(DEFAULT_SOURCE_H5),
                         help="Reference only; recorded in the manifest. Split HDF5s are built from --episodes-root.")
    prepare.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT),
                         help="Directory of per-episode .npy folders used to assemble self-contained split HDF5s.")
    prepare.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    prepare.add_argument("--output-dir", default="head_to_head_results/run_100")
    prepare.add_argument("--lewm-root", default=str(DEFAULT_LEWM_ROOT))
    prepare.add_argument("--pretrained-path", default=DEFAULT_PRETRAINED)
    prepare.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    prepare.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    prepare.add_argument("--max-epochs", type=int, default=20)
    prepare.add_argument("--batch-size", type=int, default=16)
    prepare.add_argument("--lr", type=float, default=5e-5)
    prepare.add_argument("--dry-run", action="store_true")

    run = sub.add_parser("run-manifest")
    run.add_argument("--manifest", required=True)
    run.add_argument("--no-skip-existing", action="store_true")
    run.add_argument("--max-cells", type=int, default=None)
    run.add_argument("--keep-epoch-checkpoints", action="store_true")
    run.add_argument("--keep-stale-runs", action="store_true")

    args = parser.parse_args()
    if args.cmd == "verify-signals":
        print(json.dumps(verify_signal_files(args.manifest, args.local_root, args.plan_dir), indent=2, sort_keys=True))
    elif args.cmd == "prepare":
        manifest = prepare_handoff(
            plan_dir=args.plan_dir,
            source_h5=args.source_h5,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            lewm_root=args.lewm_root,
            pretrained_path=args.pretrained_path,
            seeds=tuple(args.seeds),
            sizes=tuple(args.sizes),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dry_run=args.dry_run,
            episodes_root=args.episodes_root,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.cmd == "run-manifest":
        run_command_manifest(
            args.manifest,
            skip_existing=not args.no_skip_existing,
            max_cells=args.max_cells,
            cleanup_epoch_checkpoints=not args.keep_epoch_checkpoints,
            clean_stale_runs=not args.keep_stale_runs,
        )


if __name__ == "__main__":
    main()
