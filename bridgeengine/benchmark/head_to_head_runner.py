from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import yaml


DEFAULT_LOCAL_ROOT = Path("D:/bridgedata_v2_subset")
DEFAULT_SOURCE_H5 = DEFAULT_LOCAL_ROOT / "datasets" / "bridgedata_v2_100ep.h5"
DEFAULT_MANIFEST = DEFAULT_LOCAL_ROOT / "manifest_100.json"
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
) -> dict[str, Any]:
    out = Path(output_dir)
    datasets_dir = out / "datasets"
    configs_dir = out / "configs"
    logs_dir = out / "logs"
    runs_dir = out / "runs"
    for path in (datasets_dir, configs_dir, logs_dir, runs_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(manifest_path)
    manifest_ids = [_episode_id(row) for row in manifest]
    split_payloads = _load_split_payloads(plan_dir, sizes)
    source_h5 = Path(source_h5)
    if not source_h5.exists():
        raise FileNotFoundError(f"Source HDF5 not found: {source_h5}")

    h5_exports = []
    config_paths = []
    commands: list[dict[str, Any]] = []
    for size, split in split_payloads.items():
        train_name = f"be_h2h_scale_{size}_train"
        heldout_name = f"be_h2h_scale_{size}_heldout"
        train_h5 = datasets_dir / f"{train_name}.h5"
        heldout_h5 = datasets_dir / f"{heldout_name}.h5"
        if not dry_run:
            export_h5_subset(source_h5, manifest_ids, split["train_episode_ids"], train_name, train_h5)
            export_h5_subset(source_h5, manifest_ids, split["heldout_episode_ids"], heldout_name, heldout_h5)
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
        "plan_dir": str(Path(plan_dir)),
        "source_h5": str(source_h5),
        "seeds": [int(x) for x in seeds],
        "sizes": [int(x) for x in sizes],
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "h5_exports": h5_exports,
        "config_paths": config_paths,
        "commands": commands,
        "stop_rule": "Prepared only. Run scripts/run_head_to_head_100.ps1 to launch the long grid.",
    }
    manifest_path_out = out / "command_manifest.json"
    manifest_path_out.write_text(json.dumps(command_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def run_command_manifest(manifest_path: str | Path, *, skip_existing: bool = True, max_cells: int | None = None) -> None:
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
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] scale={cell['scale_n']} condition={cell['condition']} seed={cell['seed']}")
        subprocess.run(cell["train_cmd"], check=True)
        print(f"[eval] {run_dir}")
        subprocess.run(cell["eval_cmd"], check=True)


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
    prepare.add_argument("--source-h5", default=str(DEFAULT_SOURCE_H5))
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
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.cmd == "run-manifest":
        run_command_manifest(args.manifest, skip_existing=not args.no_skip_existing, max_cells=args.max_cells)


if __name__ == "__main__":
    main()
