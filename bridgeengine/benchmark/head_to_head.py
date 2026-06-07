from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.benchmark.scale_curve import plan_scale_curve
from bridgeengine.paths import data_root as resolve_data_root


DEFAULT_SNAPSHOT = "snap_2026_05_11_1dde3edf5d_human_gold_labels"
DEFAULT_SIZES = (25, 50, 100)
DEFAULT_TRAIN_SEEDS = (42, 137, 256)
DEFAULT_SPLIT_SEED = 0
DEFAULT_HELDOUT_COUNT = 10
DEFAULT_LEWM_MANIFEST = Path("D:/bridgedata_v2_subset/manifest_100.json")
DEFAULT_LEWM_H5 = Path("D:/bridgedata_v2_subset/datasets/bridgedata_v2_100ep.h5")
DEFAULT_LEWM_RUN_DIRS = (
    Path("D:/lewm_runs/boring3d_fullscale/20260407_180433"),
    Path("D:/lewm_runs/boring3d_fullscale/20260411_133458"),
)
DEFAULT_BRIDGEENGINE_SCALE_CSV = Path("scale_results/human_gold_labels_100/scale_curve_results.csv")


@dataclass(frozen=True)
class HeadToHeadCondition:
    name: str
    paradigm: str
    mechanism: str
    source: str
    cached_at_100: bool
    notes: str


CONDITIONS = (
    HeadToHeadCondition(
        name="baseline",
        paradigm="shared_reference",
        mechanism="no auxiliary prediction target and no pi0.7 conditioning",
        source="LeWM A baseline and BridgeEngine baseline must be evaluated on one fixed split before plotting together",
        cached_at_100=True,
        notes="There are two historical baseline implementations; the preregistered run must report which evaluator produced the plotted baseline.",
    ),
    HeadToHeadCondition(
        name="cv_B_depth_aux",
        paradigm="LeWM perceptive",
        mechanism="Video-Depth-Anything depth as an auxiliary prediction target",
        source="C:/Users/Kevin/projects/LeWM_testbed/configs/finetune/fullscale_B_depth.yaml",
        cached_at_100=True,
        notes="Native LeWM aux-head condition B: predicts 56x56 depth with aux weight 0.1.",
    ),
    HeadToHeadCondition(
        name="cv_D_tracks_aux",
        paradigm="LeWM perceptive",
        mechanism="CoTracker3 point displacements as an auxiliary prediction target",
        source="C:/Users/Kevin/projects/LeWM_testbed/configs/finetune/fullscale_D_tracks.yaml",
        cached_at_100=True,
        notes="Native LeWM aux-head condition D: predicts 400x2 point-track displacements with aux weight 0.1.",
    ),
    HeadToHeadCondition(
        name="cv_E_depth_tracks_aux",
        paradigm="LeWM perceptive",
        mechanism="VDA depth and CoTracker3 tracks as auxiliary prediction targets",
        source="C:/Users/Kevin/projects/LeWM_testbed/configs/finetune/fullscale_E_depth_tracks.yaml",
        cached_at_100=True,
        notes="Native LeWM aux-head condition E: combines depth and track aux heads, each weight 0.1.",
    ),
    HeadToHeadCondition(
        name="pi07_rich_text",
        paradigm="BridgeEngine pi0.7",
        mechanism="subtask text as a conditioning input",
        source="bridgeengine.benchmark.train_lewm family rich_text",
        cached_at_100=False,
        notes="Uses Kevin-reviewed boundaries where available and VLM boundaries elsewhere.",
    ),
    HeadToHeadCondition(
        name="pi07_rich_text_metadata",
        paradigm="BridgeEngine pi0.7",
        mechanism="subtask text plus speed, quality, mistake, and control metadata as conditioning inputs",
        source="bridgeengine.benchmark.train_lewm family rich_text_metadata",
        cached_at_100=False,
        notes="Uses Kevin-calibrated quality scores in the human-gold snapshot.",
    ),
    HeadToHeadCondition(
        name="pi07_full_metadata_subgoal",
        paradigm="BridgeEngine pi0.7",
        mechanism="subtask text, metadata, and end-of-segment subgoal frame as conditioning inputs",
        source="bridgeengine.benchmark.train_lewm family rich_text_metadata_subgoal",
        cached_at_100=False,
        notes="Current POC has the full-stack subgoal family, not an isolated subgoal-only family.",
    ),
)


def build_head_to_head_plan(
    snapshot_id: str = DEFAULT_SNAPSHOT,
    output_dir: str | Path = "head_to_head_results/preregistered_100",
    data_root: str | Path | None = None,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    heldout_count: int = DEFAULT_HELDOUT_COUNT,
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_seeds: tuple[int, ...] = DEFAULT_TRAIN_SEEDS,
    lewm_manifest: str | Path = DEFAULT_LEWM_MANIFEST,
    lewm_h5: str | Path = DEFAULT_LEWM_H5,
    lewm_run_dirs: tuple[str | Path, ...] = DEFAULT_LEWM_RUN_DIRS,
    bridgeengine_scale_csv: str | Path = DEFAULT_BRIDGEENGINE_SCALE_CSV,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shared = compare_snapshot_to_lewm_manifest(snapshot_id, data_root, lewm_manifest)
    scale_plan = plan_scale_curve(
        snapshot_id=snapshot_id,
        sizes=tuple(int(x) for x in sizes),
        data_root=data_root,
        output_dir=out,
        heldout_count=int(heldout_count),
        quality_stratified=True,
        seed=int(split_seed),
    )
    runtime = estimate_runtime(
        sizes=tuple(int(x) for x in sizes),
        train_seeds=tuple(int(x) for x in train_seeds),
        lewm_run_dirs=tuple(Path(x) for x in lewm_run_dirs),
        bridgeengine_scale_csv=Path(bridgeengine_scale_csv),
    )
    plan = {
        "snapshot_id": snapshot_id,
        "created_by": "bridgeengine.benchmark.head_to_head",
        "metric": "held-out next-latent MSE",
        "training_config_reference": {
            "lewm_aux_native": {
                "optimizer": "AdamW",
                "lr": 5e-5,
                "batch_size": 16,
                "aux_weight": 0.1,
                "epochs": 20,
                "precision": "bf16-mixed",
                "source": "LeWM_testbed configs/finetune/fullscale_*.yaml",
            },
            "bridgeengine_pi07_current": {
                "optimizer": "AdamW",
                "lr": "3e-4 by default",
                "batch_size": 12,
                "epochs": 8,
                "source": "bridgeengine.benchmark.train_lewm",
                "note": (
                    "This is the current pi0.7-conditioning adapter path. It is not the same "
                    "mechanism as the LeWM aux-head CV conditions."
                ),
            },
        },
        "mechanism_disclosure": (
            "LeWM CV signals are native auxiliary prediction targets. BridgeEngine pi0.7 "
            "signals are conditioning inputs. The planned plot is an annotation-strategy "
            "comparison, not a pure signal-content comparison."
        ),
        "shared_episode_check": shared,
        "lewm_artifacts": {
            "manifest": str(Path(lewm_manifest)),
            "h5": str(Path(lewm_h5)),
            "h5_exists": Path(lewm_h5).exists(),
            "cached_run_dirs": [str(Path(x)) for x in lewm_run_dirs],
        },
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "split_seed": int(split_seed),
        "train_seeds": [int(x) for x in train_seeds],
        "sizes": [int(x) for x in sizes],
        "scale_plan_file": str(out / "scale_curve_plan.json"),
        "runtime_estimate": runtime,
        "blocking_notes": [
            (
                "Do not put old LeWM CV numbers and BridgeEngine pi0.7 numbers on one axis "
                "until they have been evaluated on the same fixed split."
            ),
            (
                "LeWM_testbed/scripts/evaluate_boring3d.py currently creates a per-seed random "
                "90/10 split; it must be adapted or bypassed for the preregistered fixed-split evaluator."
            ),
            (
                "The 100-episode HDF5 cache contains VDA depth and CoTracker3 tracks. It does "
                "not contain the pilot-only mask/centroid/shape fields, so the first shared "
                "100-episode comparison should be A/B/D/E plus pi0.7 families."
            ),
        ],
        "stop_rule": (
            "This command is planning-only. Launch the unified grid only after Kevin approves "
            "the seed count and the fixed-split evaluator path."
        ),
    }
    _write_json(out / "head_to_head_plan.json", plan)
    _write_runtime_markdown(out / "runtime_estimate.md", plan)
    return plan


def compare_snapshot_to_lewm_manifest(
    snapshot_id: str,
    data_root: str | Path | None = None,
    lewm_manifest: str | Path = DEFAULT_LEWM_MANIFEST,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    episodes_path = root / "snapshots" / snapshot_id / "episodes.parquet"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Snapshot episodes.parquet not found: {episodes_path}")
    if not Path(lewm_manifest).exists():
        raise FileNotFoundError(f"LeWM manifest not found: {lewm_manifest}")
    episodes = pd.read_parquet(episodes_path)
    snapshot_ids = sorted(str(x) for x in episodes["episode_id"].tolist())
    manifest_payload = json.loads(Path(lewm_manifest).read_text(encoding="utf-8-sig"))
    manifest_ids = sorted(_manifest_episode_id(row) for row in manifest_payload)
    return {
        "snapshot_episode_count": len(snapshot_ids),
        "lewm_manifest_episode_count": len(manifest_ids),
        "same_episode_set": set(snapshot_ids) == set(manifest_ids),
        "snapshot_only": sorted(set(snapshot_ids) - set(manifest_ids)),
        "lewm_only": sorted(set(manifest_ids) - set(snapshot_ids)),
        "first_ten_episode_ids": snapshot_ids[:10],
    }


def estimate_runtime(
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    train_seeds: tuple[int, ...] = DEFAULT_TRAIN_SEEDS,
    lewm_run_dirs: tuple[Path, ...] = DEFAULT_LEWM_RUN_DIRS,
    bridgeengine_scale_csv: Path = DEFAULT_BRIDGEENGINE_SCALE_CSV,
) -> dict[str, Any]:
    lewm_rows = _read_lewm_ablation_rows(lewm_run_dirs)
    lewm = _estimate_lewm_seconds(lewm_rows, sizes, train_seeds)
    bridgeengine = _estimate_bridgeengine_seconds(bridgeengine_scale_csv, train_seeds)
    total_from_scratch = lewm["estimated_all_scales_from_scratch_seconds"] + bridgeengine["estimated_seconds"]
    total_incremental_reuse_cv_100 = lewm["estimated_incremental_reusing_cached_100_seconds"] + bridgeengine["estimated_seconds"]
    return {
        "requested_sizes": [int(x) for x in sizes],
        "requested_train_seeds": [int(x) for x in train_seeds],
        "lewm_cv_aux": lewm,
        "bridgeengine_pi07": bridgeengine,
        "total_from_scratch_seconds": round(float(total_from_scratch), 3),
        "total_from_scratch_hours": round(float(total_from_scratch) / 3600.0, 3),
        "total_incremental_reusing_cached_cv_100_seconds": round(float(total_incremental_reuse_cv_100), 3),
        "total_incremental_reusing_cached_cv_100_hours": round(float(total_incremental_reuse_cv_100) / 3600.0, 3),
        "estimate_notes": [
            "LeWM CV estimates scale cached N=100 elapsed times linearly by N.",
            "BridgeEngine pi0.7 estimate scales the existing human-gold 2-seed scale curve to the requested seed count.",
            "Evaluation and fixed-split adapter development time are not included.",
        ],
    }


def _manifest_episode_id(row: dict[str, Any]) -> str:
    if "episode_id" in row:
        value = str(row["episode_id"])
        return value if value.startswith("episode_") else f"episode_{int(value):06d}"
    if "episode_index" in row:
        return f"episode_{int(row['episode_index']):06d}"
    raise KeyError("Manifest rows must contain episode_id or episode_index")


def _read_lewm_ablation_rows(run_dirs: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        path = run_dir / "ablation_results.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows.extend(row for row in payload if row.get("status") == "ok")
    return rows


def _estimate_lewm_seconds(
    rows: list[dict[str, Any]],
    sizes: tuple[int, ...],
    train_seeds: tuple[int, ...],
) -> dict[str, Any]:
    if not rows:
        return {
            "cached_rows_found": 0,
            "cached_100_seconds": 0.0,
            "average_100_seconds_by_condition": {},
            "estimated_all_scales_from_scratch_seconds": 0.0,
            "estimated_incremental_reusing_cached_100_seconds": 0.0,
        }
    df = pd.DataFrame(rows)
    df["elapsed_sec"] = df["elapsed_sec"].astype(float)
    avg = df.groupby("condition")["elapsed_sec"].mean().to_dict()
    cached_100_seconds = float(df["elapsed_sec"].sum())
    requested_conditions = ("A", "B", "D", "E")
    seed_count = len(train_seeds)
    estimated_all = 0.0
    estimated_incremental = 0.0
    for condition in requested_conditions:
        base_seconds = float(avg.get(condition, 0.0))
        for size in sizes:
            seconds = base_seconds * (float(size) / 100.0) * seed_count
            estimated_all += seconds
            if int(size) != 100:
                estimated_incremental += seconds
    return {
        "cached_rows_found": int(len(rows)),
        "cached_100_seconds": round(cached_100_seconds, 3),
        "cached_100_hours": round(cached_100_seconds / 3600.0, 3),
        "average_100_seconds_by_condition": {str(k): round(float(v), 3) for k, v in sorted(avg.items())},
        "estimated_all_scales_from_scratch_seconds": round(estimated_all, 3),
        "estimated_all_scales_from_scratch_hours": round(estimated_all / 3600.0, 3),
        "estimated_incremental_reusing_cached_100_seconds": round(estimated_incremental, 3),
        "estimated_incremental_reusing_cached_100_hours": round(estimated_incremental / 3600.0, 3),
        "conditions": list(requested_conditions),
    }


def _estimate_bridgeengine_seconds(scale_csv: Path, train_seeds: tuple[int, ...]) -> dict[str, Any]:
    if not scale_csv.exists():
        return {
            "source_csv": str(scale_csv),
            "observed_rows_found": 0,
            "observed_seconds": 0.0,
            "observed_seed_count": 0,
            "estimated_seconds": 0.0,
            "estimated_hours": 0.0,
        }
    df = pd.read_csv(scale_csv)
    if "wall_clock_seconds_training" not in df.columns:
        return {
            "source_csv": str(scale_csv),
            "observed_rows_found": int(len(df)),
            "observed_seconds": 0.0,
            "observed_seed_count": int(df["seed"].nunique()) if "seed" in df.columns else 0,
            "estimated_seconds": 0.0,
            "estimated_hours": 0.0,
            "note": "CSV lacks wall_clock_seconds_training.",
        }
    observed_seconds = float(df["wall_clock_seconds_training"].sum())
    observed_seed_count = int(df["seed"].nunique()) if "seed" in df.columns else 0
    multiplier = (len(train_seeds) / observed_seed_count) if observed_seed_count else 0.0
    estimated = observed_seconds * multiplier
    by_size_family = (
        df.groupby(["scale_n", "family"])["wall_clock_seconds_training"].mean().reset_index().to_dict(orient="records")
        if {"scale_n", "family"}.issubset(df.columns)
        else []
    )
    return {
        "source_csv": str(scale_csv),
        "observed_rows_found": int(len(df)),
        "observed_seed_count": observed_seed_count,
        "observed_seconds": round(observed_seconds, 3),
        "observed_hours": round(observed_seconds / 3600.0, 3),
        "estimated_seconds": round(estimated, 3),
        "estimated_hours": round(estimated / 3600.0, 3),
        "per_size_family_mean_seconds": [
            {
                "scale_n": int(row["scale_n"]),
                "family": str(row["family"]),
                "seconds": round(float(row["wall_clock_seconds_training"]), 3),
            }
            for row in by_size_family
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_runtime_markdown(path: Path, plan: dict[str, Any]) -> None:
    runtime = plan["runtime_estimate"]
    shared = plan["shared_episode_check"]
    lines = [
        "# Head-To-Head Runtime Estimate",
        "",
        f"Snapshot: `{plan['snapshot_id']}`",
        "",
        "## Shared Episode Check",
        "",
        f"- same 100-episode set: `{shared['same_episode_set']}`",
        f"- BridgeEngine episodes: {shared['snapshot_episode_count']}",
        f"- LeWM manifest episodes: {shared['lewm_manifest_episode_count']}",
        "",
        "## Planned Conditions",
        "",
        "| condition | paradigm | mechanism |",
        "|---|---|---|",
    ]
    for condition in plan["conditions"]:
        lines.append(f"| `{condition['name']}` | {condition['paradigm']} | {condition['mechanism']} |")
    lines.extend(
        [
            "",
            "## Estimate",
            "",
            f"- requested sizes: `{runtime['requested_sizes']}`",
            f"- requested train seeds: `{runtime['requested_train_seeds']}`",
            f"- LeWM CV fullscale cached N=100 training: {runtime['lewm_cv_aux']['cached_100_hours']:.3f} hours",
            f"- LeWM CV all scales from scratch estimate: {runtime['lewm_cv_aux']['estimated_all_scales_from_scratch_hours']:.3f} hours",
            f"- LeWM CV incremental estimate reusing cached N=100: {runtime['lewm_cv_aux']['estimated_incremental_reusing_cached_100_hours']:.3f} hours",
            f"- BridgeEngine pi0.7 3-seed estimate: {runtime['bridgeengine_pi07']['estimated_hours']:.3f} hours",
            f"- total from scratch estimate: {runtime['total_from_scratch_hours']:.3f} hours",
            f"- total incremental estimate reusing cached CV N=100: {runtime['total_incremental_reusing_cached_cv_100_hours']:.3f} hours",
            "",
            "## Stop Reason",
            "",
            "Do not launch the unified grid until the fixed-split evaluator path is chosen. "
            "The cached LeWM evaluator currently uses a per-seed random 90/10 split, while the preregistered comparison requires one shared fixed split.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a preregistered BridgeEngine-vs-LeWM head-to-head plan.")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="head_to_head_results/preregistered_100")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--heldout-count", type=int, default=DEFAULT_HELDOUT_COUNT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--lewm-manifest", default=str(DEFAULT_LEWM_MANIFEST))
    parser.add_argument("--lewm-h5", default=str(DEFAULT_LEWM_H5))
    parser.add_argument("--bridgeengine-scale-csv", default=str(DEFAULT_BRIDGEENGINE_SCALE_CSV))
    args = parser.parse_args()

    plan = build_head_to_head_plan(
        snapshot_id=args.snapshot,
        output_dir=args.output_dir,
        data_root=Path(args.data_root) if args.data_root else None,
        sizes=tuple(args.sizes),
        heldout_count=args.heldout_count,
        split_seed=args.split_seed,
        train_seeds=tuple(args.train_seeds),
        lewm_manifest=Path(args.lewm_manifest),
        lewm_h5=Path(args.lewm_h5),
        bridgeengine_scale_csv=Path(args.bridgeengine_scale_csv),
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
