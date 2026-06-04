from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from bridgeengine.benchmark.run_grid import run_grid
from bridgeengine.paths import data_root as resolve_data_root


DEFAULT_SIZES = (50, 200, 800)
DEFAULT_SEEDS = (0, 1, 2)


def plan_scale_curve(
    snapshot_id: str,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    data_root: str | Path | None = None,
    output_dir: str | Path = "scale_results",
    heldout_count: int | None = None,
    quality_stratified: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    episode_ids = sorted(str(x) for x in episodes["episode_id"].tolist())
    qualities = _quality_by_episode(snapshot_path)
    ordered = _stable_episode_order(episode_ids, f"{snapshot_id}:{seed}:scale-curve")
    if heldout_count is None:
        heldout_count = max(3, min(25, int(round(len(ordered) * 0.2))))
    if heldout_count <= 0:
        raise ValueError("heldout_count must be positive")
    if heldout_count >= len(ordered):
        raise ValueError("heldout_count must be smaller than the available episode count")

    heldout_ids = sorted(ordered[:heldout_count])
    train_pool = [episode_id for episode_id in ordered if episode_id not in set(heldout_ids)]
    if quality_stratified:
        train_pool = _quality_stratified_order(train_pool, qualities)

    out = Path(output_dir)
    split_dir = out / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    available: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for size in sizes:
        size = int(size)
        train_count = size - heldout_count
        if train_count <= 0:
            unavailable.append(
                {
                    "size": size,
                    "reason": f"size must be greater than heldout_count={heldout_count}",
                }
            )
            continue
        if size > len(episode_ids):
            unavailable.append(
                {
                    "size": size,
                    "reason": f"only {len(episode_ids)} episodes are present in snapshot {snapshot_id}",
                }
            )
            continue
        train_ids = sorted(train_pool[:train_count])
        split = {
            "split_id": _split_id(snapshot_id, size, heldout_ids, train_ids, quality_stratified, seed),
            "snapshot_reference": snapshot_id,
            "size": size,
            "quality_stratified": bool(quality_stratified),
            "seed": int(seed),
            "note": (
                "Scale-curve split. Held-out ids are fixed and disjoint from the nested training "
                "pools for all planned sizes."
            ),
            "train_episode_ids": train_ids,
            "heldout_episode_ids": heldout_ids,
            "quality_distribution_train": _quality_counts(train_ids, qualities),
            "quality_distribution_heldout": _quality_counts(heldout_ids, qualities),
        }
        split_path = split_dir / f"scale_{size}_split.json"
        _write_json(split_path, split)
        available.append(
            {
                "size": size,
                "split_file": str(split_path),
                "train_episode_count": len(train_ids),
                "heldout_episode_count": len(heldout_ids),
                "filter_sql": _episode_filter_sql(train_ids + heldout_ids),
                "quality_distribution_train": split["quality_distribution_train"],
                "quality_distribution_heldout": split["quality_distribution_heldout"],
            }
        )

    plan = {
        "snapshot_id": snapshot_id,
        "available_episode_count": len(episode_ids),
        "heldout_count": int(heldout_count),
        "quality_stratified": bool(quality_stratified),
        "seed": int(seed),
        "requested_sizes": [int(x) for x in sizes],
        "available_sizes": available,
        "unavailable_sizes": unavailable,
        "stop_rule": "Planning only unless --run is passed. Do not run larger curves until Kevin approves the target N.",
    }
    _write_json(out / "scale_curve_plan.json", plan)
    return plan


def run_scale_curve(
    snapshot_id: str,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    data_root: str | Path | None = None,
    output_dir: str | Path = "scale_results",
    heldout_count: int | None = None,
    quality_stratified: bool = False,
    seed: int = 0,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    plan = plan_scale_curve(
        snapshot_id=snapshot_id,
        sizes=sizes,
        data_root=data_root,
        output_dir=output_dir,
        heldout_count=heldout_count,
        quality_stratified=quality_stratified,
        seed=seed,
    )
    rows = []
    for item in plan["available_sizes"]:
        size = int(item["size"])
        results = run_grid(
            snapshot_id=snapshot_id,
            data_root=data_root,
            cut_name=f"scale_curve_{size}",
            filter_sql=item["filter_sql"],
            output_dir=Path(output_dir) / f"size_{size}",
            seeds=seeds,
            split_file=item["split_file"],
        )
        results.insert(0, "scale_n", size)
        rows.append(results)
    if not rows:
        raise ValueError("No requested sizes are available for this snapshot.")
    all_results = pd.concat(rows, ignore_index=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "scale_curve_results.csv"
    all_results.to_csv(csv_path, index=False)
    write_scale_curve_plot(csv_path, out / "scale_curve.png")
    return all_results


def write_scale_curve_plot(csv_path: str | Path, output_path: str | Path) -> Path:
    results = pd.read_csv(csv_path)
    grouped = results.groupby(["scale_n", "family"])["latent_mse"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, family_rows in grouped.groupby("family"):
        family_rows = family_rows.sort_values("scale_n")
        ax.errorbar(
            family_rows["scale_n"],
            family_rows["mean"],
            yerr=family_rows["std"].fillna(0.0),
            marker="o",
            capsize=4,
            label=family,
        )
    ax.set_title("Smoke-Scale LeWM Ablation vs Dataset Size")
    ax.set_xlabel("labeled episodes in slice")
    ax.set_ylabel("held-out latent MSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _quality_by_episode(snapshot_path: Path) -> dict[str, int]:
    labels_path = snapshot_path / "labels.parquet"
    if not labels_path.exists():
        return {}
    labels = pd.read_parquet(labels_path)
    if labels.empty or "metadata_payload_json" not in labels.columns:
        return {}
    result: dict[str, int] = {}
    rows = labels[labels["labeler_name"] == "episode_metadata"]
    for _, row in rows.iterrows():
        payload = row.get("metadata_payload_json")
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        if data.get("quality") is not None:
            result[str(row["episode_id"])] = int(data["quality"])
    return result


def _quality_stratified_order(episode_ids: list[str], qualities: dict[str, int]) -> list[str]:
    buckets: dict[int | str, list[str]] = {}
    for episode_id in episode_ids:
        buckets.setdefault(qualities.get(episode_id, "unknown"), []).append(episode_id)
    for bucket in buckets.values():
        bucket.sort()
    ordered: list[str] = []
    keys = sorted(buckets, key=lambda x: (99 if x == "unknown" else int(x)))
    while any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    return ordered


def _quality_counts(episode_ids: list[str], qualities: dict[str, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode_id in episode_ids:
        key = str(qualities.get(episode_id, "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _stable_episode_order(episode_ids: list[str], salt: str) -> list[str]:
    return sorted(episode_ids, key=lambda episode_id: hashlib.sha256(f"{salt}:{episode_id}".encode("utf-8")).hexdigest())


def _split_id(
    snapshot_id: str,
    size: int,
    heldout_ids: list[str],
    train_ids: list[str],
    quality_stratified: bool,
    seed: int,
) -> str:
    payload = {
        "snapshot_id": snapshot_id,
        "size": int(size),
        "heldout_ids": heldout_ids,
        "train_ids": train_ids,
        "quality_stratified": bool(quality_stratified),
        "seed": int(seed),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"scale_{size}_{digest}"


def _episode_filter_sql(episode_ids: list[str]) -> str:
    quoted = ", ".join("'" + episode_id.replace("'", "''") + "'" for episode_id in sorted(episode_ids))
    return f"e.episode_id IN ({quoted})"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run BridgeEngine scale-curve ablations.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="scale_results")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--heldout-count", type=int, default=None)
    parser.add_argument("--quality-stratified", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--run", action="store_true", help="Actually launch LeWM training for available sizes.")
    args = parser.parse_args()

    if args.run:
        results = run_scale_curve(
            snapshot_id=args.snapshot,
            sizes=tuple(args.sizes),
            data_root=Path(args.data_root) if args.data_root else None,
            output_dir=args.output_dir,
            heldout_count=args.heldout_count,
            quality_stratified=args.quality_stratified,
            seed=args.seed,
            seeds=tuple(args.benchmark_seeds),
        )
        print(results.to_string(index=False))
    else:
        plan = plan_scale_curve(
            snapshot_id=args.snapshot,
            sizes=tuple(args.sizes),
            data_root=Path(args.data_root) if args.data_root else None,
            output_dir=args.output_dir,
            heldout_count=args.heldout_count,
            quality_stratified=args.quality_stratified,
            seed=args.seed,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
