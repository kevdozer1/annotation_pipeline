from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bridgeengine.benchmark.plot import write_bar_chart
from bridgeengine.benchmark.train_lewm import FAMILIES, run_family_seed
from bridgeengine.export.cut import export_cut
from bridgeengine.goldset import reliability_report
from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.quality_gate import evaluate_snapshot_quality


def run_grid(
    snapshot_id: str,
    data_root: str | Path | None = None,
    cut_name: str = "cut_mode_a_all_labels",
    output_dir: str | Path = "bench_results",
    seeds: tuple[int, ...] = (0, 1, 2),
    allow_scaffolding_labels: bool = False,
    gold_file: str | Path | None = None,
) -> pd.DataFrame:
    root = resolve_data_root(data_root)
    _assert_benchmarkable_labels(root / "snapshots" / snapshot_id, allow_scaffolding_labels)
    cut_root = root / "training_cuts"
    cut_manifest = export_cut(snapshot_id, "TRUE", cut_root, cut_name, data_root=root)
    cut_path = cut_root / cut_name
    scale = int(cut_manifest["episode_count"])
    rows = []
    for family in FAMILIES:
        for seed in seeds:
            rows.append(
                run_family_seed(
                    cut_path,
                    family=family,
                    seed=seed,
                    scale=scale,
                    contract_smoke=allow_scaffolding_labels,
                )
            )
    results = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "bench_results.csv"
    results.to_csv(csv_path, index=False)
    write_bar_chart(csv_path, out / "bench_bar.png")
    reliability = None
    if gold_file is not None:
        reliability = reliability_report(snapshot_id, gold_file, data_root=root)
        (out / "gold_reliability.json").write_text(
            json.dumps(reliability, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_summary(results, out / "bench_summary.md", reliability)
    return results


def _assert_benchmarkable_labels(snapshot_path: Path, allow_scaffolding_labels: bool) -> None:
    if allow_scaffolding_labels:
        return
    report = evaluate_snapshot_quality(snapshot_path)
    report.raise_if_failed()


def _write_summary(results: pd.DataFrame, path: Path, reliability: dict | None = None) -> None:
    backend = str(results.get("benchmark_backend", pd.Series(["unknown"])).iloc[0])
    grouped = results.groupby("family")["latent_mse"].mean().sort_values()
    baseline = float(grouped["baseline"])
    best = grouped.index[0]
    best_delta = (float(grouped[best]) - baseline) / baseline * 100.0
    stds = results.groupby("family")["latent_mse"].std().fillna(0.0)
    metadata_mean = float(grouped.get("rich_text_metadata", float("nan")))
    metadata_std = float(stds.get("rich_text_metadata", 0.0))
    baseline_std = float(stds.get("baseline", 0.0))
    metadata_delta = (metadata_mean - baseline) / baseline * 100.0
    beyond_seed_noise = abs(metadata_mean - baseline) > (baseline_std + metadata_std)
    if backend == "contract_smoke_no_science":
        result_sentence = (
            "This is a contract-smoke run only: it validates artifact shape and CI plumbing, "
            "not learned model behavior."
        )
    else:
        result_sentence = (
            f"Rich-text + metadata is {metadata_delta:+.1f}% relative to baseline "
            f"({'beyond' if beyond_seed_noise else 'within'} seed-noise by the simple std-sum check)."
        )
    if best == "baseline":
        best_sentence = (
            f"Baseline is the best-scoring family in this POC table ({baseline:.6f} latent MSE); "
            "no richer conditioning family beats it on mean held-out MSE. "
        )
    else:
        best_sentence = (
            f"{best} is the best-scoring family in this POC table ({grouped[best]:.6f} latent MSE), "
            f"beating baseline by {abs(best_delta):.1f}% at 13 episodes. "
        )
    reliability_text = ""
    if reliability is not None:
        reliability_text = (
            "\n\nGold-set reliability alongside this ablation:\n\n"
            f"- reviewed episodes: {reliability.get('reviewed_episode_count')} / {reliability.get('episode_count')}\n"
            f"- subtask boundary temporal IoU mean: {reliability.get('subtask_boundary_temporal_iou_mean')}\n"
            f"- quality exact agreement: {reliability.get('quality_exact_agreement')}\n"
            f"- quality within-one agreement: {reliability.get('quality_within_one_agreement')}\n"
            f"- subgoal selection agreement: {reliability.get('subgoal_selection_agreement')}\n"
        )
    text = (
        "# Benchmark Summary\n\n"
        f"The Mode A grid contains {len(results)} runs: 4 annotation families x 3 seeds. "
        f"{best_sentence}"
        "The rich-text conditions use VLM-derived subtask segmentation quality, so this should be read as a test of "
        "whether hosted-VLM segmentation is enough to produce the pi0.7 effect, not as a claim about ideal human-validated segmentation. "
        f"Benchmark backend: `{backend}`. {result_sentence} "
        "At 13 episodes this is a smoke-scale ablation, not a robust conclusion."
        f"{reliability_text}\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BridgeEngine Mode A benchmark grid.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="bench_results")
    parser.add_argument(
        "--gold-file",
        default=None,
        help="Optional filled gold-set JSON. If provided, reliability metrics are written next to the ablation.",
    )
    parser.add_argument(
        "--allow-scaffolding-labels",
        action="store_true",
        help="Allow fallback labels for CI/plumbing tests only. Do not use for demo claims.",
    )
    args = parser.parse_args()
    results = run_grid(
        snapshot_id=args.snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        output_dir=args.output_dir,
        allow_scaffolding_labels=args.allow_scaffolding_labels,
        gold_file=Path(args.gold_file) if args.gold_file else None,
    )
    print(results.to_string(index=False))
    if args.gold_file:
        reliability_path = Path(args.output_dir) / "gold_reliability.json"
        if reliability_path.exists():
            print(reliability_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
