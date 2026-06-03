from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bridgeengine.benchmark.plot import write_bar_chart
from bridgeengine.benchmark.train_lewm import FAMILIES, run_family_seed
from bridgeengine.export.cut import export_cut
from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.quality_gate import evaluate_snapshot_quality


def run_grid(
    snapshot_id: str,
    data_root: str | Path | None = None,
    cut_name: str = "cut_mode_a_all_labels",
    output_dir: str | Path = "bench_results",
    seeds: tuple[int, ...] = (0, 1, 2),
    allow_scaffolding_labels: bool = False,
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
            rows.append(run_family_seed(cut_path, family=family, seed=seed, scale=scale))
    results = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "bench_results.csv"
    results.to_csv(csv_path, index=False)
    write_bar_chart(csv_path, out / "bench_bar.png")
    _write_summary(results, out / "bench_summary.md")
    return results


def _assert_benchmarkable_labels(snapshot_path: Path, allow_scaffolding_labels: bool) -> None:
    if allow_scaffolding_labels:
        return
    report = evaluate_snapshot_quality(snapshot_path)
    report.raise_if_failed()


def _write_summary(results: pd.DataFrame, path: Path) -> None:
    grouped = results.groupby("family")["latent_mse"].mean().sort_values()
    baseline = float(grouped["baseline"])
    best = grouped.index[0]
    best_delta = (float(grouped[best]) - baseline) / baseline * 100.0
    text = (
        "# Benchmark Summary\n\n"
        f"The Mode A proxy grid contains {len(results)} runs: 4 annotation families x 3 seeds. "
        f"{best} is the best-scoring family in this POC table ({grouped[best]:.6f} latent MSE), "
        f"beating baseline by {abs(best_delta):.1f}% at 13 episodes. "
        "The rich-text conditions use VLM-derived subtask segmentation quality, so this should be read as a test of "
        "whether Moondream-style segmentation is enough to produce the pi0.7 effect, not as a claim about ideal human-validated segmentation. "
        "These numbers are deterministic CPU proxy results wired to the LEWM experiment contract; "
        "the project is ready for a real GPU LEWM sweep by replacing `bridgeengine.benchmark.train_lewm.run_family_seed` "
        "with the heavyweight training adapter.\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BridgeEngine Mode A benchmark grid.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="bench_results")
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
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
