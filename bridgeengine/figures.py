from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.quality_gate import evaluate_snapshot_quality


def generate_figures(
    snapshot_id: str,
    data_root: str | Path | None = None,
    output_dir: str | Path = "figures",
    compare_snapshot_id: str | None = None,
) -> dict[str, str]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "quality_summary": output / "quality_summary.png",
        "snapshot_overview": output / "snapshot_overview.png",
        "benchmark_placeholder": output / "benchmark_placeholder.png",
    }
    _quality_summary(snapshot_path, paths["quality_summary"], root, compare_snapshot_id)
    _snapshot_overview(snapshot_path, paths["snapshot_overview"])
    _benchmark_placeholder(paths["benchmark_placeholder"])
    return {name: str(path.resolve()) for name, path in paths.items()}


def _quality_summary(snapshot_path: Path, output_path: Path, root: Path, compare_snapshot_id: str | None) -> None:
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    report = evaluate_snapshot_quality(snapshot_path)
    issue_counts = pd.Series([issue.check for issue in report.issues]).value_counts().sort_index()
    quality_counts = _quality_counts(labels)
    segment_counts = _segment_counts(labels)
    repeated_current = _repeated_text_count(labels)
    repeated_labels = ["current"]
    repeated_values = [repeated_current]
    if compare_snapshot_id:
        compare_path = root / "snapshots" / compare_snapshot_id
        if (compare_path / "labels.parquet").exists():
            repeated_labels.insert(0, "before")
            repeated_values.insert(0, _repeated_text_count(pd.read_parquet(compare_path / "labels.parquet")))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"BridgeEngine Quality Summary: {snapshot_path.name}", fontsize=13)

    ax = axes[0, 0]
    if issue_counts.empty:
        ax.bar(["pass"], [1], color="#2E7D32")
        ax.set_ylabel("status")
    else:
        ax.bar(issue_counts.index.tolist(), issue_counts.values.tolist(), color="#B3261E")
        ax.set_ylabel("failed rows")
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Gate {'PASS' if report.passed else 'FAIL'} by Check")

    ax = axes[0, 1]
    if quality_counts:
        ax.bar([str(k) for k in quality_counts], list(quality_counts.values()), color="#247BA0")
    else:
        ax.bar(["none"], [0], color="#A0A0A0")
    ax.set_title("Quality-Score Distribution")
    ax.set_xlabel("quality")
    ax.set_ylabel("episodes")

    ax = axes[1, 0]
    if segment_counts:
        ax.bar([str(k) for k in segment_counts], list(segment_counts.values()), color="#8C5E2A")
    else:
        ax.bar(["none"], [0], color="#A0A0A0")
    ax.set_title("Segment-Count Distribution")
    ax.set_xlabel("segments per episode")
    ax.set_ylabel("episodes")

    ax = axes[1, 1]
    ax.bar(repeated_labels, repeated_values, color=["#5B6770", "#B3261E"][: len(repeated_values)])
    ax.set_title("Repeated-Text Episodes")
    ax.set_ylabel("episode count")
    ax.set_ylim(0, max(repeated_values + [1]) + 1)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _snapshot_overview(snapshot_path: Path, output_path: Path) -> None:
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    steps = pd.read_parquet(snapshot_path / "steps.parquet")
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    label_counts = labels["labeler_name"].value_counts().sort_index() if not labels.empty else pd.Series(dtype=int)
    subgoal_count = int((labels["labeler_name"] == "subgoal_images").sum()) if not labels.empty else 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f"BridgeEngine Snapshot Overview: {snapshot_path.name}", fontsize=13)

    ax = axes[0]
    names = ["episodes", "steps", "labels", "subgoals"]
    values = [len(episodes), len(steps), len(labels), subgoal_count]
    ax.bar(names, values, color=["#247BA0", "#2E7D32", "#8C5E2A", "#5B6770"])
    ax.set_title("Snapshot Counts")
    ax.set_ylabel("count")

    ax = axes[1]
    if not label_counts.empty:
        ax.bar(label_counts.index.tolist(), label_counts.values.tolist(), color="#247BA0")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.bar(["none"], [0], color="#A0A0A0")
    ax.set_title("Labels Per Labeler")
    ax.set_ylabel("rows")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _benchmark_placeholder(output_path: Path) -> None:
    csv_path = Path("bench_results/bench_results.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if csv_path.exists():
        rows = pd.read_csv(csv_path)
        grouped = (
            rows.groupby("family", as_index=False)
            .agg(latent_mse_mean=("latent_mse", "mean"), latent_mse_std=("latent_mse", "std"))
            .sort_values("latent_mse_mean")
        )
        backend = str(rows.get("benchmark_backend", pd.Series(["unknown"])).iloc[0])
        ax.bar(
            grouped["family"].tolist(),
            grouped["latent_mse_mean"].tolist(),
            yerr=grouped["latent_mse_std"].fillna(0.0).tolist(),
            color="#247BA0",
            capsize=4,
        )
        ax.set_ylabel("latent MSE")
        title = "Real LeWM Smoke Ablation" if backend != "contract_smoke_no_science" else "Benchmark Contract Smoke"
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.text(
            0.0,
            -0.32,
            "Fixed 10/3 episode split; bars are seed mean +/- std. Smoke-scale only.",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
        )
    else:
        families = ["baseline", "rich_text", "rich_text_metadata", "rich_text_metadata_subgoal"]
        ax.bar(families, [0, 0, 0, 0], color="#D0D0D0")
        ax.text(
            0.5,
            0.55,
            "Benchmark intentionally blocked until labels pass quality gates",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("latent MSE")
        ax.set_title("Benchmark Placeholder")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _quality_counts(labels: pd.DataFrame) -> dict[int, int]:
    values = []
    for value in labels.loc[labels["labeler_name"] == "episode_metadata", "metadata_payload_json"].dropna().tolist():
        data = _parse_json(value)
        if data.get("quality") is not None:
            values.append(int(data["quality"]))
    return pd.Series(values, dtype="int64").value_counts().sort_index().to_dict() if values else {}


def _segment_counts(labels: pd.DataFrame) -> dict[int, int]:
    counts = []
    for path in labels.loc[labels["labeler_name"] == "subtask_segmenter", "label_payload_path"].dropna().tolist():
        payload = _read_json(Path(path))
        counts.append(len(payload.get("segments", [])))
    return pd.Series(counts, dtype="int64").value_counts().sort_index().to_dict() if counts else {}


def _repeated_text_count(labels: pd.DataFrame) -> int:
    total = 0
    for path in labels.loc[labels["labeler_name"] == "subtask_segmenter", "label_payload_path"].dropna().tolist():
        payload = _read_json(Path(path))
        texts = [str(s.get("subtask_text", "")).strip().lower() for s in payload.get("segments", [])]
        if texts and len(set(texts)) < len(texts):
            total += 1
    return total


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BridgeEngine data-driven status figures.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--compare-snapshot", default=None)
    args = parser.parse_args()
    paths = generate_figures(
        snapshot_id=args.snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        output_dir=args.output_dir,
        compare_snapshot_id=args.compare_snapshot,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
