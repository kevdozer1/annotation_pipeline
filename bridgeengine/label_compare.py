from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.scoring import metadata_quality, task_success_quality


def compare_snapshots(
    left_snapshot: str,
    right_snapshot: str,
    data_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    left_path = root / "snapshots" / left_snapshot
    right_path = root / "snapshots" / right_snapshot
    left = _metadata_frame(left_path)
    right = _metadata_frame(right_path)
    merged = left.merge(right, on="episode_id", suffixes=("_left", "_right"))
    if merged.empty:
        raise ValueError("No overlapping episode metadata rows to compare.")

    merged["curation_abs_diff"] = (merged["curation_quality_left"] - merged["curation_quality_right"]).abs()
    merged["task_success_abs_diff"] = (merged["task_success_quality_left"] - merged["task_success_quality_right"]).abs()
    merged["keep_agree"] = merged["curation_keep_left"] == merged["curation_keep_right"]
    if "task" not in merged.columns:
        merged["task"] = merged.get("task_left", merged.get("task_right", ""))
    report = {
        "left_snapshot_id": left_snapshot,
        "right_snapshot_id": right_snapshot,
        "overlap_episode_count": int(len(merged)),
        "curation_quality": _agreement(merged["curation_quality_left"], merged["curation_quality_right"]),
        "task_success_quality": _agreement(merged["task_success_quality_left"], merged["task_success_quality_right"]),
        "keep_decision_agreement": round(float(merged["keep_agree"].mean()), 4),
        "left_distribution": _distribution(merged["curation_quality_left"]),
        "right_distribution": _distribution(merged["curation_quality_right"]),
        "top_disagreements": _top_disagreements(merged),
    }
    target = Path(output_path) if output_path else right_path / f"compare_{left_snapshot}_vs_{right_snapshot}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def format_compare_report(report: dict[str, Any]) -> str:
    lines = [
        f"Label comparison: {report['left_snapshot_id']} vs {report['right_snapshot_id']}",
        f"Overlapping episodes: {report['overlap_episode_count']}",
        "Curation-quality agreement:",
        f"- exact: {report['curation_quality']['exact_agreement']:.3f}",
        f"- within one: {report['curation_quality']['within_one_agreement']:.3f}",
        f"- mean abs diff: {report['curation_quality']['mean_abs_diff']:.3f}",
        "Task-success agreement:",
        f"- exact: {report['task_success_quality']['exact_agreement']:.3f}",
        f"- within one: {report['task_success_quality']['within_one_agreement']:.3f}",
        f"- mean abs diff: {report['task_success_quality']['mean_abs_diff']:.3f}",
        f"Keep-decision agreement: {report['keep_decision_agreement']:.3f}",
        f"Left curation distribution: {report['left_distribution']}",
        f"Right curation distribution: {report['right_distribution']}",
        "Top disagreements:",
    ]
    for row in report["top_disagreements"][:10]:
        lines.append(
            "- "
            f"{row['episode_id']}: left={row['curation_quality_left']} right={row['curation_quality_right']} "
            f"task={row['task']}"
        )
    return "\n".join(lines)


def _metadata_frame(snapshot_path: Path) -> pd.DataFrame:
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")[["episode_id", "language_instruction"]]
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    rows = []
    for row in labels.loc[labels["labeler_name"] == "episode_metadata"].to_dict("records"):
        metadata = _parse_json(row.get("metadata_payload_json"))
        curation = metadata_quality(metadata)
        task_success = task_success_quality(metadata)
        if curation is None:
            continue
        rows.append(
            {
                "episode_id": str(row["episode_id"]),
                "curation_quality": int(curation),
                "task_success_quality": int(task_success if task_success is not None else curation),
                "curation_keep": int(curation) >= 4,
                "reason": str(metadata.get("reason", "")),
                "scoring_reason": str(metadata.get("scoring_reason", "")),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.merge(episodes, on="episode_id", how="left").rename(columns={"language_instruction": "task"})


def _agreement(left: pd.Series, right: pd.Series) -> dict[str, float]:
    diff = (left.astype(int) - right.astype(int)).abs()
    return {
        "exact_agreement": round(float((diff == 0).mean()), 4),
        "within_one_agreement": round(float((diff <= 1).mean()), 4),
        "mean_abs_diff": round(float(diff.mean()), 4),
    }


def _distribution(values: pd.Series) -> dict[int, int]:
    return {int(k): int(v) for k, v in values.astype(int).value_counts().sort_index().to_dict().items()}


def _top_disagreements(merged: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "episode_id",
        "task",
        "curation_quality_left",
        "curation_quality_right",
        "task_success_quality_left",
        "task_success_quality_right",
        "curation_abs_diff",
        "reason_left",
        "reason_right",
        "scoring_reason_left",
        "scoring_reason_right",
    ]
    rows = merged.sort_values(["curation_abs_diff", "episode_id"], ascending=[False, True]).head(15)[cols].to_dict("records")
    for row in rows:
        for key in ("curation_quality_left", "curation_quality_right", "task_success_quality_left", "task_success_quality_right"):
            row[key] = int(row[key])
        row["curation_abs_diff"] = int(row["curation_abs_diff"])
    return rows


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BridgeEngine metadata labels between two snapshots.")
    parser.add_argument("--left-snapshot", required=True)
    parser.add_argument("--right-snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()
    report = compare_snapshots(
        args.left_snapshot,
        args.right_snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        output_path=Path(args.output_path) if args.output_path else None,
    )
    print(format_compare_report(report))


if __name__ == "__main__":
    main()
