from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.quality_gate import evaluate_snapshot_quality


def inspect_snapshot(
    snapshot_id: str,
    data_root: str | Path | None = None,
    max_episodes: int = 4,
) -> str:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    selected = _select_episodes(episodes, max_episodes)
    lines: list[str] = []
    lines.append(f"Snapshot: {snapshot_id}")
    lines.append(f"Episodes: {len(episodes)}")
    lines.append("")
    lines.extend(_metadata_distribution(labels))
    lines.extend(_fallback_report(labels))
    lines.append("")
    lines.append(evaluate_snapshot_quality(snapshot_path).to_text())
    lines.append("")
    for episode_id in selected:
        ep = episodes[episodes["episode_id"] == episode_id].iloc[0].to_dict()
        ep_labels = labels[labels["episode_id"] == episode_id]
        lines.append("=" * 88)
        lines.append(f"{episode_id} | steps={ep.get('num_steps')} | task={ep.get('language_instruction')}")
        lines.append("")
        lines.append("Subtask segments:")
        segment_payload = _first_payload(ep_labels, "subtask_segmenter")
        lines.append(_json_block(segment_payload.get("segments", [])))
        raw_path = segment_payload.get("raw_vlm_output_path")
        if raw_path:
            lines.append(f"Raw segmenter output: {raw_path}")
            raw_answer = _raw_answer(raw_path)
            if raw_answer:
                lines.append(f"Raw segmenter answer: {raw_answer[:700]}")
        lines.append("")
        lines.append("Metadata payload:")
        metadata_payload = _first_payload(ep_labels, "episode_metadata")
        lines.append(_json_block(metadata_payload.get("metadata", {})))
        raw_path = metadata_payload.get("raw_vlm_output_path")
        if raw_path:
            lines.append(f"Raw metadata output: {raw_path}")
            raw_answer = _raw_answer(raw_path)
            if raw_answer:
                lines.append(f"Raw metadata answer: {raw_answer[:700]}")
        lines.append("")
        lines.append("Subgoal image paths:")
        for path in ep_labels["subgoal_image_path"].dropna().tolist():
            lines.append(f"- {path}")
        lines.append("")
    return "\n".join(lines)


def _select_episodes(episodes: pd.DataFrame, max_episodes: int) -> list[str]:
    ordered = episodes.sort_values(["num_steps", "episode_id"])
    candidates = []
    if not ordered.empty:
        candidates.append(str(ordered.iloc[0]["episode_id"]))
        candidates.append(str(ordered.iloc[-1]["episode_id"]))
    task_text = episodes["language_instruction"].fillna("").str.lower()
    easy = episodes[task_text.str.contains("can|pot|cup|sink", regex=True)]
    if not easy.empty:
        candidates.append(str(easy.sort_values("episode_id").iloc[0]["episode_id"]))
    if len(episodes) > 0:
        candidates.append(str(episodes.sort_values("episode_id").iloc[len(episodes) // 2]["episode_id"]))
    selected = []
    for episode_id in candidates:
        if episode_id not in selected:
            selected.append(episode_id)
        if len(selected) >= max_episodes:
            break
    return selected


def _metadata_distribution(labels: pd.DataFrame) -> list[str]:
    rows = labels[labels["labeler_name"] == "episode_metadata"]
    quality_counts: dict[str, int] = {}
    mistake_counts: dict[str, int] = {}
    for value in rows["metadata_payload_json"].dropna().tolist():
        data = _parse_json(value)
        quality_counts[str(data.get("quality"))] = quality_counts.get(str(data.get("quality")), 0) + 1
        mistake_counts[str(data.get("mistake"))] = mistake_counts.get(str(data.get("mistake")), 0) + 1
    return [
        "Metadata distribution:",
        f"- quality counts: {dict(sorted(quality_counts.items()))}",
        f"- mistake counts: {dict(sorted(mistake_counts.items()))}",
    ]


def _fallback_report(labels: pd.DataFrame) -> list[str]:
    fallback_rows = []
    for row in labels.to_dict("records"):
        provenance = _parse_json(row.get("provenance_json"))
        fallback_mode = provenance.get("fallback_mode")
        if fallback_mode:
            fallback_rows.append(f"{row.get('episode_id')}:{row.get('labeler_name')}:{fallback_mode}")
    if not fallback_rows:
        return ["Fallback/scaffolding labels: none detected"]
    return [
        "Fallback/scaffolding labels detected:",
        *[f"- {item}" for item in fallback_rows[:20]],
        "Do not run the benchmark on this snapshot unless this is a CI/plumbing-only run.",
    ]


def _first_payload(labels: pd.DataFrame, labeler_name: str) -> dict[str, Any]:
    rows = labels[labels["labeler_name"] == labeler_name]
    if rows.empty:
        return {}
    path = Path(str(rows.iloc[0]["label_payload_path"]))
    if not path.exists():
        return {"missing_payload_path": str(path)}
    return _parse_json(path.read_text(encoding="utf-8"))


def _raw_answer(raw_path: str) -> str:
    path = Path(raw_path)
    if not path.exists():
        return ""
    raw = _parse_json(path.read_text(encoding="utf-8"))
    response = raw.get("response_json")
    if isinstance(response, dict):
        answer = response.get("answer")
        if answer:
            return str(answer)
    return str(raw.get("response_text", ""))


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _parse_json(value: Any) -> dict:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact label eyeball-check report.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-episodes", type=int, default=4)
    args = parser.parse_args()
    print(
        inspect_snapshot(
            snapshot_id=args.snapshot,
            data_root=Path(args.data_root) if args.data_root else None,
            max_episodes=args.max_episodes,
        )
    )


if __name__ == "__main__":
    main()
