from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bridgeengine.calibration import default_gold_path
from bridgeengine.paths import data_root as resolve_data_root


DERIVED_SUBGOAL_SOURCE = "derived_from_gold_subtask_end_step"


def derive_gold_subgoals_from_boundaries(
    snapshot_id: str,
    gold_file: str | Path | None = None,
    data_root: str | Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Fill gold subgoal frame indices from reviewed subtask end boundaries.

    In the current POC, subgoal images are deterministic edge frames: the frame at
    the end of each subtask segment. If a reviewer corrects the subtask boundary,
    the corresponding gold subgoal should move with that boundary. This command
    makes that provenance explicit instead of treating old auto subgoal frames as
    independently reviewed human labels.
    """

    root = resolve_data_root(data_root)
    path = Path(gold_file) if gold_file else default_gold_path(snapshot_id, root)
    if not path.exists():
        raise FileNotFoundError(f"Gold file not found: {path}")

    gold = _read_json(path)
    updated_episodes = 0
    updated_subgoals = 0
    already_matching = 0
    changed_from_auto = 0
    skipped = 0

    for entry in gold.get("episodes", []):
        num_steps = _safe_int(entry.get("num_steps"))
        auto_subtasks = _by_idx(entry.get("auto", {}).get("subtasks", []))
        auto_subgoals = _by_idx(entry.get("auto", {}).get("subgoals", []))
        gold_subtasks = _by_idx(entry.get("gold", {}).get("subtasks", []))
        subgoals = entry.setdefault("gold", {}).setdefault("subgoals", [])
        subgoal_by_idx = _by_idx(subgoals)
        episode_changed = False

        for idx, subtask in sorted(gold_subtasks.items()):
            frame_idx = _derived_frame_idx(subtask, auto_subtasks.get(idx, {}), num_steps)
            if frame_idx is None:
                skipped += 1
                continue

            subgoal = subgoal_by_idx.get(idx)
            if subgoal is None:
                subgoal = {"segment_idx": idx, "frame_idx": None, "accept_auto": None}
                subgoals.append(subgoal)
                subgoal_by_idx[idx] = subgoal

            old_frame = _safe_int(subgoal.get("frame_idx"))
            auto_frame = _safe_int(auto_subgoals.get(idx, {}).get("frame_idx"))
            if old_frame == frame_idx and subgoal.get("source") == DERIVED_SUBGOAL_SOURCE:
                continue
            if old_frame is not None and not overwrite:
                skipped += 1
                continue

            subgoal["frame_idx"] = int(frame_idx)
            subgoal["accept_auto"] = bool(auto_frame == frame_idx)
            subgoal["source"] = DERIVED_SUBGOAL_SOURCE
            subgoal["derived_from_segment_idx"] = idx
            subgoal["derived_from_gold_end_step"] = int(frame_idx)
            if auto_frame == frame_idx:
                already_matching += 1
            else:
                changed_from_auto += 1
            updated_subgoals += 1
            episode_changed = True

        if episode_changed:
            updated_episodes += 1

    _write_json(path, gold)
    return {
        "snapshot_id": snapshot_id,
        "gold_file": str(path.resolve()),
        "updated_episodes": updated_episodes,
        "updated_subgoals": updated_subgoals,
        "already_matching_auto_subgoals": already_matching,
        "changed_from_auto_subgoals": changed_from_auto,
        "skipped_subtasks": skipped,
        "source": DERIVED_SUBGOAL_SOURCE,
    }


def _derived_frame_idx(subtask: dict[str, Any], auto_subtask: dict[str, Any], num_steps: int | None) -> int | None:
    end_step = _safe_int(subtask.get("end_step"))
    if end_step is None and subtask.get("accept_auto"):
        end_step = _safe_int(auto_subtask.get("end_step"))
    if end_step is None:
        return None
    if num_steps is not None and num_steps > 0:
        return max(0, min(end_step, num_steps - 1))
    return max(0, end_step)


def _by_idx(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in items:
        idx = _safe_int(item.get("segment_idx"))
        if idx is not None:
            result[idx] = item
    return result


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive gold subgoal frame indices from reviewed subtask end boundaries.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--gold-file", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    report = derive_gold_subgoals_from_boundaries(
        snapshot_id=args.snapshot,
        gold_file=Path(args.gold_file) if args.gold_file else None,
        data_root=Path(args.data_root) if args.data_root else None,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
