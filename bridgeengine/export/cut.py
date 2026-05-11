from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bridgeengine.ingest.snapshot import DETERMINISTIC_CREATED_AT_UTC, stable_json
from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.query.duckdb_helpers import connect_snapshot


def export_cut(
    snapshot_id: str,
    filter_sql: str,
    output_path: Path,
    cut_name: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    output_path = Path(output_path)
    cut_path = output_path / cut_name
    cut_path.mkdir(parents=True, exist_ok=True)
    con = connect_snapshot(snapshot_id, root)
    try:
        episode_ids = _resolve_episode_ids(con, filter_sql)
        label_rows = con.execute(
            """
            SELECT episode_id, labeler_name, label_payload_path, labeler_version
            FROM labels
            WHERE episode_id IN (SELECT * FROM selected_episode_ids)
            ORDER BY episode_id, labeler_name
            """
        ).fetchall()
        episode_rows = con.execute(
            """
            SELECT episode_id, source_path_actions, source_path_frames, source_path_meta, source_path_video
            FROM episodes
            WHERE episode_id IN (SELECT * FROM selected_episode_ids)
            ORDER BY episode_id
            """
        ).fetchall()
    finally:
        con.close()

    label_paths: dict[str, dict[str, str]] = {episode_id: {} for episode_id in episode_ids}
    labeler_versions: dict[str, str] = {}
    for episode_id, labeler_name, path, version in label_rows:
        label_paths[episode_id][labeler_name] = path
        labeler_versions[labeler_name] = version
    episode_sources = {
        row[0]: {
            "actions": row[1],
            "frames": row[2],
            "metadata": row[3],
            "video": row[4],
        }
        for row in episode_rows
    }

    transform_payload = {
        "cut_name": cut_name,
        "snapshot_id": snapshot_id,
        "filter_sql": filter_sql,
        "episode_ids": episode_ids,
        "label_paths": label_paths,
    }
    transform_hash = "sha256:" + hashlib.sha256(stable_json(transform_payload).encode("utf-8")).hexdigest()
    manifest = {
        "cut_name": cut_name,
        "snapshot_id": snapshot_id,
        "filter_sql": filter_sql,
        "episode_count": len(episode_ids),
        "labeler_versions": dict(sorted(labeler_versions.items())),
        "created_at_utc": DETERMINISTIC_CREATED_AT_UTC,
        "transform_hash": transform_hash,
    }

    _write_text(cut_path / "episode_list.txt", "\n".join(episode_ids) + "\n")
    _write_json(cut_path / "label_paths.json", label_paths)
    _write_json(cut_path / "episode_sources.json", episode_sources)
    _write_json(cut_path / "manifest.json", manifest)
    return manifest


class BridgeCutDataset:
    def __init__(self, cut_path: Path, transform=None):
        self.cut_path = Path(cut_path)
        self.manifest = _read_json(self.cut_path / "manifest.json")
        self.episodes = [
            line.strip()
            for line in (self.cut_path / "episode_list.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.label_paths = _read_json(self.cut_path / "label_paths.json")
        self.episode_sources = _read_json(self.cut_path / "episode_sources.json")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_id = self.episodes[idx]
        sources = self.episode_sources[episode_id]
        sample = {
            "episode_id": episode_id,
            "actions": np.load(sources["actions"], allow_pickle=False),
            "labels": self.label_paths[episode_id],
            "sources": sources,
        }
        frames_path = Path(sources["frames"])
        if frames_path.exists():
            sample["frames_shape"] = tuple(np.load(frames_path, mmap_mode="r").shape)
        if self.transform is not None:
            return self.transform(sample)
        return sample


def _resolve_episode_ids(con, filter_sql: str) -> list[str]:
    where = (filter_sql or "TRUE").strip()
    if where.upper().startswith("SELECT"):
        sql = where
    else:
        sql = f"""
            SELECT DISTINCT e.episode_id
            FROM episodes e
            LEFT JOIN labels l ON l.episode_id = e.episode_id
            WHERE {where}
            ORDER BY e.episode_id
        """
    episode_ids = [row[0] for row in con.execute(sql).fetchall()]
    con.execute("CREATE TEMP TABLE selected_episode_ids(episode_id VARCHAR)")
    if episode_ids:
        con.executemany("INSERT INTO selected_episode_ids VALUES (?)", [(x,) for x in episode_ids])
    return episode_ids


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export a deterministic BridgeEngine training cut.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--filter-sql", default="TRUE")
    parser.add_argument("--output-path", default="training_cuts")
    parser.add_argument("--cut-name", default="cut_mode_a_all_labels")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    manifest = export_cut(
        snapshot_id=args.snapshot,
        filter_sql=args.filter_sql,
        output_path=Path(args.output_path),
        cut_name=args.cut_name,
        data_root=Path(args.data_root) if args.data_root else None,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
