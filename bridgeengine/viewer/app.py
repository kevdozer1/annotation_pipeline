from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from bridgeengine.paths import data_root as resolve_data_root


st.set_page_config(page_title="BridgeEngine", layout="wide")


def main() -> None:
    st.title("BridgeEngine")
    st.caption("Mode A snapshot viewer for BridgeData episodes, pi0.7-style annotations, queries, and benchmark artifacts.")

    root = Path(st.sidebar.text_input("BridgeEngine data root", str(resolve_data_root()))).expanduser()
    bridge_root = Path(st.sidebar.text_input("BridgeData root", "D:/bridgedata_v2_subset")).expanduser()
    snapshots = _list_snapshots(root)
    if not snapshots:
        st.error(f"No snapshots found under {root / 'snapshots'}. Run scripts/poc_quickstart.ps1 first.")
        return

    snapshot_id = st.sidebar.selectbox("Snapshot", snapshots, index=len(snapshots) - 1)
    snapshot_path = root / "snapshots" / snapshot_id
    manifest = _read_json(snapshot_path / "manifest.json")
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet").sort_values("episode_id")
    labels = pd.read_parquet(snapshot_path / "labels.parquet").sort_values(["episode_id", "labeler_name"])
    steps = pd.read_parquet(snapshot_path / "steps.parquet")

    st.sidebar.metric("Episodes", len(episodes))
    st.sidebar.metric("Labels", len(labels))
    st.sidebar.metric("Steps", len(steps))

    episode_options = [
        f"{row.episode_id} - {row.language_instruction}" for row in episodes.itertuples(index=False)
    ]
    selected = st.sidebar.selectbox("Episode", episode_options)
    episode_id = selected.split(" - ", 1)[0]
    episode = episodes.loc[episodes["episode_id"] == episode_id].iloc[0].to_dict()
    episode_labels = labels.loc[labels["episode_id"] == episode_id]
    label_map: dict[str, Any] = {}
    for _, row in episode_labels.iterrows():
        item = row.to_dict()
        name = item["labeler_name"]
        if name in label_map:
            if not isinstance(label_map[name], list):
                label_map[name] = [label_map[name]]
            label_map[name].append(item)
        else:
            label_map[name] = item
    episode_path = Path(episode["source_path_meta"]).parent
    frames = _load_frames(episode_path)

    overview_tab, frame_tab, anno_tab, query_tab, bench_tab, system_tab = st.tabs(
        ["Overview", "Episode", "Annotations", "Queries", "Benchmark", "System"]
    )

    with overview_tab:
        _render_overview(manifest, episodes, labels, bridge_root)
    with frame_tab:
        _render_episode(episode, frames, label_map)
    with anno_tab:
        _render_annotations(episode_id, frames, label_map)
    with query_tab:
        _render_queries(snapshot_id, root)
    with bench_tab:
        _render_benchmark()
    with system_tab:
        _render_system(bridge_root)


def _render_overview(manifest: dict[str, Any], episodes: pd.DataFrame, labels: pd.DataFrame, bridge_root: Path) -> None:
    cols = st.columns(4)
    cols[0].metric("Snapshot", manifest["snapshot_id"])
    cols[1].metric("Episodes", len(episodes))
    cols[2].metric("Label rows", len(labels))
    cols[3].metric("BridgeData root", "present" if bridge_root.exists() else "missing")

    st.subheader("Label Coverage")
    coverage = (
        labels.groupby("labeler_name", as_index=False)
        .agg(rows=("episode_id", "count"), avg_confidence=("confidence", "mean"))
        .sort_values("labeler_name")
    )
    st.dataframe(coverage, width="stretch", hide_index=True)

    st.subheader("Episodes")
    st.dataframe(
        episodes[["episode_id", "num_steps", "language_instruction", "source_path_video"]],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Manifest")
    st.json(manifest)

    st.subheader("Generated Figures")
    figure_paths = [
        Path("figures/quality_summary.png"),
        Path("figures/snapshot_overview.png"),
        Path("figures/benchmark_placeholder.png"),
    ]
    existing = [path for path in figure_paths if path.exists()]
    if existing:
        cols = st.columns(min(3, len(existing)))
        for idx, path in enumerate(existing):
            with cols[idx % len(cols)]:
                st.image(str(path), caption=path.name, width="stretch")
    else:
        st.info("No figures found. Run `python -m bridgeengine.figures --snapshot <snapshot>`.")


def _render_episode(episode: dict[str, Any], frames: np.ndarray | None, label_map: dict[str, dict]) -> None:
    st.subheader(episode["episode_id"])
    st.write(episode["language_instruction"])
    cols = st.columns([2, 1])
    with cols[0]:
        if frames is None:
            st.warning("No frames.npy available for this episode.")
        else:
            idx = st.slider("Frame", 0, int(frames.shape[0] - 1), int(frames.shape[0] // 2))
            st.image(frames[idx], caption=f"Frame {idx}", width="stretch")
    with cols[1]:
        st.metric("Steps", int(episode["num_steps"]))
        st.metric("Labels", len(label_map))
        video_path = Path(episode["source_path_video"])
        if video_path.exists() and video_path.stat().st_size > 128:
            st.video(str(video_path))
        prompt = _load_prompt_preview(episode, label_map)
        if prompt:
            st.subheader("pi0.7 Prompt Preview")
            st.code(prompt, language="text")


def _render_annotations(episode_id: str, frames: np.ndarray | None, label_map: dict[str, dict]) -> None:
    st.subheader(f"Annotations for {episode_id}")
    if not label_map:
        st.warning("No labels for selected episode.")
        return

    frame_idx = 0
    if frames is not None:
        frame_idx = st.slider("Annotation frame", 0, int(frames.shape[0] - 1), int(frames.shape[0] // 2), key="anno_frame")

    rich_cols = st.columns([1.2, 1])
    with rich_cols[0]:
        st.markdown("**Subtask Segments**")
        segments = _load_segments(label_map.get("subtask_segmenter"))
        if segments:
            st.dataframe(pd.DataFrame(segments), width="stretch", hide_index=True)
        else:
            st.info("No subtask segment payload available.")
    with rich_cols[1]:
        st.markdown("**Episode Metadata**")
        metadata = _load_metadata(label_map.get("episode_metadata"))
        if metadata:
            st.json(metadata)
        else:
            st.info("No episode metadata payload available.")

    st.markdown("**Subgoal Images**")
    subgoals = _subgoal_rows(label_map)
    if subgoals:
        cols = st.columns(min(4, len(subgoals)))
        for i, row in enumerate(subgoals):
            with cols[i % len(cols)]:
                path = Path(row["subgoal_image_path"])
                if path.exists():
                    st.image(str(path), caption=f"segment {row['segment_idx']}", width="stretch")
    else:
        st.info("No subgoal image payloads available.")

    with st.expander("Perception comparison artifacts"):
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Mask overlay**")
            image = _mask_overlay(frames, label_map.get("perceptive_masks") or label_map.get("masks"), frame_idx)
            if image is not None:
                st.image(image, width="stretch")
            else:
                st.info("No mask payload available.")
        with cols[1]:
            st.markdown("**Depth**")
            image = _depth_image(label_map.get("perceptive_depth") or label_map.get("depth"), frame_idx)
            if image is not None:
                st.image(image, width="stretch")
            else:
                st.info("No depth payload available.")
        with cols[2]:
            st.markdown("**Tracks**")
            fig = _track_figure(frames, label_map.get("perceptive_tracks") or label_map.get("tracks"), frame_idx)
            if fig is not None:
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            else:
                st.info("No track payload available.")

    st.subheader("Label Rows")
    rows = []
    for name, row_or_rows in sorted(label_map.items()):
        row_list = row_or_rows if isinstance(row_or_rows, list) else [row_or_rows]
        for row in row_list:
            provenance = json.loads(row["provenance_json"])
            rows.append(
                {
                    "labeler": name,
                    "segment_idx": row.get("segment_idx"),
                    "confidence": row["confidence"],
                    "payload": row["label_payload_path"],
                    "subgoal": row.get("subgoal_image_path"),
                    "seconds": provenance.get("wall_clock_seconds"),
                    "version": row["labeler_version"],
                }
            )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Provenance JSON"):
        for name, row_or_rows in sorted(label_map.items()):
            row_list = row_or_rows if isinstance(row_or_rows, list) else [row_or_rows]
            for row in row_list:
                suffix = f" segment {row.get('segment_idx')}" if row.get("segment_idx") is not None else ""
                st.markdown(f"**{name}{suffix}**")
                st.json(json.loads(row["provenance_json"]))


def _render_queries(snapshot_id: str, root: Path) -> None:
    from bridgeengine.query import demo_queries, run_query

    st.subheader("Pre-canned DuckDB Queries")
    for name, sql in demo_queries().items():
        with st.expander(name, expanded=name == "mask_coverage"):
            st.code(sql, language="sql")
            result = run_query(snapshot_id, sql, data_root=root)
            st.dataframe(result, width="stretch", hide_index=True)


def _render_benchmark() -> None:
    st.subheader("Benchmark")
    csv_path = Path("bench_results/bench_results.csv")
    png_path = Path("bench_results/bench_bar.png")
    summary_path = Path("bench_results/bench_summary.md")
    if not csv_path.exists():
        st.warning("No benchmark CSV found. Run `python -m bridgeengine.benchmark.run_grid --snapshot <snapshot>`.")
        return
    results = pd.read_csv(csv_path)
    cols = st.columns([1, 1])
    with cols[0]:
        st.dataframe(results, width="stretch", hide_index=True)
    with cols[1]:
        if png_path.exists():
            st.image(str(png_path), width="stretch")
    if summary_path.exists():
        st.markdown(summary_path.read_text(encoding="utf-8"))


def _render_system(bridge_root: Path) -> None:
    from bridgeengine.system_check import collect_status

    st.subheader("System Readiness")
    status = collect_status(bridge_root=bridge_root)
    st.dataframe(pd.DataFrame(status), width="stretch", hide_index=True)
    st.caption("This is a local readiness check. It does not import heavy model packages unless they are already installed.")


def _list_snapshots(root: Path) -> list[str]:
    snap_root = root / "snapshots"
    if not snap_root.exists():
        return []
    return sorted(p.name for p in snap_root.iterdir() if (p / "manifest.json").exists())


@st.cache_data(show_spinner=False)
def _load_frames(episode_path: Path) -> np.ndarray | None:
    frames_path = episode_path / "frames.npy"
    if not frames_path.exists():
        return None
    return np.load(frames_path, allow_pickle=False)


def _load_caption(row: dict | None) -> str | None:
    if not row:
        return None
    path = Path(row["label_payload_path"])
    if not path.exists():
        return None
    return _read_json(path).get("caption")


def _load_segments(row: dict | list | None) -> list[dict]:
    if isinstance(row, list):
        row = row[0] if row else None
    if not row:
        return []
    path = Path(row["label_payload_path"])
    if not path.exists():
        return []
    return _read_json(path).get("segments", [])


def _load_metadata(row: dict | list | None) -> dict | None:
    if isinstance(row, list):
        row = row[0] if row else None
    if not row:
        return None
    if row.get("metadata_payload_json"):
        return json.loads(row["metadata_payload_json"])
    path = Path(row["label_payload_path"])
    if path.exists():
        return _read_json(path).get("metadata")
    return None


def _load_prompt_preview(episode: dict[str, Any], label_map: dict[str, Any]) -> str | None:
    segments = _load_segments(label_map.get("subtask_segmenter"))
    metadata = _load_metadata(label_map.get("episode_metadata"))
    if not segments:
        return None
    subtask = segments[0]["subtask_text"]
    prompt = f"Task: {episode['language_instruction']}. Subtask: {subtask}."
    if metadata:
        prompt += (
            f" Speed: {metadata['speed']}. Quality: {metadata['quality']}/5."
            f" Mistake: {str(metadata['mistake']).lower()}."
            f" Control Mode: {metadata['control_mode']}."
        )
    return prompt


def _subgoal_rows(label_map: dict[str, Any]) -> list[dict]:
    rows = label_map.get("subgoal_images")
    if not rows:
        return []
    if not isinstance(rows, list):
        rows = [rows]
    return sorted(rows, key=lambda x: x.get("segment_idx") if x.get("segment_idx") is not None else -1)


def _mask_overlay(frames: np.ndarray | None, row: dict | None, idx: int) -> Image.Image | None:
    if frames is None or not row:
        return None
    path = Path(row["label_payload_path"])
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        masks = data["tight"]
    idx = min(idx, masks.shape[0] - 1, frames.shape[0] - 1)
    frame = frames[idx].astype(np.float32)
    mask = masks[idx].astype(bool)
    overlay = frame.copy()
    overlay[mask] = overlay[mask] * 0.45 + np.array([30, 220, 80]) * 0.55
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _depth_image(row: dict | None, idx: int) -> Image.Image | None:
    if not row:
        return None
    path = Path(row["label_payload_path"])
    if not path.exists():
        return None
    depth = np.load(path, mmap_mode="r")
    idx = min(idx, depth.shape[0] - 1)
    image = np.asarray(depth[idx], dtype=np.float32)
    image = image - float(np.nanmin(image))
    denom = float(np.nanmax(image)) or 1.0
    image = (image / denom * 255.0).astype(np.uint8)
    return Image.fromarray(image)


def _track_figure(frames: np.ndarray | None, row: dict | None, idx: int):
    if not row:
        return None
    path = Path(row["label_payload_path"])
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        tracks = data["tracks"]
        visibility = data["visibility"]
    idx = min(idx, tracks.shape[0] - 1)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if frames is not None:
        ax.imshow(frames[min(idx, frames.shape[0] - 1)])
    pts = tracks[idx]
    vis = visibility[idx].astype(bool)
    stride = max(1, int(len(pts) / 120))
    pts = pts[vis][::stride]
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c="#39FF88", alpha=0.85)
    ax.set_xlim(0, frames.shape[2] if frames is not None else np.nanmax(tracks[..., 0]))
    ax.set_ylim(frames.shape[1] if frames is not None else np.nanmax(tracks[..., 1]), 0)
    ax.axis("off")
    return fig


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
