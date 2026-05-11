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
    st.caption("Mode A snapshot viewer for BridgeData episodes, LEWM annotations, queries, and benchmark artifacts.")

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
    label_map = {
        row["labeler_name"]: row.to_dict()
        for _, row in episode_labels.iterrows()
    }
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
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.subheader("Episodes")
    st.dataframe(
        episodes[["episode_id", "num_steps", "language_instruction", "source_path_video"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Manifest")
    st.json(manifest)


def _render_episode(episode: dict[str, Any], frames: np.ndarray | None, label_map: dict[str, dict]) -> None:
    st.subheader(episode["episode_id"])
    st.write(episode["language_instruction"])
    cols = st.columns([2, 1])
    with cols[0]:
        if frames is None:
            st.warning("No frames.npy available for this episode.")
        else:
            idx = st.slider("Frame", 0, int(frames.shape[0] - 1), int(frames.shape[0] // 2))
            st.image(frames[idx], caption=f"Frame {idx}", use_container_width=True)
    with cols[1]:
        st.metric("Steps", int(episode["num_steps"]))
        st.metric("Labels", len(label_map))
        video_path = Path(episode["source_path_video"])
        if video_path.exists() and video_path.stat().st_size > 128:
            st.video(str(video_path))
        caption = _load_caption(label_map.get("captions"))
        if caption:
            st.subheader("Caption")
            st.write(caption)


def _render_annotations(episode_id: str, frames: np.ndarray | None, label_map: dict[str, dict]) -> None:
    st.subheader(f"Annotations for {episode_id}")
    if not label_map:
        st.warning("No labels for selected episode.")
        return

    frame_idx = 0
    if frames is not None:
        frame_idx = st.slider("Annotation frame", 0, int(frames.shape[0] - 1), int(frames.shape[0] // 2), key="anno_frame")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Mask overlay**")
        image = _mask_overlay(frames, label_map.get("masks"), frame_idx)
        if image is not None:
            st.image(image, use_container_width=True)
        else:
            st.info("No mask payload available.")
    with cols[1]:
        st.markdown("**Depth**")
        image = _depth_image(label_map.get("depth"), frame_idx)
        if image is not None:
            st.image(image, use_container_width=True)
        else:
            st.info("No depth payload available.")
    with cols[2]:
        st.markdown("**Tracks**")
        fig = _track_figure(frames, label_map.get("tracks"), frame_idx)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No track payload available.")

    st.subheader("Label Rows")
    rows = []
    for name, row in sorted(label_map.items()):
        provenance = json.loads(row["provenance_json"])
        rows.append(
            {
                "labeler": name,
                "confidence": row["confidence"],
                "payload": row["label_payload_path"],
                "seconds": provenance.get("wall_clock_seconds"),
                "version": row["labeler_version"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Provenance JSON"):
        for name, row in sorted(label_map.items()):
            st.markdown(f"**{name}**")
            st.json(json.loads(row["provenance_json"]))


def _render_queries(snapshot_id: str, root: Path) -> None:
    from bridgeengine.query import demo_queries, run_query

    st.subheader("Pre-canned DuckDB Queries")
    for name, sql in demo_queries().items():
        with st.expander(name, expanded=name == "mask_coverage"):
            st.code(sql, language="sql")
            result = run_query(snapshot_id, sql, data_root=root)
            st.dataframe(result, use_container_width=True, hide_index=True)


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
        st.dataframe(results, use_container_width=True, hide_index=True)
    with cols[1]:
        if png_path.exists():
            st.image(str(png_path), use_container_width=True)
    if summary_path.exists():
        st.markdown(summary_path.read_text(encoding="utf-8"))


def _render_system(bridge_root: Path) -> None:
    from bridgeengine.system_check import collect_status

    st.subheader("System Readiness")
    status = collect_status(bridge_root=bridge_root)
    st.dataframe(pd.DataFrame(status), use_container_width=True, hide_index=True)
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

