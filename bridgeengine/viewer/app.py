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

    overview_tab, frame_tab, anno_tab, calibration_tab, value_tab, query_tab, bench_tab, system_tab = st.tabs(
        ["Overview", "Episode", "Annotations", "Calibration", "Value", "Queries", "Benchmark", "System"]
    )

    with overview_tab:
        _render_overview(manifest, episodes, labels, bridge_root)
    with frame_tab:
        _render_episode(episode, frames, label_map)
    with anno_tab:
        _render_annotations(episode_id, frames, label_map)
    with calibration_tab:
        _render_calibration(snapshot_id, snapshot_path, episodes, labels, root, episode_id)
    with value_tab:
        _render_value(snapshot_path, episodes)
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
        Path("figures/scale_curve_gemini_100.png"),
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


def _render_calibration(
    snapshot_id: str,
    snapshot_path: Path,
    episodes: pd.DataFrame,
    labels: pd.DataFrame,
    root: Path,
    current_episode_id: str,
) -> None:
    from bridgeengine.calibration import (
        calibration_reliability,
        default_gold_path,
        load_or_create_calibration_gold,
        review_summary,
        update_episode_review,
    )

    st.subheader("Human Score Calibration")
    st.caption("Review each episode video and save your calibrated curation score. Reviews are written as a gold-set JSON, so reliability reports work immediately.")

    default_path = default_gold_path(snapshot_id, root)
    gold_path = Path(st.text_input("Calibration gold file", str(default_path), key="calibration_gold_path")).expanduser()
    gold = load_or_create_calibration_gold(snapshot_id, gold_path, data_root=root)
    summary = review_summary(snapshot_id, gold_path, data_root=root)
    reviewed = int(summary["reviewed"].sum()) if not summary.empty else 0
    total = int(len(summary))
    remaining = total - reviewed
    report = calibration_reliability(snapshot_id, gold_path, data_root=root)

    metrics = st.columns(5)
    metrics[0].metric("Reviewed", f"{reviewed}/{total}")
    metrics[1].metric("Remaining", remaining)
    metrics[2].metric("Quality exact", _fmt_metric(report.get("quality_exact_agreement")))
    metrics[3].metric("Within one", _fmt_metric(report.get("quality_within_one_agreement")))
    metrics[4].metric("Boundary IoU", _fmt_metric(report.get("subtask_boundary_temporal_iou_mean")))

    episode_ids = episodes["episode_id"].astype(str).tolist()
    if not episode_ids:
        st.warning("No episodes available for calibration.")
        return
    if "calibration_episode_id" not in st.session_state or st.session_state["calibration_episode_id"] not in episode_ids:
        st.session_state["calibration_episode_id"] = current_episode_id if current_episode_id in episode_ids else episode_ids[0]

    toolbar = st.columns([1, 1, 2, 2])
    with toolbar[0]:
        if st.button("Previous", width="stretch"):
            idx = episode_ids.index(st.session_state["calibration_episode_id"])
            st.session_state["calibration_episode_id"] = episode_ids[(idx - 1) % len(episode_ids)]
            st.rerun()
    with toolbar[1]:
        if st.button("Next", width="stretch"):
            idx = episode_ids.index(st.session_state["calibration_episode_id"])
            st.session_state["calibration_episode_id"] = episode_ids[(idx + 1) % len(episode_ids)]
            st.rerun()
    with toolbar[2]:
        if st.button("Next unreviewed", width="stretch"):
            next_id = _next_unreviewed(summary, episode_ids, st.session_state["calibration_episode_id"])
            if next_id:
                st.session_state["calibration_episode_id"] = next_id
                st.rerun()
    with toolbar[3]:
        if st.button("Refresh gold summary", width="stretch"):
            st.rerun()

    episode_id = st.selectbox(
        "Review episode",
        episode_ids,
        key="calibration_episode_id",
        format_func=lambda x: _episode_option_label(episodes, summary, x),
    )
    episode = episodes.loc[episodes["episode_id"].astype(str) == episode_id].iloc[0].to_dict()
    label_map = _label_map_for(labels, episode_id)
    metadata = _load_metadata(label_map.get("episode_metadata")) or {}
    segments = _load_segments(label_map.get("subtask_segmenter"))
    subgoals = _subgoal_rows(label_map)
    gold_entry = _gold_entry(gold, episode_id)
    episode_path = Path(episode["source_path_meta"]).parent
    frames = _load_frames(episode_path)
    selected_summary = summary.loc[summary["episode_id"].astype(str) == episode_id]
    selected_summary = selected_summary.iloc[0].to_dict() if not selected_summary.empty else {}

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(f"**{episode_id}**")
        st.write(episode.get("language_instruction", ""))
        video_path = Path(str(episode.get("source_path_video", "")))
        if video_path.exists() and video_path.stat().st_size > 128:
            st.video(str(video_path))
        elif frames is not None:
            idx = st.slider("Frame", 0, int(frames.shape[0] - 1), int(frames.shape[0] // 2), key=f"cal_frame_{episode_id}")
            st.image(frames[idx], caption=f"Frame {idx}", width="stretch")
        else:
            st.warning("No video or frame array found for this episode.")
        fig = _segment_timeline(segments, int(episode.get("num_steps", 0) or 0))
        if fig is not None:
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        if subgoals:
            st.markdown("**Subgoal frames**")
            goal_cols = st.columns(min(4, len(subgoals)))
            for idx, row in enumerate(subgoals):
                with goal_cols[idx % len(goal_cols)]:
                    path = Path(str(row.get("subgoal_image_path", "")))
                    if path.exists():
                        st.image(str(path), caption=f"segment {row.get('segment_idx')}", width="stretch")

    with right:
        auto_score = _safe_int(metadata.get("curation_quality")) or _safe_int(metadata.get("quality")) or 3
        gold_score = _safe_int(selected_summary.get("gold_score")) or auto_score
        auto_mistake = bool(metadata.get("mistake", False))
        gold_mistake = selected_summary.get("gold_mistake")
        mistake_default = auto_mistake if gold_mistake is None or pd.isna(gold_mistake) else bool(gold_mistake)

        score_cols = st.columns(3)
        score_cols[0].metric("Auto score", _score_label(auto_score))
        score_cols[1].metric("Auto keep", "yes" if metadata.get("curation_keep") else "no")
        score_cols[2].metric("Boundary", str(metadata.get("boundary_clarity", "unknown")))
        st.markdown("**Auto metadata**")
        st.json(
            {
                "task_success_quality": metadata.get("task_success_quality"),
                "curation_quality": metadata.get("curation_quality", metadata.get("quality")),
                "curation_keep": metadata.get("curation_keep"),
                "mistake": metadata.get("mistake"),
                "boundary_clarity": metadata.get("boundary_clarity"),
                "structure_score": metadata.get("interaction_structure_score"),
                "scoring_reason": metadata.get("scoring_reason"),
                "vlm_reason": metadata.get("reason"),
            }
        )
        with st.form(f"calibration_form_{episode_id}"):
            calibrated_score = st.radio(
                "Your calibrated score",
                [1, 2, 3, 4, 5],
                index=max(0, min(4, int(gold_score) - 1)),
                format_func=_score_label,
                horizontal=False,
            )
            mistake = st.checkbox("Mistake visible", value=mistake_default)
            accept_auto_metadata = st.checkbox("This matches the auto metadata judgment", value=int(calibrated_score) == int(auto_score) and mistake == auto_mistake)
            accept_auto_subtasks = st.checkbox("Also accept auto subtask boundaries for reliability scoring", value=_all_accept_auto(gold_entry, "subtasks"))
            accept_auto_subgoals = st.checkbox("Also accept auto subgoal frames for reliability scoring", value=_all_accept_auto(gold_entry, "subgoals"))
            reason = st.text_area("Calibration reason", value=str(metadata.get("scoring_reason") or ""), height=90)
            notes = st.text_area("Review notes", value=str(selected_summary.get("notes") or ""), height=90)
            submitted = st.form_submit_button("Save review", width="stretch")
        if submitted:
            update_episode_review(
                snapshot_id,
                episode_id,
                int(calibrated_score),
                mistake=mistake,
                reason=reason,
                review_notes=notes,
                accept_auto_metadata=accept_auto_metadata,
                accept_auto_subtasks=accept_auto_subtasks,
                accept_auto_subgoals=accept_auto_subgoals,
                gold_file=gold_path,
                data_root=root,
            )
            st.success(f"Saved review for {episode_id}")
            st.rerun()

    with st.expander("Review queue", expanded=False):
        st.dataframe(
            summary[
                [
                    "reviewed",
                    "episode_id",
                    "auto_score",
                    "gold_score",
                    "auto_keep",
                    "gold_keep",
                    "boundary_clarity",
                    "task",
                    "notes",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    st.caption(f"Gold file: `{gold_path}`")


def _fmt_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _next_unreviewed(summary: pd.DataFrame, episode_ids: list[str], current_episode_id: str) -> str | None:
    if summary.empty:
        return None
    reviewed = {
        str(row.episode_id)
        for row in summary.itertuples(index=False)
        if bool(row.reviewed)
    }
    if len(reviewed) >= len(episode_ids):
        return None
    start = episode_ids.index(current_episode_id) if current_episode_id in episode_ids else -1
    for offset in range(1, len(episode_ids) + 1):
        candidate = episode_ids[(start + offset) % len(episode_ids)]
        if candidate not in reviewed:
            return candidate
    return None


def _episode_option_label(episodes: pd.DataFrame, summary: pd.DataFrame, episode_id: str) -> str:
    episode = episodes.loc[episodes["episode_id"].astype(str) == episode_id]
    task = str(episode.iloc[0].get("language_instruction", "")) if not episode.empty else ""
    row = summary.loc[summary["episode_id"].astype(str) == episode_id] if not summary.empty else pd.DataFrame()
    if row.empty:
        return f"[ ] {episode_id} - {task}"
    item = row.iloc[0]
    mark = "x" if bool(item.get("reviewed")) else " "
    auto = item.get("auto_score")
    gold = item.get("gold_score")
    score = f"auto {auto}" if pd.isna(gold) else f"auto {auto} -> gold {int(gold)}"
    return f"[{mark}] {episode_id} ({score}) - {task}"


def _label_map_for(labels: pd.DataFrame, episode_id: str) -> dict[str, Any]:
    episode_labels = labels.loc[labels["episode_id"].astype(str) == str(episode_id)]
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
    return label_map


def _segment_timeline(segments: list[dict[str, Any]], num_steps: int):
    if not segments:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 1.8))
    colors = ["#e8752a", "#4c78a8", "#59a14f", "#b279a2", "#edc948", "#76b7b2"]
    max_step = max(num_steps - 1, max(int(s.get("end_step", 0) or 0) for s in segments), 1)
    for idx, segment in enumerate(segments):
        start = int(segment.get("start_step", 0) or 0)
        end = int(segment.get("end_step", start) or start)
        color = colors[idx % len(colors)]
        ax.axvspan(start, end + 1, color=color, alpha=0.72)
        label = str(segment.get("subtask_text", "")).strip()
        if len(label) > 34:
            label = label[:31] + "..."
        ax.text((start + end + 1) / 2, 0.5, label, ha="center", va="center", fontsize=8, color="white", weight="bold", wrap=True)
    ax.set_xlim(0, max_step + 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("episode step")
    ax.set_title("Auto subtask timeline")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _score_label(score: int) -> str:
    labels = {
        1: "1 - clear reject",
        2: "2 - reject",
        3: "3 - near reject",
        4: "4 - near keep",
        5: "5 - clear keep",
    }
    return labels.get(int(score), str(score))


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _gold_entry(gold: dict[str, Any], episode_id: str) -> dict[str, Any]:
    for entry in gold.get("episodes", []):
        if str(entry.get("episode_id")) == str(episode_id):
            return entry
    return {}


def _all_accept_auto(gold_entry: dict[str, Any], key: str) -> bool:
    items = gold_entry.get("gold", {}).get(key, [])
    return bool(items) and all(bool(item.get("accept_auto")) for item in items)


def _render_value(snapshot_path: Path, episodes: pd.DataFrame) -> None:
    st.subheader("Value-Aware Curation")
    if "value_score" not in episodes.columns or episodes["value_score"].isna().all():
        st.info("No value scores yet. Run `python -m bridgeengine.value report --snapshot <snapshot>`.")
        return
    scored = episodes.dropna(subset=["value_score"]).copy()
    scored["value_score"] = scored["value_score"].astype(float)
    cols = st.columns(4)
    cols[0].metric("Scored episodes", len(scored))
    cols[1].metric("Mean score", f"{scored['value_score'].mean():.4f}")
    cols[2].metric("Max score", f"{scored['value_score'].max():.4f}")
    cols[3].metric("Method", str(scored["value_method"].dropna().iloc[0]) if scored["value_method"].notna().any() else "unknown")

    st.markdown("**Top Outliers**")
    display_cols = [
        "value_rank",
        "episode_id",
        "value_score",
        "value_percentile",
        "num_steps",
        "language_instruction",
    ]
    existing = [col for col in display_cols if col in scored.columns]
    st.dataframe(
        scored.sort_values("value_score", ascending=False)[existing].head(20),
        width="stretch",
        hide_index=True,
    )

    report_path = snapshot_path / "value_report.json"
    if report_path.exists():
        with st.expander("Value report JSON"):
            st.json(_read_json(report_path))

    compression_root = snapshot_path / "value_compression"
    reports = sorted(compression_root.glob("*/report.json")) if compression_root.exists() else []
    if reports:
        latest = reports[-1]
        compression = _read_json(latest)
        st.markdown("**Tiered Compression**")
        ccols = st.columns(4)
        ccols[0].metric("Uniform zstd", compression.get("uniform_zstd_size_bytes", 0))
        ccols[1].metric("Tiered", compression.get("tiered_size_bytes", 0))
        ccols[2].metric("Savings vs uniform", f"{compression.get('tiered_vs_uniform_savings_pct', 0.0):.2f}%")
        ccols[3].metric("High-value episodes", len(compression.get("high_value_episode_ids", [])))


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
