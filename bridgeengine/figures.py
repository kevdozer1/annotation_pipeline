from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont

from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.quality_gate import evaluate_snapshot_quality
from bridgeengine.scoring import metadata_quality


SEGMENT_COLORS = ["#247BA0", "#E87D1E", "#2E7D32", "#7B5EA7", "#D9822B"]


def generate_figures(
    snapshot_id: str,
    data_root: str | Path | None = None,
    output_dir: str | Path = "figures",
    compare_snapshot_id: str | None = None,
    include_threshold_diagram: bool = False,
    include_threshold_animation: bool = False,
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
    if include_threshold_diagram:
        paths["threshold_annotation_diagram"] = output / "threshold_annotation_diagram.png"
        _threshold_annotation_diagram(snapshot_path, paths["threshold_annotation_diagram"])
    if include_threshold_animation:
        paths["threshold_annotation_animation"] = output / "threshold_annotation_animation.gif"
        _threshold_annotation_animation(snapshot_path, paths["threshold_annotation_animation"])
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


def _threshold_annotation_diagram(snapshot_path: Path, output_path: Path, quality_threshold: int = 4) -> None:
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    examples = _select_threshold_examples(episodes, labels, quality_threshold)

    fig = plt.figure(figsize=(18, 13.5))
    grid = fig.add_gridspec(
        nrows=4,
        ncols=3,
        width_ratios=[1.25, 3.3, 2.45],
        hspace=0.42,
        wspace=0.12,
    )
    fig.suptitle(
        f"BridgeEngine Annotation Threshold Diagram: {snapshot_path.name}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.958,
        f"Threshold: quality >= {quality_threshold} counts as KEEP for curation. "
        "Each row shows the video storyboard, VLM subtask spans, subgoal endpoints, and compact episode metadata.",
        ha="center",
        fontsize=10,
        color="#303030",
    )

    for row_idx, example in enumerate(examples):
        summary_ax = fig.add_subplot(grid[row_idx, 0])
        video_ax = fig.add_subplot(grid[row_idx, 1])
        annotation_ax = fig.add_subplot(grid[row_idx, 2])
        _draw_threshold_summary(summary_ax, example, quality_threshold)
        _draw_video_storyboard(video_ax, example)
        _draw_annotation_payload(annotation_ax, example)

    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _threshold_annotation_animation(
    snapshot_path: Path,
    output_path: Path,
    quality_threshold: int = 4,
    frame_count: int = 34,
    duration_ms: int = 150,
) -> None:
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    examples = _select_threshold_examples(episodes, labels, quality_threshold)
    for example in examples:
        example["_frames"] = _load_frames(Path(str(example["episode"].get("source_path_frames", ""))))

    frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        progress = frame_idx / max(frame_count - 1, 1)
        frames.append(_render_threshold_animation_frame(examples, progress, snapshot_path.name, quality_threshold))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _render_threshold_animation_frame(
    examples: list[dict[str, Any]],
    progress: float,
    snapshot_name: str,
    quality_threshold: int,
) -> Image.Image:
    width, height = 1500, 980
    margin = 18
    header_h = 82
    row_h = 214
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    fonts = _animation_fonts()

    draw.text((width // 2, 18), "BridgeEngine Animated Annotation Threshold View", anchor="ma", font=fonts["title"], fill="#202124")
    draw.text(
        (width // 2, 48),
        f"{snapshot_name} | quality >= {quality_threshold} is KEEP | scoring favors clear, unoccluded cause-effect boundaries; anomaly tracks unusual structure",
        anchor="ma",
        font=fonts["small"],
        fill="#4A4A4A",
    )

    for row_idx, example in enumerate(examples):
        y = header_h + row_idx * row_h
        _draw_animation_row(draw, image, example, progress, y, row_h, quality_threshold, fonts)

    return image


def _draw_animation_row(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    example: dict[str, Any],
    progress: float,
    y: int,
    row_h: int,
    quality_threshold: int,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    color = ImageColor.getrgb(example["color"])
    pale = _blend(color, (255, 255, 255), 0.86)
    quality = int(example["quality"])
    status = "KEEP" if quality >= quality_threshold else "REJECT"
    metadata = example["metadata"]
    frames = example.get("_frames")
    num_steps = int(example["episode"].get("num_steps", len(frames) if frames is not None else 0) or 0)
    current_step = int(round(progress * max(num_steps - 1, 0)))
    active_segment = _active_segment_for_step(example["segments"], current_step)
    active_idx = _safe_int(active_segment.get("segment_idx"))
    active_color_hex = _segment_color(active_idx if active_idx is not None else 0)
    active_color = ImageColor.getrgb(active_color_hex)
    active_pale = _blend(active_color, (255, 255, 255), 0.84)

    x_card, card_w = 18, 260
    x_video, video_w, video_h = 300, 235, 176
    x_mid, mid_w = 560, 450
    x_meta, meta_w = 1035, 445

    draw.rounded_rectangle((x_card, y + 16, x_card + card_w, y + row_h - 16), radius=12, outline=color, width=3, fill="#FFFFFF")
    draw.text((x_card + 16, y + 31), example["category"].upper(), font=fonts["small_bold"], fill=color)
    draw.text((x_card + 16, y + 59), f"Quality {quality}/5", font=fonts["huge"], fill="#202124")
    draw.text((x_card + 16, y + 103), f"Result: {status}", font=fonts["body_bold"], fill=color)
    draw.text((x_card + 16, y + 129), f"Mistake: {str(bool(metadata.get('mistake'))).lower()}", font=fonts["body"], fill="#303030")
    draw.text((x_card + 16, y + 153), f"Step: {current_step}/{max(num_steps - 1, 0)}", font=fonts["body"], fill="#303030")
    draw.text((x_card + 16, y + 180), example["episode_id"], font=fonts["small"], fill="#5B6770")

    frame = _animation_frame_image(frames, current_step, (video_w, video_h))
    canvas.paste(frame, (x_video, y + 20))
    draw.rectangle((x_video, y + 20, x_video + video_w, y + 20 + video_h), outline="#FFFFFF", width=2)
    draw.rectangle((x_video - 1, y + 19, x_video + video_w + 1, y + 21 + video_h), outline=color, width=2)
    draw.text((x_video, y + 202), "current video frame", font=fonts["small"], fill="#303030")

    task = str(example["episode"].get("language_instruction", example["episode_id"]))
    draw.text((x_mid, y + 18), "Task", font=fonts["body_bold"], fill="#202124")
    _draw_wrapped(draw, task, (x_mid, y + 43), mid_w, fonts["small"], "#303030", max_lines=2)

    timeline_y = y + 98
    draw.text((x_mid, timeline_y - 36), "Subtask timeline", font=fonts["body_bold"], fill="#202124")
    _draw_animation_timeline(draw, example, x_mid, timeline_y, mid_w, 34, num_steps, current_step, active_idx)

    active_text = str(active_segment.get("subtask_text", "")).strip() or "No active subtask"
    active_box = (x_mid, y + 140, x_mid + mid_w, y + row_h - 4)
    draw.rounded_rectangle(active_box, radius=10, outline=active_color, width=2, fill=active_pale)
    draw.text(
        (x_mid + 12, y + 151),
        f"Active subtask {active_idx if active_idx is not None else '-'}",
        font=fonts["small_bold"],
        fill=active_color,
    )
    _draw_wrapped(draw, active_text, (x_mid + 12, y + 174), mid_w - 24, fonts["tiny"], "#202124", max_lines=2)

    draw.rounded_rectangle((x_meta, y + 18, x_meta + meta_w, y + row_h - 5), radius=10, outline="#CBD3D8", width=2, fill="#F7F9FA")
    draw.text((x_meta + 12, y + 33), "Annotation payload", font=fonts["body_bold"], fill="#202124")
    payload_y = y + 61
    for segment in example["segments"][:4]:
        idx = _safe_int(segment.get("segment_idx"))
        start = _safe_int(segment.get("start_step"))
        end = _safe_int(segment.get("end_step"))
        segment_text = str(segment.get("subtask_text", "")).strip()
        draw.text((x_meta + 12, payload_y), f"{idx}. {start}-{end}", font=fonts["tiny_bold"], fill=ImageColor.getrgb(_segment_color(idx)))
        _draw_wrapped(draw, segment_text, (x_meta + 72, payload_y), meta_w - 88, fonts["tiny"], "#303030", max_lines=1)
        payload_y += 22
    meta = (
        f"quality={metadata_quality(metadata)}  "
        f"task_success={metadata.get('task_success_quality', metadata.get('quality'))}  "
        f"mistake={str(bool(metadata.get('mistake'))).lower()}  speed={metadata.get('speed')}"
    )
    draw.text((x_meta + 12, y + 147), "Metadata", font=fonts["tiny_bold"], fill="#202124")
    _draw_wrapped(draw, meta, (x_meta + 12, y + 169), meta_w - 24, fonts["tiny"], "#303030", max_lines=1)
    _draw_wrapped(draw, f"Decision cue: {_decision_cue(example)}", (x_meta + 12, y + 188), meta_w - 24, fonts["tiny"], "#303030", max_lines=1)


def _draw_animation_timeline(
    draw: ImageDraw.ImageDraw,
    example: dict[str, Any],
    x: int,
    y: int,
    width: int,
    height: int,
    num_steps: int,
    current_step: int,
    active_idx: int | None,
) -> None:
    draw.rounded_rectangle((x, y, x + width, y + height), radius=7, fill="#ECEFF1", outline="#CBD3D8", width=1)
    for i, segment in enumerate(example["segments"]):
        start = max(0, _safe_int(segment.get("start_step")) or 0)
        end = max(start, _safe_int(segment.get("end_step")) or start)
        sx = x + int(start / max(num_steps - 1, 1) * width)
        ex = x + int(min(1.0, (end + 1) / max(num_steps, 1)) * width)
        fill = ImageColor.getrgb(_segment_color(i))
        outline = "#202124" if _safe_int(segment.get("segment_idx")) == active_idx else "#FFFFFF"
        draw.rectangle((sx, y, max(ex, sx + 4), y + height), fill=fill, outline=outline, width=2)
        _draw_centered_text(draw, str(i), (sx, y, max(ex, sx + 4), y + height), _animation_fonts()["tiny_bold"], "#FFFFFF")
        marker_x = max(ex, sx + 4)
        draw.polygon(
            [(marker_x - 6, y - 2), (marker_x + 6, y - 2), (marker_x, y - 14)],
            fill="#202124",
        )
    playhead_x = x + int(current_step / max(num_steps - 1, 1) * width)
    draw.line((playhead_x, y - 16, playhead_x, y + height + 16), fill="#C62828", width=4)
    draw.ellipse((playhead_x - 6, y + height + 7, playhead_x + 6, y + height + 19), fill="#C62828")


def _animation_frame_image(frames: np.ndarray | None, step: int, size: tuple[int, int]) -> Image.Image:
    if frames is None or len(frames) == 0:
        return Image.new("RGB", size, "#E0E0E0")
    arr = np.asarray(frames[min(max(step, 0), len(frames) - 1)])
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", size, "#111111")
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    background.paste(image, offset)
    return background


def _active_segment_for_step(segments: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for segment in segments:
        start = _safe_int(segment.get("start_step"))
        end = _safe_int(segment.get("end_step"))
        if start is not None and end is not None and start <= step <= end:
            return segment
    return segments[-1] if segments else {"segment_idx": None, "subtask_text": "", "start_step": 0, "end_step": 0}


def _animation_fonts() -> dict[str, ImageFont.ImageFont]:
    return {
        "title": _pil_font(28, bold=True),
        "huge": _pil_font(34, bold=True),
        "body": _pil_font(20),
        "body_bold": _pil_font(20, bold=True),
        "small": _pil_font(16),
        "small_bold": _pil_font(16, bold=True),
        "tiny": _pil_font(14),
        "tiny_bold": _pil_font(14, bold=True),
    }


def _pil_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width_px: int,
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int],
    max_lines: int,
) -> None:
    avg_char = max(6, int(draw.textlength("abcdefghijklmnopqrstuvwxyz", font=font) / 26))
    width_chars = max(12, int(width_px / avg_char))
    lines = textwrap.wrap(str(text), width=width_chars, break_long_words=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    x, y = xy
    line_h = int(font.size * 1.12) if hasattr(font, "size") else 16
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - text_w) / 2, y0 + (y1 - y0 - text_h) / 2 - 1), text, font=font, fill=fill)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(a[i] * (1.0 - amount) + b[i] * amount) for i in range(3))


def _segment_color(index: int | None) -> str:
    if index is None:
        return SEGMENT_COLORS[0]
    return SEGMENT_COLORS[int(index) % len(SEGMENT_COLORS)]


def _quality_note(reason: str) -> str:
    reason = " ".join(str(reason).split())
    markers = [
        "The only limitation is ",
        "The only imperfection is ",
        "The only minor issue is ",
        "but ",
        "though ",
        "so ",
    ]
    lowered = reason.lower()
    for marker in markers:
        idx = lowered.find(marker.lower())
        if idx >= 0:
            return reason[idx:].strip().rstrip(".")
    return reason.strip().rstrip(".")


def _decision_cue(example: dict[str, Any]) -> str:
    cues = {
        "clear_reject": "no completed visible object-transfer cycle",
        "near_reject": "unclear target contact/end-state attempt, not reliable enough to keep",
        "near_keep": "clear stacked boundaries despite task imperfection",
        "clear_keep_stacked": "long multi-step task with clear visible subtask boundaries",
        "clear_keep": "clean approach, grasp, transport, and release cycle",
    }
    return cues.get(str(example.get("category_key")), "quality label based on visible task evidence")


def _select_threshold_examples(episodes: pd.DataFrame, labels: pd.DataFrame, quality_threshold: int) -> list[dict[str, Any]]:
    metadata_rows = labels[labels["labeler_name"] == "episode_metadata"]
    segment_rows = labels[labels["labeler_name"] == "subtask_segmenter"]
    subgoal_rows = labels[labels["labeler_name"] == "subgoal_images"]
    by_episode = episodes.set_index("episode_id").to_dict("index")
    segment_path_by_episode = {
        str(row.episode_id): str(row.label_payload_path)
        for row in segment_rows.itertuples(index=False)
        if getattr(row, "label_payload_path", None)
    }
    subgoals_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in subgoal_rows.itertuples(index=False):
        subgoals_by_episode.setdefault(str(row.episode_id), []).append(
            {
                "segment_idx": _safe_int(getattr(row, "segment_idx", None)),
                "path": str(getattr(row, "subgoal_image_path", "") or ""),
            }
        )

    candidates: dict[str, dict[str, Any]] = {}
    for row in metadata_rows.itertuples(index=False):
        episode_id = str(row.episode_id)
        if episode_id not in by_episode:
            continue
        metadata = _parse_json(getattr(row, "metadata_payload_json", None))
        quality = metadata_quality(metadata)
        if quality is None:
            continue
        segment_path = segment_path_by_episode.get(episode_id)
        segment_payload = _read_json(Path(segment_path)) if segment_path else {}
        segments = list(segment_payload.get("segments", []))
        candidates[episode_id] = {
            "episode_id": episode_id,
            "episode": by_episode[episode_id],
            "metadata": metadata,
            "quality": quality,
            "segments": segments,
            "subgoals": sorted(subgoals_by_episode.get(episode_id, []), key=lambda item: item.get("segment_idx") or 0),
        }

    preferred = [
        ("Clear reject", "clear_reject", 1, "#B3261E", "episode_001972"),
        ("Near reject", "near_reject", max(1, quality_threshold - 1), "#D9822B", "episode_005164"),
        ("Clear keep", "clear_keep_stacked", 5, "#247BA0", "episode_015003"),
        ("Clear keep", "clear_keep", 5, "#2E7D32", "episode_003087"),
    ]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for title, key, target_quality, color, preferred_episode in preferred:
        item = None
        if preferred_episode in candidates and candidates[preferred_episode]["quality"] == target_quality:
            item = dict(candidates[preferred_episode])
        else:
            pool = [dict(value) for value in candidates.values() if value["quality"] == target_quality and value["episode_id"] not in used]
            if not pool:
                pool = sorted(
                    [dict(value) for value in candidates.values() if value["episode_id"] not in used],
                    key=lambda value: (abs(value["quality"] - target_quality), value["episode_id"]),
                )
            if pool:
                item = sorted(pool, key=lambda value: value["episode_id"])[0]
        if item is None:
            continue
        item.update({"category": title, "category_key": key, "target_quality": target_quality, "color": color})
        selected.append(item)
        used.add(item["episode_id"])
    return selected


def _draw_threshold_summary(ax, example: dict[str, Any], quality_threshold: int) -> None:
    ax.set_axis_off()
    color = example["color"]
    metadata = example["metadata"]
    quality = int(example["quality"])
    status = "KEEP" if quality >= quality_threshold else "REJECT"
    mistake = str(bool(metadata.get("mistake"))).lower()
    speed = metadata.get("speed", example["episode"].get("num_steps", "unknown"))
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.9,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#FFFFFF",
        )
    )
    ax.text(0.08, 0.86, example["category"].upper(), fontsize=11, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.08, 0.72, f"Quality {quality}/5", fontsize=22, fontweight="bold", color="#202124", transform=ax.transAxes)
    ax.text(0.08, 0.61, f"Result: {status}", fontsize=10, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.08, 0.51, f"Mistake: {mistake}", fontsize=10, color="#303030", transform=ax.transAxes)
    ax.text(0.08, 0.43, f"Speed: {speed} steps", fontsize=10, color="#303030", transform=ax.transAxes)
    task = str(example["episode"].get("language_instruction", example["episode_id"]))
    ax.text(
        0.08,
        0.32,
        "Task:",
        fontsize=9,
        fontweight="bold",
        color="#303030",
        transform=ax.transAxes,
    )
    ax.text(
        0.08,
        0.24,
        _truncate_wrapped(task, width=28, max_lines=2),
        fontsize=8.8,
        color="#303030",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(0.08, 0.065, example["episode_id"], fontsize=8, color="#5B6770", transform=ax.transAxes)


def _draw_video_storyboard(ax, example: dict[str, Any]) -> None:
    ax.set_axis_off()
    frames = _load_frames(Path(str(example["episode"].get("source_path_frames", ""))))
    num_steps = int(example["episode"].get("num_steps", len(frames) if frames is not None else 0) or 0)
    frame_indices = _storyboard_indices(num_steps)
    ax.text(0.0, 1.03, "Video storyboard with subtask spans", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    image_y0, image_y1 = 0.40, 0.92
    gap = 0.012
    tile_w = (1.0 - gap * (len(frame_indices) - 1)) / max(len(frame_indices), 1)
    for idx, step_idx in enumerate(frame_indices):
        x0 = idx * (tile_w + gap)
        x1 = x0 + tile_w
        if frames is not None and len(frames):
            image = np.asarray(frames[min(step_idx, len(frames) - 1)])
        else:
            image = np.full((64, 64, 3), 235, dtype=np.uint8)
        ax.imshow(image, extent=(x0, x1, image_y0, image_y1), aspect="auto")
        ax.add_patch(patches.Rectangle((x0, image_y0), tile_w, image_y1 - image_y0, fill=False, edgecolor="#FFFFFF", linewidth=1.0))
        ax.text(
            x0 + tile_w / 2,
            image_y0 - 0.025,
            f"t={step_idx}",
            ha="center",
            va="top",
            fontsize=8,
            color="#303030",
        )

    timeline_y0, timeline_y1 = 0.18, 0.29
    ax.add_patch(patches.Rectangle((0.0, timeline_y0), 1.0, timeline_y1 - timeline_y0, facecolor="#ECEFF1", edgecolor="#CBD3D8"))
    for i, segment in enumerate(example["segments"]):
        start = max(0, _safe_int(segment.get("start_step")) or 0)
        end = max(start, _safe_int(segment.get("end_step")) or start)
        x0 = start / max(num_steps - 1, 1)
        x1 = min(1.0, (end + 1) / max(num_steps, 1))
        ax.add_patch(
            patches.Rectangle(
                (x0, timeline_y0),
                max(x1 - x0, 0.012),
                timeline_y1 - timeline_y0,
                facecolor=_segment_color(i),
                edgecolor="white",
                linewidth=1.0,
                alpha=0.9,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            timeline_y0 + (timeline_y1 - timeline_y0) / 2,
            str(i),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
        ax.plot([x1, x1], [timeline_y1, timeline_y1 + 0.055], color="#202124", linewidth=1.0)
        ax.scatter([x1], [timeline_y1 + 0.055], marker="v", s=35, color="#202124")
    ax.text(0.0, 0.07, "Colored bars = subtask segments; black triangles = extracted subgoal frames", fontsize=8, color="#303030")


def _draw_annotation_payload(ax, example: dict[str, Any]) -> None:
    ax.set_axis_off()
    metadata = example["metadata"]
    ax.text(0.0, 0.98, "Annotation payload", fontsize=10, fontweight="bold", transform=ax.transAxes, va="top")
    y = 0.88
    for i, segment in enumerate(example["segments"]):
        text = str(segment.get("subtask_text", "")).strip()
        start = _safe_int(segment.get("start_step"))
        end = _safe_int(segment.get("end_step"))
        ax.text(0.0, y, f"{i}. steps {start}-{end}", fontsize=8.5, fontweight="bold", color="#202124", transform=ax.transAxes, va="top")
        y -= 0.045
        wrapped = _truncate_wrapped(text, width=60, max_lines=1)
        line_count = max(1, len(wrapped.splitlines()))
        ax.text(0.03, y, wrapped, fontsize=8.2, color="#303030", transform=ax.transAxes, va="top")
        y -= 0.05 + 0.04 * line_count
        if y < 0.34:
            break

    reason = str(metadata.get("reason", "")).strip()
    meta_lines = [
        f"quality={metadata_quality(metadata)} task_success={metadata.get('task_success_quality', metadata.get('quality'))} mistake={str(bool(metadata.get('mistake'))).lower()}",
        "cue: " + _decision_cue(example),
    ]
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.0, 0.02),
            0.98,
            0.27,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#CBD3D8",
            facecolor="#F7F9FA",
        )
    )
    ax.text(0.03, 0.245, "Episode metadata", fontsize=8.5, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(0.03, 0.19, _wrap(meta_lines[0], 54), fontsize=8, color="#303030", transform=ax.transAxes, va="top")
    ax.text(0.03, 0.13, _wrap(meta_lines[1], 61), fontsize=7.5, color="#303030", transform=ax.transAxes, va="top")


def _load_frames(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        frames = np.load(path, mmap_mode="r")
        if frames.ndim == 4:
            return frames
    except Exception:
        return None
    return None


def _storyboard_indices(num_steps: int, count: int = 5) -> list[int]:
    if num_steps <= 0:
        return [0]
    if num_steps <= count:
        return list(range(num_steps))
    return sorted({int(round(x)) for x in np.linspace(0, num_steps - 1, count)})


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False)) or ""


def _truncate_wrapped(text: str, width: int, max_lines: int) -> str:
    lines = textwrap.wrap(str(text), width=width, break_long_words=False)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept[-1] = kept[-1].rstrip(". ") + "..."
    return "\n".join(kept)


def _quality_counts(labels: pd.DataFrame) -> dict[int, int]:
    values = []
    for value in labels.loc[labels["labeler_name"] == "episode_metadata", "metadata_payload_json"].dropna().tolist():
        data = _parse_json(value)
        quality = metadata_quality(data)
        if quality is not None:
            values.append(int(quality))
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


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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
    parser.add_argument(
        "--threshold-diagram",
        action="store_true",
        help="Also render a four-row threshold keep/reject annotation storyboard.",
    )
    parser.add_argument(
        "--threshold-animation",
        action="store_true",
        help="Also render a four-row animated GIF with the episodes playing and annotations updating.",
    )
    args = parser.parse_args()
    paths = generate_figures(
        snapshot_id=args.snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        output_dir=args.output_dir,
        compare_snapshot_id=args.compare_snapshot,
        include_threshold_diagram=args.threshold_diagram,
        include_threshold_animation=args.threshold_animation,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
