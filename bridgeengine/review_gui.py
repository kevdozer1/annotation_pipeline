from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from bridgeengine.calibration import (
    calibration_reliability,
    default_gold_path,
    load_or_create_calibration_gold,
    review_summary,
    update_episode_review,
)
from bridgeengine.paths import data_root as resolve_data_root


class ReviewDataset:
    def __init__(self, snapshot_id: str, data_root: str | Path | None = None, gold_file: str | Path | None = None):
        self.snapshot_id = snapshot_id
        self.root = resolve_data_root(data_root)
        self.snapshot_path = self.root / "snapshots" / snapshot_id
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {self.snapshot_path}")
        self.gold_file = Path(gold_file) if gold_file else default_gold_path(snapshot_id, self.root)
        load_or_create_calibration_gold(snapshot_id, self.gold_file, data_root=self.root)
        self._load_tables()
        self._lock = threading.Lock()

    def _load_tables(self) -> None:
        self.episodes = pd.read_parquet(self.snapshot_path / "episodes.parquet").sort_values("episode_id").reset_index(drop=True)
        self.labels = pd.read_parquet(self.snapshot_path / "labels.parquet").sort_values(["episode_id", "labeler_name"]).reset_index(drop=True)
        self.episode_ids = [str(x) for x in self.episodes["episode_id"].tolist()]

    def state(self) -> dict[str, Any]:
        summary = review_summary(self.snapshot_id, self.gold_file, data_root=self.root)
        reviewed = int(summary["reviewed"].sum()) if not summary.empty else 0
        report = calibration_reliability(self.snapshot_id, self.gold_file, data_root=self.root)
        queue = [
            {
                "episode_id": str(row.episode_id),
                "task": str(row.task),
                "reviewed": bool(row.reviewed),
                "auto_score": _none_if_nan(row.auto_score),
                "gold_score": _none_if_nan(row.gold_score),
                "auto_keep": _none_if_nan(row.auto_keep),
                "gold_keep": _none_if_nan(row.gold_keep),
                "boundary_clarity": _none_if_nan(row.boundary_clarity),
                "notes": str(row.notes or ""),
            }
            for row in summary.itertuples(index=False)
        ]
        return {
            "snapshot_id": self.snapshot_id,
            "gold_file": str(self.gold_file.resolve()),
            "episode_count": len(self.episode_ids),
            "reviewed_count": reviewed,
            "remaining_count": len(self.episode_ids) - reviewed,
            "quality_exact_agreement": report.get("quality_exact_agreement"),
            "quality_within_one_agreement": report.get("quality_within_one_agreement"),
            "boundary_iou": report.get("subtask_boundary_temporal_iou_mean"),
            "subgoal_agreement": report.get("subgoal_selection_agreement"),
            "queue": queue,
        }

    def episode_payload(self, episode_id: str) -> dict[str, Any]:
        if episode_id not in self.episode_ids:
            raise KeyError(f"Unknown episode: {episode_id}")
        episode = self.episodes.loc[self.episodes["episode_id"].astype(str) == episode_id].iloc[0].to_dict()
        label_map = self._label_map(episode_id)
        metadata = _load_metadata(label_map.get("episode_metadata")) or {}
        segments = _load_segments(label_map.get("subtask_segmenter"))
        subgoals = self._subgoals(label_map)
        summary = review_summary(self.snapshot_id, self.gold_file, data_root=self.root)
        row = summary.loc[summary["episode_id"].astype(str) == episode_id]
        review = row.iloc[0].to_dict() if not row.empty else {}
        auto_score = _safe_int(metadata.get("curation_quality")) or _safe_int(metadata.get("quality")) or 3
        gold_score = _safe_int(review.get("gold_score")) or auto_score
        video_path = Path(str(episode.get("source_path_video", "")))
        return {
            "episode_id": episode_id,
            "task": str(episode.get("language_instruction", "")),
            "num_steps": int(episode.get("num_steps", 0) or 0),
            "video_url": f"/video/{episode_id}" if self.video_path(episode_id) is not None else None,
            "video_exists": self.video_path(episode_id) is not None,
            "video_path": str(video_path),
            "segments": _color_segments(segments),
            "metadata": _jsonable(metadata),
            "subgoals": subgoals,
            "review": {
                "reviewed": bool(review.get("reviewed", False)),
                "auto_score": auto_score,
                "gold_score": gold_score,
                "auto_mistake": bool(metadata.get("mistake", False)),
                "gold_mistake": _none_if_nan(review.get("gold_mistake")),
                "notes": "" if review.get("notes") is None or pd.isna(review.get("notes")) else str(review.get("notes")),
            },
            "prev_episode_id": self.prev_episode_id(episode_id),
            "next_episode_id": self.next_episode_id(episode_id),
            "next_unreviewed_episode_id": self.next_unreviewed_episode_id(episode_id),
        }

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id = str(payload.get("episode_id", ""))
        if episode_id not in self.episode_ids:
            raise KeyError(f"Unknown episode: {episode_id}")
        with self._lock:
            update_episode_review(
                self.snapshot_id,
                episode_id,
                curation_quality=int(payload.get("score", 3)),
                mistake=bool(payload.get("mistake", False)),
                reason=str(payload.get("reason", "")),
                review_notes=str(payload.get("notes", "")),
                accept_auto_metadata=bool(payload.get("accept_auto_metadata", False)),
                accept_auto_subtasks=bool(payload.get("accept_auto_subtasks", False)),
                accept_auto_subgoals=bool(payload.get("accept_auto_subgoals", False)),
                gold_file=self.gold_file,
                data_root=self.root,
            )
        next_id = self.next_unreviewed_episode_id(episode_id) or self.next_episode_id(episode_id)
        return {"saved": True, "episode_id": episode_id, "next_episode_id": next_id, "state": self.state()}

    def video_path(self, episode_id: str) -> Path | None:
        if episode_id not in self.episode_ids:
            return None
        episode = self.episodes.loc[self.episodes["episode_id"].astype(str) == episode_id].iloc[0].to_dict()
        path = Path(str(episode.get("source_path_video", "")))
        if path.exists() and path.stat().st_size > 128:
            return path
        return None

    def subgoal_path(self, episode_id: str, segment_idx: int) -> Path | None:
        label_map = self._label_map(episode_id)
        for row in self._subgoals(label_map, with_urls=False):
            if int(row.get("segment_idx", -1)) == int(segment_idx):
                path = Path(str(row.get("subgoal_image_path", "")))
                return path if path.exists() else None
        return None

    def next_episode_id(self, episode_id: str) -> str | None:
        idx = self.episode_ids.index(episode_id)
        return self.episode_ids[(idx + 1) % len(self.episode_ids)] if self.episode_ids else None

    def prev_episode_id(self, episode_id: str) -> str | None:
        idx = self.episode_ids.index(episode_id)
        return self.episode_ids[(idx - 1) % len(self.episode_ids)] if self.episode_ids else None

    def next_unreviewed_episode_id(self, episode_id: str) -> str | None:
        summary = review_summary(self.snapshot_id, self.gold_file, data_root=self.root)
        reviewed = {str(row.episode_id) for row in summary.itertuples(index=False) if bool(row.reviewed)}
        if len(reviewed) >= len(self.episode_ids):
            return None
        start = self.episode_ids.index(episode_id)
        for offset in range(1, len(self.episode_ids) + 1):
            candidate = self.episode_ids[(start + offset) % len(self.episode_ids)]
            if candidate not in reviewed:
                return candidate
        return None

    def _label_map(self, episode_id: str) -> dict[str, Any]:
        episode_labels = self.labels.loc[self.labels["episode_id"].astype(str) == str(episode_id)]
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

    def _subgoals(self, label_map: dict[str, Any], with_urls: bool = True) -> list[dict[str, Any]]:
        rows = label_map.get("subgoal_images")
        if not rows:
            return []
        if not isinstance(rows, list):
            rows = [rows]
        out = []
        for row in sorted(rows, key=lambda x: x.get("segment_idx") if x.get("segment_idx") is not None else -1):
            segment_idx = int(row.get("segment_idx") if row.get("segment_idx") is not None else -1)
            path = Path(str(row.get("subgoal_image_path", "")))
            item = {
                "segment_idx": segment_idx,
                "subgoal_image_path": str(path),
                "exists": path.exists(),
            }
            if with_urls and path.exists():
                episode_id = str(row.get("episode_id"))
                item["url"] = f"/subgoal/{episode_id}/{segment_idx}"
            out.append(item)
        return out


def make_handler(dataset: ReviewDataset):
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "BridgeEngineReviewGUI/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/":
                    self._send_html(INDEX_HTML)
                elif path == "/api/state":
                    self._send_json(dataset.state())
                elif path.startswith("/api/episode/"):
                    episode_id = unquote(path.rsplit("/", 1)[-1])
                    self._send_json(dataset.episode_payload(episode_id))
                elif path.startswith("/video/"):
                    episode_id = unquote(path.rsplit("/", 1)[-1])
                    video = dataset.video_path(episode_id)
                    if video is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
                    else:
                        self._send_file(video)
                elif path.startswith("/subgoal/"):
                    match = re.match(r"^/subgoal/([^/]+)/(-?\d+)$", path)
                    if not match:
                        self.send_error(HTTPStatus.NOT_FOUND, "Subgoal not found")
                        return
                    episode_id = unquote(match.group(1))
                    segment_idx = int(match.group(2))
                    image = dataset.subgoal_path(episode_id, segment_idx)
                    if image is None:
                        self.send_error(HTTPStatus.NOT_FOUND, "Subgoal not found")
                    else:
                        self._send_file(image)
                elif path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:  # pragma: no cover - defensive server guard
                self._send_json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
                if parsed.path == "/api/review":
                    self._send_json(dataset.save_review(payload))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:  # pragma: no cover - defensive server guard
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_html(self, html: str) -> None:
            encoded = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            encoded = json.dumps(_jsonable(payload), sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def _send_file(self, path: Path) -> None:
            size = path.stat().st_size
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            range_header = self.headers.get("Range")
            if range_header:
                match = re.match(r"bytes=(\d*)-(\d*)", range_header)
                if not match:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                start = int(match.group(1) or 0)
                end = int(match.group(2) or size - 1)
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with path.open("rb") as handle:
                    handle.seek(start)
                    self.wfile.write(handle.read(length))
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)

    return ReviewHandler


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BridgeEngine Gold Review</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #64748b;
      --line: #d9dee7;
      --blue: #2563eb;
      --green: #16835b;
      --orange: #e8752a;
      --red: #c53b3b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 54px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    header h1 { font-size: 18px; margin: 0; }
    header .stats { display: flex; gap: 18px; color: var(--muted); font-size: 13px; }
    main {
      height: calc(100vh - 54px);
      display: grid;
      grid-template-columns: minmax(240px, 300px) minmax(520px, 1fr) minmax(360px, 440px);
      gap: 12px;
      padding: 12px;
    }
    aside, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-height: 0;
    }
    .queue { display: flex; flex-direction: column; }
    .queue-toolbar { padding: 10px; border-bottom: 1px solid var(--line); }
    .queue-list { overflow-y: auto; padding: 6px; }
    .queue-item {
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      background: transparent;
      padding: 8px;
      border-radius: 6px;
      cursor: pointer;
      color: var(--ink);
    }
    .queue-item:hover { background: #f1f5f9; }
    .queue-item.active { border-color: var(--blue); background: #eff6ff; }
    .queue-item.reviewed .qid { color: var(--green); }
    .qid { font-size: 12px; font-weight: 700; }
    .qtask { font-size: 12px; color: var(--muted); line-height: 1.25; margin-top: 3px; }
    .qscore { font-size: 12px; margin-top: 4px; }
    .video-panel { display: flex; flex-direction: column; }
    .video-wrap { padding: 12px; border-bottom: 1px solid var(--line); }
    video {
      width: 100%;
      max-height: 58vh;
      background: #111827;
      border-radius: 6px;
    }
    .episode-title { padding: 12px 12px 0; }
    .episode-title h2 { margin: 0 0 5px; font-size: 18px; }
    .episode-title p { margin: 0; color: var(--muted); line-height: 1.35; }
    .timeline { padding: 12px; }
    .active-subtask {
      min-height: 48px;
      border-radius: 6px;
      color: white;
      font-weight: 700;
      padding: 10px;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
    }
    .bar { display: flex; width: 100%; height: 42px; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
    .seg {
      height: 100%;
      color: white;
      font-size: 11px;
      line-height: 1.1;
      padding: 5px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      border-right: 1px solid rgba(255,255,255,0.3);
    }
    .seg.active { outline: 3px solid #111827; outline-offset: -3px; }
    .subgoals { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 0 12px 12px; }
    .subgoals img { width: 100%; border-radius: 6px; border: 1px solid var(--line); }
    .subgoals span { font-size: 11px; color: var(--muted); }
    .review { overflow-y: auto; padding: 12px; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button, input, textarea {
      font: inherit;
    }
    button.primary {
      background: var(--blue);
      color: white;
      border: 0;
      padding: 11px 12px;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
      width: 100%;
    }
    button.secondary {
      background: #fff;
      color: var(--ink);
      border: 1px solid var(--line);
      padding: 9px 10px;
      border-radius: 6px;
      cursor: pointer;
    }
    .score-grid { display: grid; grid-template-columns: 1fr; gap: 7px; margin: 10px 0; }
    .score-grid label {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      cursor: pointer;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .score-grid label:has(input:checked) {
      border-color: var(--blue);
      background: #eff6ff;
    }
    .meta {
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.4;
      white-space: pre-wrap;
    }
    textarea {
      width: 100%;
      min-height: 74px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      margin: 5px 0 10px;
    }
    .checks label { display: block; margin: 8px 0; font-size: 13px; }
    .muted { color: var(--muted); }
    .pill {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; height: auto; }
      aside, section { min-height: 320px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>BridgeEngine Gold Review</h1>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <aside class="queue">
      <div class="queue-toolbar">
        <div class="row">
          <button class="secondary" onclick="prevEpisode()">Previous</button>
          <button class="secondary" onclick="nextEpisode()">Next</button>
        </div>
        <div class="row" style="margin-top:8px">
          <button class="secondary" onclick="nextUnreviewed()">Next unreviewed</button>
        </div>
      </div>
      <div class="queue-list" id="queue"></div>
    </aside>
    <section class="video-panel">
      <div class="episode-title">
        <h2 id="episodeId">Loading...</h2>
        <p id="task"></p>
      </div>
      <div class="video-wrap">
        <video id="video" controls autoplay muted playsinline></video>
      </div>
      <div class="timeline">
        <div class="active-subtask" id="activeSubtask">No active subtask</div>
        <div class="bar" id="timelineBar"></div>
      </div>
      <div class="subgoals" id="subgoals"></div>
    </section>
    <section class="review">
      <div class="row">
        <span class="pill" id="autoScore"></span>
        <span class="pill" id="boundary"></span>
        <span class="pill" id="autoKeep"></span>
      </div>
      <h3>Your calibrated score</h3>
      <div class="score-grid" id="scoreGrid"></div>
      <div class="checks">
        <label><input type="checkbox" id="mistake"> Mistake visible</label>
        <label><input type="checkbox" id="acceptMeta"> This matches the auto metadata judgment</label>
        <label><input type="checkbox" id="acceptSubtasks"> Accept auto subtask boundaries for reliability</label>
        <label><input type="checkbox" id="acceptSubgoals"> Accept auto subgoal frames for reliability</label>
      </div>
      <label class="muted">Calibration reason</label>
      <textarea id="reason"></textarea>
      <label class="muted">Review notes</label>
      <textarea id="notes"></textarea>
      <button class="primary" onclick="saveReview(true)">Save review and next</button>
      <div class="row" style="margin-top:8px">
        <button class="secondary" onclick="saveReview(false)">Save only</button>
      </div>
      <h3>Auto metadata</h3>
      <div class="meta" id="metadata"></div>
    </section>
  </main>
<script>
const scoreLabels = {
  1: '1 - clear reject',
  2: '2 - reject',
  3: '3 - near reject',
  4: '4 - near keep',
  5: '5 - clear keep'
};
let appState = null;
let episode = null;
let currentId = null;

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function init() {
  appState = await fetchJSON('/api/state');
  currentId = firstUnreviewed() || appState.queue[0]?.episode_id;
  renderState();
  await loadEpisode(currentId);
}

function renderState() {
  document.getElementById('stats').innerHTML = `
    <span>${appState.snapshot_id}</span>
    <span>${appState.reviewed_count}/${appState.episode_count} reviewed</span>
    <span>exact ${fmt(appState.quality_exact_agreement)}</span>
    <span>within-one ${fmt(appState.quality_within_one_agreement)}</span>
  `;
  const queue = document.getElementById('queue');
  queue.innerHTML = '';
  for (const item of appState.queue) {
    const button = document.createElement('button');
    button.className = `queue-item ${item.reviewed ? 'reviewed' : ''} ${item.episode_id === currentId ? 'active' : ''}`;
    button.onclick = () => loadEpisode(item.episode_id);
    const score = item.gold_score == null ? `auto ${item.auto_score}` : `auto ${item.auto_score} -> gold ${item.gold_score}`;
    button.innerHTML = `<div class="qid">${item.reviewed ? 'reviewed' : 'open'} · ${item.episode_id}</div><div class="qtask">${escapeHtml(item.task || '')}</div><div class="qscore">${score}</div>`;
    queue.appendChild(button);
  }
}

async function loadEpisode(id) {
  if (!id) return;
  currentId = id;
  episode = await fetchJSON(`/api/episode/${encodeURIComponent(id)}`);
  renderState();
  renderEpisode();
}

function renderEpisode() {
  document.getElementById('episodeId').textContent = episode.episode_id;
  document.getElementById('task').textContent = episode.task;
  const video = document.getElementById('video');
  if (episode.video_url) {
    video.style.display = 'block';
    video.src = `${episode.video_url}?t=${Date.now()}`;
    video.load();
    video.play().catch(() => {});
  } else {
    video.removeAttribute('src');
    video.style.display = 'none';
  }
  document.getElementById('autoScore').textContent = `Auto score: ${episode.review.auto_score}`;
  document.getElementById('boundary').textContent = `Boundary: ${episode.metadata.boundary_clarity || 'unknown'}`;
  document.getElementById('autoKeep').textContent = `Auto keep: ${episode.metadata.curation_keep ? 'yes' : 'no'}`;
  document.getElementById('mistake').checked = Boolean(episode.review.gold_mistake ?? episode.review.auto_mistake);
  document.getElementById('acceptMeta').checked = Boolean(episode.review.gold_score === episode.review.auto_score);
  document.getElementById('acceptSubtasks').checked = false;
  document.getElementById('acceptSubgoals').checked = false;
  document.getElementById('reason').value = episode.metadata.scoring_reason || '';
  document.getElementById('notes').value = episode.review.notes || '';
  renderScores(episode.review.gold_score || episode.review.auto_score || 3);
  renderTimeline();
  renderSubgoals();
  renderMetadata();
}

function renderScores(selected) {
  const grid = document.getElementById('scoreGrid');
  grid.innerHTML = '';
  for (const score of [1, 2, 3, 4, 5]) {
    const id = `score-${score}`;
    const label = document.createElement('label');
    label.innerHTML = `<input type="radio" name="score" id="${id}" value="${score}" ${score === Number(selected) ? 'checked' : ''}> ${scoreLabels[score]}`;
    grid.appendChild(label);
  }
}

function renderTimeline() {
  const bar = document.getElementById('timelineBar');
  bar.innerHTML = '';
  const total = Math.max(1, episode.num_steps || 1);
  for (const segment of episode.segments || []) {
    const width = Math.max(1, ((segment.end_step - segment.start_step + 1) / total) * 100);
    const div = document.createElement('div');
    div.className = 'seg';
    div.dataset.segmentIdx = segment.segment_idx;
    div.style.width = `${width}%`;
    div.style.background = segment.color;
    div.title = `${segment.start_step}-${segment.end_step}: ${segment.subtask_text}`;
    div.textContent = segment.subtask_text;
    bar.appendChild(div);
  }
  updateActiveSubtask();
}

function renderSubgoals() {
  const root = document.getElementById('subgoals');
  root.innerHTML = '';
  for (const subgoal of episode.subgoals || []) {
    if (!subgoal.url) continue;
    const item = document.createElement('div');
    item.innerHTML = `<img src="${subgoal.url}" alt="subgoal ${subgoal.segment_idx}"><span>segment ${subgoal.segment_idx}</span>`;
    root.appendChild(item);
  }
}

function renderMetadata() {
  const selected = {
    task_success_quality: episode.metadata.task_success_quality,
    curation_quality: episode.metadata.curation_quality ?? episode.metadata.quality,
    curation_keep: episode.metadata.curation_keep,
    mistake: episode.metadata.mistake,
    boundary_clarity: episode.metadata.boundary_clarity,
    structure_score: episode.metadata.interaction_structure_score,
    scoring_reason: episode.metadata.scoring_reason,
    vlm_reason: episode.metadata.reason
  };
  document.getElementById('metadata').textContent = JSON.stringify(selected, null, 2);
}

function updateActiveSubtask() {
  if (!episode || !episode.segments) return;
  const video = document.getElementById('video');
  const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
  const step = Math.round((video.currentTime / duration) * Math.max(1, (episode.num_steps || 1) - 1));
  let active = episode.segments.find(s => step >= s.start_step && step <= s.end_step) || episode.segments[0];
  const box = document.getElementById('activeSubtask');
  if (active) {
    box.style.background = active.color;
    box.textContent = `Step ${step}: ${active.subtask_text}`;
  }
  for (const div of document.querySelectorAll('.seg')) {
    div.classList.toggle('active', active && Number(div.dataset.segmentIdx) === Number(active.segment_idx));
  }
}

async function saveReview(advance) {
  const score = Number(document.querySelector('input[name="score"]:checked')?.value || 3);
  const payload = {
    episode_id: episode.episode_id,
    score,
    mistake: document.getElementById('mistake').checked,
    accept_auto_metadata: document.getElementById('acceptMeta').checked,
    accept_auto_subtasks: document.getElementById('acceptSubtasks').checked,
    accept_auto_subgoals: document.getElementById('acceptSubgoals').checked,
    reason: document.getElementById('reason').value,
    notes: document.getElementById('notes').value
  };
  const result = await fetchJSON('/api/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  appState = result.state;
  if (advance && result.next_episode_id) {
    await loadEpisode(result.next_episode_id);
  } else {
    await loadEpisode(episode.episode_id);
  }
}

function firstUnreviewed() {
  return appState.queue.find(item => !item.reviewed)?.episode_id;
}

function nextUnreviewed() {
  const currentIndex = appState.queue.findIndex(item => item.episode_id === currentId);
  for (let i = 1; i <= appState.queue.length; i++) {
    const item = appState.queue[(currentIndex + i) % appState.queue.length];
    if (!item.reviewed) return loadEpisode(item.episode_id);
  }
}

function nextEpisode() {
  if (episode?.next_episode_id) loadEpisode(episode.next_episode_id);
}

function prevEpisode() {
  if (episode?.prev_episode_id) loadEpisode(episode.prev_episode_id);
}

function fmt(value) {
  return value == null ? 'n/a' : Number(value).toFixed(3);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
}

document.getElementById('video').addEventListener('timeupdate', updateActiveSubtask);
document.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.key === 'Enter') saveReview(true);
  if (event.key === 'ArrowRight' && !event.target.matches('textarea,input')) nextEpisode();
  if (event.key === 'ArrowLeft' && !event.target.matches('textarea,input')) prevEpisode();
});
init().catch(err => {
  document.body.innerHTML = `<pre style="padding:20px;color:#b91c1c">${escapeHtml(err.message)}</pre>`;
});
</script>
</body>
</html>"""


def _load_segments(row: dict | list | None) -> list[dict[str, Any]]:
    if isinstance(row, list):
        row = row[0] if row else None
    if not row:
        return []
    path = Path(str(row.get("label_payload_path", "")))
    if not path.exists():
        return []
    return _read_json(path).get("segments", [])


def _load_metadata(row: dict | list | None) -> dict[str, Any] | None:
    if isinstance(row, list):
        row = row[0] if row else None
    if not row:
        return None
    if row.get("metadata_payload_json"):
        return json.loads(row["metadata_payload_json"])
    path = Path(str(row.get("label_payload_path", "")))
    if path.exists():
        return _read_json(path).get("metadata")
    return None


def _color_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    colors = ["#e8752a", "#4c78a8", "#59a14f", "#b279a2", "#edc948", "#76b7b2"]
    output = []
    for idx, segment in enumerate(segments):
        item = dict(segment)
        item["segment_idx"] = int(item.get("segment_idx", idx))
        item["start_step"] = int(item.get("start_step", 0) or 0)
        item["end_step"] = int(item.get("end_step", item["start_step"]) or item["start_step"])
        item["subtask_text"] = str(item.get("subtask_text", ""))
        item["color"] = colors[idx % len(colors)]
        output.append(item)
    return output


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _jsonable(value: Any) -> Any:
    value = _none_if_nan(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found starting at {start}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a browser-based BridgeEngine gold review GUI.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--gold-file", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    dataset = ReviewDataset(args.snapshot, data_root=Path(args.data_root) if args.data_root else None, gold_file=args.gold_file)
    port = _find_free_port(args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(dataset))
    url = f"http://{args.host}:{port}"
    print(f"BridgeEngine review GUI: {url}")
    print(f"Snapshot: {args.snapshot}")
    print(f"Gold file: {dataset.gold_file.resolve()}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review GUI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
