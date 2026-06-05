from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from bridgeengine.ingest.snapshot import LABELER_VERSIONS

from .base import LabelResult, episode_id_from_path, read_metadata, sha256_file, write_json
from .backends import VisionLanguageBackend, build_vlm_backend

PROMPT_TEMPLATE = """You are annotating a BridgeData V2 robot manipulation episode.
Given the task instruction and evenly-spaced keyframes, return a JSON array of 2-5 semantic subtasks.
Each subtask must include start_step, end_step, and subtask_text.
Use action-relevant boundaries. Avoid generic fixed templates unless the video genuinely supports them.
Task: {task}
"""


class SubtaskSegmenter:
    """pi0.7-style subtask instruction labeler.

    Live VLM backends are the production path for demo labels. The
    deterministic adapter is kept only for CI and downstream plumbing tests,
    and its provenance is marked so benchmark runs can reject it.
    """

    name = "subtask_segmenter"
    version = LABELER_VERSIONS[name]

    def __init__(
        self,
        output_root: Path,
        n_keyframes: int = 6,
        allow_fallback: bool = False,
        backend: VisionLanguageBackend | None = None,
        backend_name: str | None = None,
        backend_model: str | None = None,
    ):
        self.output_root = output_root
        self.n_keyframes = n_keyframes
        self.allow_fallback = allow_fallback
        self._backend = backend
        self.backend_name = backend_name
        self.backend_model = backend_model

    def label_episode(self, episode_path: Path, snapshot_id: str) -> LabelResult:
        t0 = time.perf_counter()
        episode_id = episode_id_from_path(episode_path)
        metadata = read_metadata(episode_path)
        task = metadata.get("task") or metadata.get("language_instruction") or episode_id
        frames_path = episode_path / "frames.npy"
        actions_path = episode_path / "actions.npy"
        frames = _load_frames(frames_path)
        actions = _load_actions(actions_path)
        num_steps = _num_steps(frames, actions)
        boundary_steps = _action_boundaries(num_steps, actions)
        keyframes = _keyframe_indices(num_steps, self.n_keyframes, boundary_steps)
        prompt_hash = _prompt_hash(PROMPT_TEMPLATE)
        raw_root = self.output_root / "snapshots" / snapshot_id / "raw_vlm_outputs" / episode_id
        observe_output_path = raw_root / "subtask_segmenter_observe.json"
        label_output_path = raw_root / "subtask_segmenter_label.json"
        raw_output_path_used = label_output_path
        observations: dict | list | None = None
        quality_warnings: list[str] = []
        backend_id = None
        backend_model = None
        if frames is not None:
            try:
                segments, observations, raw_output_path_used, quality_warnings, backend_id, backend_model = _segments_from_backend(
                    self._get_backend(),
                    frames,
                    keyframes,
                    task,
                    num_steps,
                    boundary_steps,
                    observe_output_path,
                    label_output_path,
                )
                fallback_mode = None
                confidence = _confidence_for_segments(segments, num_steps, fallback=False)
            except Exception as exc:
                if not self.allow_fallback:
                    raise
                segments = _segment_episode(task, num_steps, actions)
                fallback_mode = "deterministic_action_aware_adapter"
                confidence = _confidence_for_segments(segments, num_steps, fallback=True)
                _write_fallback_raw(label_output_path, str(exc), segments, keyframes)
        elif self.allow_fallback:
            segments = _segment_episode(task, num_steps, actions)
            fallback_mode = "deterministic_action_aware_adapter"
            confidence = _confidence_for_segments(segments, num_steps, fallback=True)
            _write_fallback_raw(label_output_path, "frames.npy missing", segments, keyframes)
        else:
            raise FileNotFoundError(f"frames.npy missing for live VLM segmentation: {episode_path}")
        prompt_components = [
            _prompt_text(task, segment["subtask_text"], None)
            for segment in segments
        ]
        payload = {
            "episode_id": episode_id,
            "task": task,
            "segments": segments,
            "prompt_components": prompt_components,
            "n_keyframes": len(keyframes),
            "keyframe_indices": keyframes,
            "prompt_template_hash": prompt_hash,
            "fallback_mode": fallback_mode,
            "vlm_backend": backend_id,
            "vlm_model": backend_model,
            "stage_one_observations": observations,
            "quality_warnings": quality_warnings,
            "boundary_source": "gripper_transition_vlm_text" if fallback_mode is None else "fallback_adapter",
            "intended_vlm": backend_model or "fallback_adapter",
            "raw_observation_output_path": str(observe_output_path.resolve()),
            "raw_vlm_output_path": str(raw_output_path_used.resolve()),
        }
        payload_path = self.output_root / "labels" / "subtask_segments" / snapshot_id / f"{episode_id}.json"
        write_json(payload_path, payload)
        dt = time.perf_counter() - t0
        provenance = {
            "input_sha256": sha256_file(frames_path),
            "actions_sha256": sha256_file(actions_path),
            "labeler_version": self.version,
            "n_keyframes_used": len(keyframes),
            "keyframe_indices": keyframes,
            "prompt_template_hash": prompt_hash,
            "vlm_logprobs": None,
            "fallback_mode": fallback_mode,
            "vlm_backend": backend_id,
            "vlm_model": backend_model,
            "quality_warnings": quality_warnings,
            "boundary_source": "gripper_transition_vlm_text" if fallback_mode is None else "fallback_adapter",
            "raw_vlm_output_path": str(raw_output_path_used.resolve()),
            "raw_observation_output_path": str(observe_output_path.resolve()),
            "raw_label_output_path": str(label_output_path.resolve()),
            "wall_clock_seconds": dt,
            "segment_count": len(segments),
        }
        return LabelResult(self.name, self.version, episode_id, payload_path, confidence, provenance)

    def _get_backend(self) -> VisionLanguageBackend:
        if self._backend is None:
            self._backend = build_vlm_backend(self.backend_name, self.backend_model)
        return self._backend


def _load_frames(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def _load_actions(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def _num_steps(frames: np.ndarray | None, actions: np.ndarray | None) -> int:
    if frames is not None:
        return int(frames.shape[0])
    if actions is not None:
        return int(actions.shape[0])
    return 1


def _keyframe_indices(num_steps: int, n_keyframes: int, boundaries: list[int] | None = None) -> list[int]:
    if num_steps <= 1:
        return [0]
    wanted = set(int(x) for x in np.linspace(0, num_steps - 1, min(n_keyframes, num_steps)).round())
    if boundaries:
        for boundary in boundaries:
            wanted.add(min(max(int(boundary), 0), num_steps - 1))
            wanted.add(min(max(int(boundary) - 1, 0), num_steps - 1))
    return sorted(wanted)


def _segment_episode(task: str, num_steps: int, actions: np.ndarray | None) -> list[dict]:
    boundaries = _action_boundaries(num_steps, actions)
    texts = _subtask_texts(task, len(boundaries) - 1)
    segments = []
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        end = max(start, end - 1)
        segments.append(
            {
                "segment_idx": idx,
                "start_step": int(start),
                "end_step": int(min(end, num_steps - 1)),
                "subtask_text": texts[idx],
            }
        )
    return _validate_segments(segments, num_steps)


def _segments_from_backend(
    backend: VisionLanguageBackend,
    frames: np.ndarray,
    keyframes: list[int],
    task: str,
    num_steps: int,
    boundaries: list[int],
    observe_output_path: Path,
    label_output_path: Path,
) -> tuple[list[dict], dict | list, Path, list[str], str, str]:
    observed = backend.query_contact_sheet(
        frames,
        keyframes,
        _observation_question(task, num_steps, boundaries),
        observe_output_path,
    )
    observations = _extract_json(observed.answer)
    first = backend.query_contact_sheet(
        frames,
        keyframes,
        _segmenter_question(task, num_steps, boundaries, observations),
        label_output_path,
    )
    try:
        segments = _segments_from_answer(first.answer, num_steps, boundaries)
        issues = _segment_quality_issues(segments, task, observations)
        if issues:
            raise ValueError("; ".join(issues))
        return segments, observations, label_output_path, [], backend.name, backend.model
    except Exception as first_error:
        retry_path = label_output_path.with_name("subtask_segmenter_label_retry.json")
        retry = backend.query_contact_sheet(
            frames,
            keyframes,
            _segmenter_retry_question(task, num_steps, boundaries, observations, first.answer, str(first_error)),
            retry_path,
        )
        segments = _segments_from_answer(retry.answer, num_steps, boundaries)
        issues = _segment_quality_issues(segments, task, observations)
        if issues:
            return segments, observations, retry_path, issues, backend.name, backend.model
        return segments, observations, retry_path, [], backend.name, backend.model


def _action_boundaries(num_steps: int, actions: np.ndarray | None) -> list[int]:
    if num_steps <= 4:
        return [0, num_steps]
    candidates: list[int] = []
    if actions is not None and actions.ndim == 2 and actions.shape[1] >= 1:
        grip = actions[:, -1]
        transitions = np.where(np.abs(np.diff(grip)) > 0.25)[0] + 1
        candidates.extend(int(x) for x in transitions if 2 <= x <= num_steps - 1)
    interior = _merge_close_points(sorted(set(candidates)), min_gap=3)
    if len(interior) < 1:
        default_count = 3 if num_steps >= 16 else 2
        defaults = [int(round(x)) for x in np.linspace(0, num_steps, default_count + 1)]
        interior = defaults[1:-1]
    if len(interior) > 3:
        interior = _choose_spread(interior, 3, num_steps)
    return [0] + interior + [num_steps]


def _merge_close_points(points: list[int], min_gap: int) -> list[int]:
    merged: list[int] = []
    for point in points:
        if not merged or point - merged[-1] >= min_gap:
            merged.append(point)
        else:
            merged[-1] = int(round((merged[-1] + point) / 2))
    return merged


def _choose_spread(points: list[int], max_points: int, num_steps: int) -> list[int]:
    targets = np.linspace(0, num_steps, max_points + 2)[1:-1]
    chosen = []
    remaining = points[:]
    for target in targets:
        nearest = min(remaining, key=lambda x: abs(x - target))
        chosen.append(nearest)
        remaining.remove(nearest)
    return sorted(chosen)


def _subtask_texts(task: str, count: int) -> list[str]:
    task = task.strip().rstrip(".")
    if count <= 1:
        return [task]
    if count == 2:
        return [f"approach the object for: {task}", f"complete the placement for: {task}"]
    if count == 3:
        return [
            f"approach the target object for: {task}",
            f"grasp and lift the target object",
            f"move to the destination and complete: {task}",
        ]
    return [
        f"approach the target object for: {task}",
        "grasp the target object",
        "transport the object toward the destination",
        f"place or release the object to finish: {task}",
    ][:count]


def _segments_from_answer(answer: str, num_steps: int, boundaries: list[int] | None = None) -> list[dict]:
    data = _extract_json(answer)
    if isinstance(data, dict):
        data = data.get("segments", data.get("subtasks", []))
    if not isinstance(data, list):
        raise ValueError(f"Moondream answer did not contain a JSON list: {answer[:500]}")
    if boundaries:
        expected = len(boundaries) - 1
        if len(data) < expected:
            raise ValueError(f"Moondream returned {len(data)} segments, expected {expected}")
        segments = []
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            item = data[idx]
            if not isinstance(item, dict):
                raise ValueError(f"Segment {idx} was not an object")
            text = str(item.get("subtask_text", item.get("text", item.get("description", "")))).strip()
            if not text:
                raise ValueError(f"Segment {idx} had empty subtask_text")
            segments.append(
                {
                    "segment_idx": idx,
                    "start_step": int(start),
                    "end_step": int(min(max(end - 1, start), num_steps - 1)),
                    "subtask_text": text[:120],
                }
            )
        return segments
    segments = []
    for idx, item in enumerate(data[:5]):
        if not isinstance(item, dict):
            continue
        segments.append(
            {
                "segment_idx": idx,
                "start_step": item.get("start_step", item.get("start_frame")),
                "end_step": item.get("end_step", item.get("end_frame")),
                "subtask_text": item.get("subtask_text", item.get("text", item.get("description", ""))),
            }
        )
    return _validate_segments(segments, num_steps)


def _extract_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_candidates = [idx for idx in [cleaned.find("["), cleaned.find("{")] if idx >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _validate_segments(segments: list[dict], num_steps: int) -> list[dict]:
    if not segments:
        return [{"segment_idx": 0, "start_step": 0, "end_step": max(0, num_steps - 1), "subtask_text": "complete the task"}]
    clean = []
    cursor = 0
    for idx, segment in enumerate(segments[:5]):
        start = cursor
        end = min(max(start, int(segment.get("end_step") or start)), max(0, num_steps - 1))
        clean.append(
            {
                "segment_idx": idx,
                "start_step": start,
                "end_step": end,
                "subtask_text": str(segment["subtask_text"]).strip()[:120],
            }
        )
        cursor = end + 1
    clean[0]["start_step"] = 0
    clean[-1]["end_step"] = max(0, num_steps - 1)
    return clean


def _segment_quality_issues(segments: list[dict], task: str, observations: dict | list | None = None) -> list[str]:
    issues: list[str] = []
    texts = [str(s["subtask_text"]).strip().lower() for s in segments]
    if len(set(texts)) < len(texts):
        issues.append("subtask_text repeated")
    reverse_words = {"remove", "take out", "take away"}
    task_lower = task.lower()
    if not any(word in task_lower for word in reverse_words):
        if any(any(word in text for word in reverse_words) for text in texts):
            issues.append("subtask_text describes removal contrary to task")
    if any(text in {"one short sentence", "complete the task"} for text in texts):
        issues.append("placeholder subtask_text")
    observation_text = json.dumps(observations or {}, sort_keys=True).lower()
    for text in texts:
        for token in _object_like_tokens(text):
            if token not in observation_text and token not in task_lower:
                issues.append(f"object not grounded in observation: {token}")
                break
    return issues


def _object_like_tokens(text: str) -> list[str]:
    stop = {
        "after",
        "approach",
        "across",
        "above",
        "and",
        "at",
        "before",
        "beside",
        "black",
        "blue",
        "carry",
        "clear",
        "destination",
        "edge",
        "empty",
        "finish",
        "from",
        "green",
        "grasp",
        "gray",
        "grey",
        "leave",
        "lift",
        "grasping",
        "metal",
        "move",
        "object",
        "orange",
        "partial",
        "place",
        "placing",
        "pink",
        "pickup",
        "purple",
        "red",
        "release",
        "retract",
        "settle",
        "silver",
        "sink",
        "task",
        "that",
        "the",
        "to",
        "toward",
        "white",
        "withdraw",
        "with",
        "yellow",
    }
    tokens = [t.strip(".,:;()[]{}").lower() for t in text.split()]
    return [t for t in tokens if len(t) >= 4 and t not in stop]


def _confidence_for_segments(segments: list[dict], num_steps: int, fallback: bool) -> float:
    if not segments or num_steps <= 0:
        return 0.25
    coverage = sum(max(0, s["end_step"] - s["start_step"] + 1) for s in segments) / num_steps
    count_score = 1.0 if 2 <= len(segments) <= 5 else 0.65
    base = 0.48 if fallback else 0.64
    return round(float(np.clip(base + 0.2 * coverage + 0.12 * count_score, 0.0, 0.96)), 4)


def _prompt_text(task: str, subtask: str, metadata: dict | None) -> str:
    text = f"Task: {task}. Subtask: {subtask}."
    if metadata:
        text += (
            f" Speed: {metadata['speed']}. Quality: {metadata['quality']}/5."
            f" Mistake: {str(metadata['mistake']).lower()}."
            f" Control Mode: {metadata['control_mode']}."
        )
    return text


def _prompt_hash(template: str) -> str:
    return "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()


def _observation_question(task: str, num_steps: int, boundaries: list[int]) -> str:
    intervals = [
        {"segment_idx": idx, "start_step": int(start), "end_step": int(end - 1)}
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    return (
        "You are observing a BridgeData V2 robot manipulation episode before assigning labels."
        + f"\nTask: {task}"
        + f"\nThe episode has steps 0 through {num_steps - 1}."
        + "\nRobot gripper/action transitions define these intervals:"
        + "\n" + json.dumps(intervals)
        + "\nFor each interval, describe only physical evidence: visible objects, gripper state, arm motion, and likely contact/release."
        + "\nDo not decide subtask labels yet."
        + "\nReturn ONLY valid JSON in this shape:"
        + '\n{"observations":[{"segment_idx":0,"objects":["object","destination"],"gripper":"open/closed evidence","motion":"visible motion","summary":"physical observation"}],"episode_summary":"one physical summary"}'
    )


def _segmenter_question(task: str, num_steps: int, boundaries: list[int], observations: dict | list) -> str:
    intervals = [
        {"segment_idx": idx, "start_step": int(start), "end_step": int(end - 1)}
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    return (
        PROMPT_TEMPLATE.format(task=task)
        + "\nThe contact sheet shows evenly-spaced frames from a robot episode."
        + f"\nThe episode has steps 0 through {num_steps - 1}."
        + "\nRobot gripper/action transitions define these segment intervals:"
        + "\n" + json.dumps(intervals)
        + "\nStage-one physical observations:"
        + "\n" + json.dumps(observations, sort_keys=True)
        + "\nDo not change the start_step or end_step values. Return exactly one object per interval."
        + "\nFor each interval, write a short visible action phrase that names the object and destination when possible."
        + "\nUse temporal roles if needed: move to object, grasp/lift object, carry object, place/release object."
        + "\nDo not use removal/reversal language unless the task instruction itself asks for removal."
        + "\nDo not repeat the exact same subtask_text across intervals."
        + "\nReturn ONLY valid JSON in this exact shape:"
        + '\n{"segments":[{"segment_idx":0,"start_step":0,"end_step":3,"subtask_text":"short action-specific phrase"}]}'
    )


def _segmenter_retry_question(
    task: str,
    num_steps: int,
    boundaries: list[int],
    observations: dict | list,
    previous_answer: str,
    error: str,
) -> str:
    return (
        _segmenter_question(task, num_steps, boundaries, observations)
        + "\nThe previous answer could not be parsed as valid JSON. "
        + "It may also have repeated phrases, contradicted the task, or used invalid boundaries. "
        + "Do not include markdown fences, prose, comments, or trailing commas."
        + f"\nValidation error: {error}"
        + f"\nPrevious answer excerpt: {previous_answer[:500]}"
    )


def _write_fallback_raw(path: Path, error: str, segments: list[dict], keyframes: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fallback_mode": "deterministic_action_aware_adapter",
        "error": error,
        "keyframe_indices": keyframes,
        "segments": segments,
        "warning": "Scaffolding-only fallback output. Do not use for benchmark claims.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
