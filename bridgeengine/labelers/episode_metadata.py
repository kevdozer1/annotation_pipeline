from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from bridgeengine.ingest.snapshot import LABELER_VERSIONS

from .base import LabelResult, episode_id_from_path, read_metadata, sha256_file, write_json
from .backends import VisionLanguageBackend, build_vlm_backend

PROMPT_TEMPLATE = """Rate a robot demonstration from evenly-spaced keyframes and the task instruction.
Return JSON with task_success_quality_1_to_5, curation_quality_1_to_5, and mistake_boolean.
Task-success quality measures whether the requested task was completed.
Curation quality measures whether the episode is useful training data: visible approach, grasp/contact, transport, and release/state-change boundaries.
Quality 5 means clean, unambiguous evidence. Quality 1 means no useful target interaction is visible.
"""
PILOT_PATH = Path("C:/Users/Kevin/projects/LeWM_testbed/outputs/pilot_subset.json")


class EpisodeMetadataLabeler:
    """pi0.7-style episode metadata labeler."""

    name = "episode_metadata"
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
        self._pilot_scores = _load_pilot_scores()

    def label_episode(self, episode_path: Path, snapshot_id: str) -> LabelResult:
        t0 = time.perf_counter()
        episode_id = episode_id_from_path(episode_path)
        metadata = read_metadata(episode_path)
        task = metadata.get("task") or metadata.get("language_instruction") or episode_id
        frames_path = episode_path / "frames.npy"
        actions_path = episode_path / "actions.npy"
        num_steps = _num_steps(frames_path, actions_path)
        keyframes = _keyframe_indices(num_steps, self.n_keyframes)
        raw_root = self.output_root / "snapshots" / snapshot_id / "raw_vlm_outputs" / episode_id
        observe_output_path = raw_root / "episode_metadata_observe.json"
        label_output_path = raw_root / "episode_metadata_label.json"
        frames = np.load(frames_path, mmap_mode="r") if frames_path.exists() else None
        observations: dict | list | None = None
        reason = ""
        metadata_extra: dict[str, object] = {}
        quality_warnings: list[str] = []
        backend_id = None
        backend_model = None
        if frames is not None:
            try:
                backend = self._get_backend()
                observed = backend.query_contact_sheet(
                    frames,
                    keyframes,
                    _metadata_observation_question(task, num_steps),
                    observe_output_path,
                )
                observations = _extract_json(observed.answer)
                judged = backend.query_contact_sheet(
                    frames,
                    keyframes,
                    _metadata_question(task, num_steps, observations),
                    label_output_path,
                )
                quality, mistake, reason, metadata_extra = _metadata_from_answer(judged.answer)
                quality_warnings = _metadata_quality_issues(quality, mistake, reason)
                source = f"{backend.name}_judge"
                backend_id = backend.name
                backend_model = backend.model
            except Exception as exc:
                if not self.allow_fallback:
                    raise
                quality, mistake, source, reason = _quality_and_mistake(episode_id, self._pilot_scores, num_steps)
                _write_fallback_raw(label_output_path, str(exc), quality, mistake, keyframes)
        elif self.allow_fallback:
            quality, mistake, source, reason = _quality_and_mistake(episode_id, self._pilot_scores, num_steps)
            _write_fallback_raw(label_output_path, "frames.npy missing", quality, mistake, keyframes)
        else:
            raise FileNotFoundError(f"frames.npy missing for live VLM metadata judge: {episode_path}")
        fallback_mode = None if source.endswith("_judge") else source
        payload = {
            "episode_id": episode_id,
            "task": task,
            "metadata": {
                "speed": int(num_steps),
                "quality": int(quality),
                "mistake": bool(mistake),
                "control_mode": "end_effector",
                "reason": reason,
                **metadata_extra,
            },
            "judge_source": source,
            "fallback_mode": fallback_mode,
            "vlm_backend": backend_id,
            "vlm_model": backend_model,
            "stage_one_observations": observations,
            "quality_warnings": quality_warnings,
            "n_keyframes": len(keyframes),
            "keyframe_indices": keyframes,
            "prompt_template_hash": _prompt_hash(PROMPT_TEMPLATE),
            "intended_vlm": backend_model or "fallback_adapter",
            "raw_observation_output_path": str(observe_output_path.resolve()),
            "raw_vlm_output_path": str(label_output_path.resolve()),
        }
        payload_path = self.output_root / "labels" / "episode_metadata" / snapshot_id / f"{episode_id}.json"
        write_json(payload_path, payload)
        dt = time.perf_counter() - t0
        provenance = {
            "input_sha256": sha256_file(frames_path),
            "actions_sha256": sha256_file(actions_path),
            "labeler_version": self.version,
            "vlm_version": backend_model,
            "vlm_backend": backend_id,
            "judge_source": source,
            "n_keyframes_used": len(keyframes),
            "keyframe_indices": keyframes,
            "prompt_template_hash": _prompt_hash(PROMPT_TEMPLATE),
            "judge_logprobs": None,
            "fallback_mode": fallback_mode,
            "quality_warnings": quality_warnings,
            "raw_observation_output_path": str(observe_output_path.resolve()),
            "raw_vlm_output_path": str(label_output_path.resolve()),
            "wall_clock_seconds": dt,
        }
        confidence = 0.84 if source.endswith("_judge") else 0.62
        return LabelResult(
            self.name,
            self.version,
            episode_id,
            payload_path,
            confidence,
            provenance,
            metadata_payload_json=json.dumps(payload["metadata"], sort_keys=True),
        )

    def _get_backend(self) -> VisionLanguageBackend:
        if self._backend is None:
            self._backend = build_vlm_backend(self.backend_name, self.backend_model)
        return self._backend


def _load_pilot_scores() -> dict[str, dict]:
    if not PILOT_PATH.exists():
        return {}
    data = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    return {entry["episode_id"]: entry for entry in data.get("core", [])}


def _num_steps(frames_path: Path, actions_path: Path) -> int:
    if frames_path.exists():
        return int(np.load(frames_path, mmap_mode="r").shape[0])
    if actions_path.exists():
        return int(np.load(actions_path, mmap_mode="r").shape[0])
    return 1


def _keyframe_indices(num_steps: int, n_keyframes: int) -> list[int]:
    if num_steps <= 1:
        return [0]
    return [int(x) for x in np.linspace(0, num_steps - 1, min(n_keyframes, num_steps)).round()]


def _quality_and_mistake(episode_id: str, pilot_scores: dict[str, dict], num_steps: int) -> tuple[int, bool, str, str]:
    if episode_id in pilot_scores:
        score = float(pilot_scores[episode_id].get("score", 12.0))
        if score >= 16.0:
            quality = 5
        elif score >= 14.0:
            quality = 4
        elif score >= 12.0:
            quality = 3
        elif score >= 10.5:
            quality = 2
        else:
            quality = 1
        track_grade = str(pilot_scores[episode_id].get("track_grade", "ok"))
        mistake = quality <= 2 or track_grade == "poor"
        return quality, mistake, "lewm_pilot_quality_proxy", "Fallback quality from LEWM pilot proxy."
    if num_steps <= 20:
        return 4, False, "duration_heuristic", "Fallback quality from short episode duration."
    if num_steps >= 45:
        return 2, True, "duration_heuristic", "Fallback quality from long episode duration."
    return 3, False, "duration_heuristic", "Fallback quality from medium episode duration."


def _prompt_hash(template: str) -> str:
    return "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()


def _metadata_observation_question(task: str, num_steps: int) -> str:
    return (
        "You are observing a robot demonstration before rating it."
        + f"\nTask instruction: {task}"
        + f"\nEpisode length: {num_steps} timesteps."
        + "\nDescribe physical evidence only: visible objects, approach, grasp/contact, transport, release/state-change boundaries, whether the requested task appears completed, and visible mistakes."
        + "\nReturn ONLY valid JSON with this shape:"
        + '\n{"objects":["object","destination"],"boundary_evidence":"...", "completion_evidence":"...", "mistake_evidence":"...", "summary":"..."}'
    )


def _metadata_question(task: str, num_steps: int, observations: dict | list) -> str:
    return (
        PROMPT_TEMPLATE
        + f"\nTask instruction: {task}"
        + f"\nEpisode length: {num_steps} timesteps."
        + "\nPhysical observations:"
        + "\n" + json.dumps(observations, sort_keys=True)
        + "\nReturn ONLY valid JSON with this shape:"
        + '\n{"task_success_quality":3,"curation_quality":4,"mistake":false,"boundary_clarity":"clear","reason":"specific evidence consistent with both scores"}'
        + "\nUse task_success_quality for final task completion."
        + "\nUse curation_quality for training usefulness: visible approach/grasp/contact/transport/release or state-change boundaries."
        + "\nA clip can have task_success_quality=3 but curation_quality=4 if the manipulation boundaries are clear but the final requested placement is ambiguous."
        + "\nSet mistake=true for clear wrong object, failed grasp, wrong destination, or unfinished requested task, even if curation_quality is high."
    )


def _metadata_from_answer(answer: str) -> tuple[int, bool, str, dict[str, object]]:
    data = _extract_json(answer)
    if not isinstance(data, dict):
        raise ValueError(f"Moondream metadata answer was not a JSON object: {answer[:500]}")
    task_success_quality = _safe_int(data.get("task_success_quality", data.get("task_success_quality_1_to_5")))
    curation_quality = _safe_int(data.get("curation_quality", data.get("curation_quality_1_to_5")))
    quality = curation_quality if curation_quality is not None else _safe_int(data.get("quality", data.get("quality_1_to_5")))
    if quality is None:
        raise ValueError(f"Metadata answer had no quality score: {answer[:500]}")
    quality = max(1, min(5, quality))
    mistake = data.get("mistake", data.get("mistake_boolean"))
    if isinstance(mistake, str):
        mistake = mistake.strip().lower() in {"true", "yes", "1"}
    reason = str(data.get("reason", "")).strip()
    extra: dict[str, object] = {}
    if task_success_quality is not None:
        extra["task_success_quality"] = max(1, min(5, int(task_success_quality)))
    if curation_quality is not None:
        extra["curation_quality"] = max(1, min(5, int(curation_quality)))
    if data.get("boundary_clarity"):
        extra["boundary_clarity"] = str(data["boundary_clarity"]).strip().lower()
    return quality, bool(mistake), reason, extra


def _metadata_quality_issues(quality: int, mistake: bool, reason: str) -> list[str]:
    issues: list[str] = []
    lower = reason.lower()
    if not reason or lower in {"one short sentence", "specific evidence consistent with the numeric score"}:
        issues.append("placeholder metadata reason")
    success_words = {"success", "successfully", "completed", "clean", "placed", "resting", "inside"}
    failure_words = {"fail", "failed", "wrong", "incomplete", "mistake", "incorrect", "drop"}
    says_success = _has_unnegated_word(lower, success_words)
    says_failure = _has_unnegated_word(lower, failure_words)
    if quality <= 2 and says_success and not says_failure:
        issues.append("quality score contradicts success reason")
    if quality >= 4 and says_failure and not says_success:
        issues.append("quality score contradicts failure reason")
    if mistake and quality >= 5:
        issues.append("mistake true with perfect quality")
    return issues


def _has_unnegated_word(text: str, target_words: set[str]) -> bool:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    negators = {"no", "not", "without", "none", "never"}
    for idx, token in enumerate(tokens):
        if token not in target_words:
            continue
        window = tokens[max(0, idx - 10) : idx]
        if any(word in negators for word in window):
            continue
        return True
    return False


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _write_fallback_raw(path: Path, error: str, quality: int, mistake: bool, keyframes: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fallback_mode": "quality_proxy_adapter",
        "error": error,
        "keyframe_indices": keyframes,
        "quality": quality,
        "mistake": mistake,
        "warning": "Scaffolding-only fallback output. Do not use for benchmark claims.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
