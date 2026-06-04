from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GateIssue:
    episode_id: str
    check: str
    detail: str
    severity: str = "fail"

    def format(self) -> str:
        return f"{self.episode_id}: {self.check}: {self.detail}"


@dataclass(frozen=True)
class GateReport:
    passed: bool
    issues: tuple[GateIssue, ...]
    episode_count: int
    checked_episode_count: int
    quality_counts: dict[int, int]

    @property
    def pass_rate(self) -> float:
        if self.checked_episode_count <= 0:
            return 0.0
        failed = {issue.episode_id for issue in self.issues if issue.episode_id != "__dataset__"}
        return (self.checked_episode_count - len(failed)) / self.checked_episode_count

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        examples = ", ".join(issue.format() for issue in self.issues[:8])
        raise RuntimeError(
            "Refusing to run the benchmark because label quality gates failed. "
            "Inspect and repair labels before benchmarking, or pass --allow-scaffolding-labels for plumbing-only runs. "
            f"Examples: {examples}"
        )

    def to_text(self) -> str:
        lines = [
            f"Quality gate: {'PASS' if self.passed else 'FAIL'}",
            f"Episode pass rate: {self.pass_rate:.3f}",
            f"Quality counts: {self.quality_counts}",
        ]
        if self.issues:
            lines.append("Issues:")
            lines.extend(f"- {issue.format()}" for issue in self.issues)
        return "\n".join(lines)


def evaluate_snapshot_quality(snapshot_path: str | Path) -> GateReport:
    snapshot_path = Path(snapshot_path)
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    issues: list[GateIssue] = []
    quality_values: list[int] = []
    if labels.empty:
        issues.append(GateIssue("__dataset__", "missing_labels", "labels.parquet has no label rows"))

    for row in labels.to_dict("records"):
        episode_id = str(row.get("episode_id"))
        labeler_name = row.get("labeler_name")
        provenance = _parse_json(row.get("provenance_json"))
        fallback_mode = provenance.get("fallback_mode")
        judge_source = provenance.get("judge_source")
        if fallback_mode or (judge_source and str(judge_source).endswith("_proxy")):
            issues.append(GateIssue(episode_id, "fallback", str(fallback_mode or judge_source)))

        if labeler_name == "subtask_segmenter":
            payload = _read_payload(row.get("label_payload_path"))
            issues.extend(_check_subtasks(episode_id, payload))

        if labeler_name == "episode_metadata":
            payload = _read_payload(row.get("label_payload_path"))
            metadata = _parse_json(row.get("metadata_payload_json")) or payload.get("metadata", {})
            quality = _safe_int(metadata.get("quality"))
            if quality is not None:
                quality_values.append(quality)
            issues.extend(_check_metadata(episode_id, metadata))

    quality_counts = {int(k): int(v) for k, v in pd.Series(quality_values, dtype="int64").value_counts().sort_index().to_dict().items()}
    if len(quality_counts) <= 2 and len(quality_values) >= 8:
        issues.append(
            GateIssue(
                "__dataset__",
                "score_dispersion",
                f"quality scores collapsed to {sorted(quality_counts)} across {len(quality_values)} episodes",
            )
        )

    return GateReport(
        passed=not issues,
        issues=tuple(issues),
        episode_count=int(len(episodes)),
        checked_episode_count=int(labels["episode_id"].nunique()) if not labels.empty else 0,
        quality_counts=quality_counts,
    )


def _check_subtasks(episode_id: str, payload: dict[str, Any]) -> list[GateIssue]:
    issues: list[GateIssue] = []
    segments = payload.get("segments", [])
    texts = [str(segment.get("subtask_text", "")).strip().lower() for segment in segments if segment.get("subtask_text")]
    if texts and len(set(texts)) < len(texts):
        issues.append(GateIssue(episode_id, "repeated_subtask_text", "duplicate subtask text within episode"))

    observations = payload.get("stage_one_observations")
    observation_text = json.dumps(observations or {}, sort_keys=True).lower()
    task_text = str(payload.get("task", "")).lower()
    for text in texts:
        for token in _object_like_tokens(text):
            if token not in observation_text and token not in task_text:
                issues.append(GateIssue(episode_id, "object_grounding", f"{token!r} not present in stage-one observation"))
                break
    return issues


def _check_metadata(episode_id: str, metadata: dict[str, Any]) -> list[GateIssue]:
    issues: list[GateIssue] = []
    quality = _safe_int(metadata.get("quality"))
    mistake = metadata.get("mistake")
    reason = str(metadata.get("reason", "")).strip()
    lower = reason.lower()
    if not reason or lower in {"one short sentence", "specific evidence consistent with the numeric score"}:
        issues.append(GateIssue(episode_id, "metadata_reason", "placeholder or missing reason"))
    if quality is None:
        issues.append(GateIssue(episode_id, "metadata_quality", "missing quality"))
        return issues
    success_words = {"success", "successfully", "completed", "clean", "placed", "resting", "inside"}
    failure_words = {"fail", "failed", "wrong", "incomplete", "mistake", "incorrect", "drop"}
    says_success = _has_unnegated_word(lower, success_words)
    says_failure = _has_unnegated_word(lower, failure_words)
    if quality <= 2 and says_success and not says_failure:
        issues.append(GateIssue(episode_id, "score_reason_consistency", "low quality paired with success reason"))
    if quality >= 4 and says_failure and not says_success:
        issues.append(GateIssue(episode_id, "score_reason_consistency", "high quality paired with failure reason"))
    if bool(mistake) and quality >= 5:
        issues.append(GateIssue(episode_id, "score_reason_consistency", "mistake=true with perfect quality"))
    return issues


def _object_like_tokens(text: str) -> list[str]:
    stop = {
        "after",
        "approach",
        "before",
        "carry",
        "destination",
        "edge",
        "from",
        "grasp",
        "leave",
        "lift",
        "move",
        "object",
        "place",
        "pickup",
        "release",
        "sink",
        "task",
        "that",
        "toward",
        "withdraw",
        "with",
    }
    tokens = [token.strip(".,:;()[]{}").lower() for token in text.split()]
    return [token for token in tokens if len(token) >= 4 and token not in stop]


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


def _read_payload(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    return _parse_json(path.read_text(encoding="utf-8"))


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


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
