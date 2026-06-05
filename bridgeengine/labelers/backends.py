from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from .moondream_client import API_URL as MOONDREAM_API_URL
from .moondream_client import _image_to_data_url, make_contact_sheet

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_VLM_MODEL = "gpt-5.5"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_VLM_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class BackendResponse:
    answer: str
    raw: dict[str, Any]
    backend_name: str
    model: str
    elapsed_seconds: float
    estimated_cost_usd: float | None = None


class VisionLanguageBackend(Protocol):
    name: str
    model: str

    def query_contact_sheet(
        self,
        frames,
        keyframe_indices: list[int],
        question: str,
        raw_output_path: Path,
    ) -> BackendResponse:
        ...


def build_vlm_backend(
    backend_name: str | None = None,
    model: str | None = None,
) -> VisionLanguageBackend:
    name = (backend_name or os.environ.get("BRIDGEENGINE_VLM_BACKEND") or "openai").strip().lower()
    if name in {"openai", "gpt", "responses"}:
        return OpenAIResponsesBackend(model=model or os.environ.get("BRIDGEENGINE_VLM_MODEL") or DEFAULT_OPENAI_VLM_MODEL)
    if name in {"moondream", "moondream_api"}:
        return MoondreamAPIBackend(model=model or os.environ.get("MOONDREAM_MODEL") or "vikhyatk/moondream2")
    if name in {"gemini", "google", "google_gemini"}:
        return GeminiGenerateContentBackend(
            model=model or os.environ.get("GEMINI_MODEL") or os.environ.get("BRIDGEENGINE_VLM_MODEL") or DEFAULT_GEMINI_VLM_MODEL
        )
    if name in {"mock", "test"}:
        return MockVisionLanguageBackend(model=model or "mock-vlm")
    raise ValueError(f"Unknown VLM backend {backend_name!r}; expected openai, gemini, moondream, or mock")


class OpenAIResponsesBackend:
    name = "openai_responses"

    def __init__(self, model: str = DEFAULT_OPENAI_VLM_MODEL, timeout_seconds: float = 120.0):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = _load_secret("OPENAI_API_KEY", ".secrets/openai_api_key.txt")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it in the shell or .secrets/openai_api_key.txt, "
                "choose --vlm-backend moondream, or pass --allow-fallback for scaffolding-only runs."
            )

    def query_contact_sheet(
        self,
        frames,
        keyframe_indices: list[int],
        question: str,
        raw_output_path: Path,
    ) -> BackendResponse:
        image_url = _image_to_data_url(make_contact_sheet(frames, keyframe_indices))
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {"type": "input_image", "image_url": image_url, "detail": "high"},
                    ],
                }
            ],
            "text": {"format": {"type": "text"}},
        }
        raw_record: dict[str, Any] = {
            "backend": self.name,
            "model": self.model,
            "endpoint": OPENAI_RESPONSES_URL,
            "question": question,
            "keyframe_indices": keyframe_indices,
            "request_image_note": "base64 omitted from raw provenance",
        }
        t0 = time.perf_counter()
        try:
            response = requests.post(
                OPENAI_RESPONSES_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_seconds,
            )
            elapsed = time.perf_counter() - t0
            raw_record["status_code"] = response.status_code
            raw_record["elapsed_seconds"] = elapsed
            raw_record["response_text"] = response.text
            response.raise_for_status()
            data = response.json()
            raw_record["response_json"] = data
            answer = _extract_openai_text(data)
            return BackendResponse(answer, raw_record, self.name, self.model, elapsed, _estimate_openai_cost(data))
        finally:
            _write_raw(raw_output_path, raw_record)


class MoondreamAPIBackend:
    name = "moondream_api"

    def __init__(self, model: str = "vikhyatk/moondream2", timeout_seconds: float = 90.0):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = _load_secret("MOONDREAM_API_KEY", ".secrets/moondream_api_key.txt")
        if not self.api_key:
            raise RuntimeError(
                "MOONDREAM_API_KEY is not set. Run scripts/set_moondream_key.ps1, "
                "set the env var, or pass --allow-fallback for scaffolding-only runs."
            )

    def query_contact_sheet(
        self,
        frames,
        keyframe_indices: list[int],
        question: str,
        raw_output_path: Path,
    ) -> BackendResponse:
        request_payload = {
            "image_url": _image_to_data_url(make_contact_sheet(frames, keyframe_indices)),
            "question": question,
        }
        raw_record: dict[str, Any] = {
            "backend": self.name,
            "model": self.model,
            "endpoint": MOONDREAM_API_URL,
            "question": question,
            "keyframe_indices": keyframe_indices,
            "request_image_note": "base64 omitted from raw provenance",
        }
        t0 = time.perf_counter()
        try:
            response = requests.post(
                MOONDREAM_API_URL,
                headers={"Content-Type": "application/json", "X-Moondream-Auth": self.api_key},
                json=request_payload,
                timeout=self.timeout_seconds,
            )
            elapsed = time.perf_counter() - t0
            raw_record["status_code"] = response.status_code
            raw_record["elapsed_seconds"] = elapsed
            raw_record["response_text"] = response.text
            response.raise_for_status()
            data = response.json()
            raw_record["response_json"] = data
            return BackendResponse(str(data.get("answer", "")), raw_record, self.name, self.model, elapsed, None)
        finally:
            _write_raw(raw_output_path, raw_record)


class GeminiGenerateContentBackend:
    name = "gemini_generate_content"

    def __init__(self, model: str = DEFAULT_GEMINI_VLM_MODEL, timeout_seconds: float = 120.0):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = (
            _load_secret("GEMINI_API_KEY", ".secrets/gemini_api_key.txt")
            or _load_secret("GOOGLE_API_KEY", ".secrets/google_api_key.txt")
        )
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set. Run scripts/set_gemini_key.ps1, "
                "set the env var, or pass --allow-fallback for scaffolding-only runs."
            )

    def query_contact_sheet(
        self,
        frames,
        keyframe_indices: list[int],
        question: str,
        raw_output_path: Path,
    ) -> BackendResponse:
        image = make_contact_sheet(frames, keyframe_indices)
        image_data = _image_to_data_url(image).split(",", 1)[1]
        endpoint = GEMINI_GENERATE_URL.format(model=self.model)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": question},
                        {"inlineData": {"mimeType": "image/jpeg", "data": image_data}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        raw_record: dict[str, Any] = {
            "backend": self.name,
            "model": self.model,
            "endpoint": endpoint,
            "question": question,
            "keyframe_indices": keyframe_indices,
            "request_image_note": "base64 omitted from raw provenance",
        }
        t0 = time.perf_counter()
        try:
            response = requests.post(
                endpoint,
                params={"key": self.api_key},
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_seconds,
            )
            elapsed = time.perf_counter() - t0
            raw_record["status_code"] = response.status_code
            raw_record["elapsed_seconds"] = elapsed
            raw_record["response_text"] = response.text
            response.raise_for_status()
            data = response.json()
            raw_record["response_json"] = data
            answer = _extract_gemini_text(data)
            return BackendResponse(answer, raw_record, self.name, self.model, elapsed, _estimate_gemini_cost(data, self.model))
        finally:
            _write_raw(raw_output_path, raw_record)


class MockVisionLanguageBackend:
    name = "mock_vlm"

    def __init__(self, model: str = "mock-vlm"):
        self.model = model

    def query_contact_sheet(
        self,
        frames,
        keyframe_indices: list[int],
        question: str,
        raw_output_path: Path,
    ) -> BackendResponse:
        t0 = time.perf_counter()
        answer = _mock_answer(question)
        raw = {
            "backend": self.name,
            "model": self.model,
            "question": question,
            "keyframe_indices": keyframe_indices,
            "response_json": {"answer": answer},
            "elapsed_seconds": 0.0,
        }
        _write_raw(raw_output_path, raw)
        return BackendResponse(answer, raw, self.name, self.model, time.perf_counter() - t0, 0.0)


def _load_secret(env_name: str, secret_path: str) -> str | None:
    value = os.environ.get(env_name)
    if value:
        return _clean_secret(value)
    path = Path(secret_path)
    if path.exists():
        return _clean_secret(path.read_text(encoding="utf-8-sig"))
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith(f"{env_name}="):
                return _clean_secret(line.split("=", 1)[1])
    return None


def _clean_secret(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lstrip("\ufeff").strip().strip('"').strip("'")
    return cleaned or None


def _extract_openai_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks)


def _estimate_openai_cost(data: dict[str, Any]) -> float | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    # Model pricing moves quickly; store token counts as raw provenance and avoid
    # a hard-coded claim unless the caller supplies a pricing layer later.
    return None


def _extract_gemini_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if part.get("text"):
                chunks.append(str(part["text"]))
    return "\n".join(chunks)


def _estimate_gemini_cost(data: dict[str, Any], model: str) -> float | None:
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    prices = _gemini_model_prices(model)
    if prices is None:
        return None
    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0)
    thoughts_tokens = int(usage.get("thoughtsTokenCount") or 0)
    return input_tokens / 1_000_000 * prices["input"] + (output_tokens + thoughts_tokens) / 1_000_000 * prices["output"]


def _gemini_model_prices(model: str) -> dict[str, float] | None:
    normalized = model.lower()
    if "flash-lite" in normalized:
        return {"input": 0.10, "output": 0.40}
    if "2.5-flash" in normalized or "gemini-2.5-flash" in normalized:
        return {"input": 0.30, "output": 2.50}
    return None


def _mock_answer(question: str) -> str:
    if "return exactly one object per interval" in question.lower() or '"segments"' in question:
        return json.dumps(
            {
                "segments": [
                    {"segment_idx": 0, "subtask_text": "approach and grasp object"},
                    {"segment_idx": 1, "subtask_text": "carry object to destination"},
                    {"segment_idx": 2, "subtask_text": "place object at destination"},
                ]
            }
        )
    if "quality" in question.lower() and "mistake" in question.lower():
        return json.dumps(
            {
                "quality": 4,
                "mistake": False,
                "reason": "The robot completes the requested placement with minor uncertainty in the final pose.",
            }
        )
    if "physical observation" in question.lower() or "observe" in question.lower():
        return json.dumps(
            {
                "observations": [
                    {
                        "segment_idx": 0,
                        "objects": ["object", "destination"],
                        "gripper": "open then closes",
                        "motion": "robot moves toward the object",
                        "summary": "The gripper approaches and grasps the object.",
                    },
                    {
                        "segment_idx": 1,
                        "objects": ["object", "destination"],
                        "gripper": "closed",
                        "motion": "robot carries the object",
                        "summary": "The robot transports the object toward the destination.",
                    },
                    {
                        "segment_idx": 2,
                        "objects": ["object", "destination"],
                        "gripper": "opens",
                        "motion": "robot releases the object",
                        "summary": "The robot places the object at the destination.",
                    },
                ],
                "episode_summary": "Robot picks up an object and places it at the requested destination.",
            }
        )
    return json.dumps({"answer": "mock"})


def _write_raw(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
