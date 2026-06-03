from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw

API_URL = "https://api.moondream.ai/v1/query"


class MoondreamClient:
    def __init__(self, api_key: str | None = None, timeout_seconds: float = 90.0):
        self.api_key = _clean_api_key(api_key or _load_api_key())
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise RuntimeError(
                "MOONDREAM_API_KEY is not set. Set it in the shell, run scripts/set_moondream_key.ps1, "
                "or pass --allow-fallback for scaffolding-only tests."
            )

    def query_contact_sheet(
        self,
        frames,
        keyframe_indices: list[int],
        question: str,
        raw_output_path: Path,
    ) -> dict[str, Any]:
        image = make_contact_sheet(frames, keyframe_indices)
        image_url = _image_to_data_url(image)
        request_payload = {"image_url": image_url, "question": question}
        raw_record: dict[str, Any] = {
            "endpoint": API_URL,
            "question": question,
            "keyframe_indices": keyframe_indices,
            "request_image_note": "base64 omitted from raw provenance",
        }
        t0 = time.perf_counter()
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Content-Type": "application/json",
                    "X-Moondream-Auth": self.api_key,
                },
                json=request_payload,
                timeout=self.timeout_seconds,
            )
            raw_record["status_code"] = response.status_code
            raw_record["elapsed_seconds"] = time.perf_counter() - t0
            raw_record["response_text"] = response.text
            response.raise_for_status()
            data = response.json()
            raw_record["response_json"] = data
            return data
        finally:
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_output_path.write_text(json.dumps(raw_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_contact_sheet(frames, keyframe_indices: list[int], thumb_width: int = 224) -> Image.Image:
    images = []
    for frame_idx in keyframe_indices:
        frame = frames[min(max(frame_idx, 0), frames.shape[0] - 1)]
        img = Image.fromarray(frame.astype("uint8")).convert("RGB")
        ratio = thumb_width / img.width
        img = img.resize((thumb_width, int(img.height * ratio)))
        canvas = Image.new("RGB", (img.width, img.height + 24), "white")
        canvas.paste(img, (0, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f"frame {frame_idx}", fill=(0, 0, 0))
        images.append(canvas)
    if not images:
        return Image.new("RGB", (thumb_width, thumb_width), "white")
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    width = cols * images[0].width
    height = rows * images[0].height
    sheet = Image.new("RGB", (width, height), "white")
    for i, img in enumerate(images):
        x = (i % cols) * img.width
        y = (i // cols) * img.height
        sheet.paste(img, (x, y))
    return sheet


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _load_api_key() -> str | None:
    key = os.environ.get("MOONDREAM_API_KEY")
    if key:
        return key
    secret_file = Path(".secrets/moondream_api_key.txt")
    if secret_file.exists():
        text = secret_file.read_text(encoding="utf-8-sig")
        if text:
            return text
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MOONDREAM_API_KEY="):
                return line.split("=", 1)[1]
    return None


def _clean_api_key(key: str | None) -> str | None:
    if key is None:
        return None
    cleaned = key.strip().lstrip("\ufeff").strip().strip('"').strip("'")
    return cleaned or None
