from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.scoring import metadata_quality


GPT55_INPUT_PER_M = 5.00
GPT55_CACHED_INPUT_PER_M = 0.50
GPT55_OUTPUT_PER_M = 30.00
GPT55_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.5"
GEMINI_FLASH_INPUT_PER_M = 0.30
GEMINI_FLASH_OUTPUT_PER_M = 2.50
GEMINI_FLASH_LITE_INPUT_PER_M = 0.10
GEMINI_FLASH_LITE_OUTPUT_PER_M = 0.40
GEMINI_PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"


def summarize_cost(
    snapshot_id: str,
    data_root: str | Path | None = None,
    projection_sizes: tuple[int, ...] = (200, 1000, 60000),
) -> dict[str, Any]:
    root = resolve_data_root(data_root)
    snapshot_path = root / "snapshots" / snapshot_id
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
    episodes = pd.read_parquet(snapshot_path / "episodes.parquet")
    labels = pd.read_parquet(snapshot_path / "labels.parquet")
    manifest = _read_json(snapshot_path / "manifest.json")
    token_totals = _raw_token_totals(snapshot_path / "raw_vlm_outputs")
    episode_count = int(len(episodes))
    wall_clock = float(sum(float(v) for v in manifest.get("labeler_runtime_seconds", {}).values()))
    pricing = _pricing_for_totals(token_totals)
    total_cost = _estimate_cost(token_totals, pricing)
    per_episode_cost = total_cost / episode_count if episode_count else 0.0
    per_episode_seconds = wall_clock / episode_count if episode_count else 0.0
    report = {
        "snapshot_id": snapshot_id,
        "episode_count": episode_count,
        "label_rows": int(len(labels)),
        "token_totals": token_totals,
        "pricing": {
            "model": pricing["model"],
            "backend": pricing["backend"],
            "input_usd_per_1m": pricing["input_usd_per_1m"],
            "cached_input_usd_per_1m": pricing.get("cached_input_usd_per_1m"),
            "output_usd_per_1m": pricing["output_usd_per_1m"],
            "source": pricing["source"],
        },
        "estimated_total_cost_usd": round(total_cost, 6),
        "estimated_cost_per_episode_usd": round(per_episode_cost, 6),
        "wall_clock_seconds_total": round(wall_clock, 6),
        "wall_clock_seconds_per_episode": round(per_episode_seconds, 6),
        "quality_distribution": _quality_distribution(labels),
        "projections": [
            {
                "episodes": int(n),
                "estimated_cost_usd": round(per_episode_cost * n, 2),
                "estimated_wall_clock_hours_serial": round(per_episode_seconds * n / 3600.0, 2),
            }
            for n in projection_sizes
        ],
        "stop_rule": "Cost probe only. Do not label larger slices until Kevin explicitly approves the target N.",
    }
    (snapshot_path / "cost_probe_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def format_cost_report(report: dict[str, Any]) -> str:
    lines = [
        f"Cost probe: {report['snapshot_id']}",
        f"Episodes: {report['episode_count']}",
        f"Label rows: {report['label_rows']}",
        f"Estimated total cost: ${report['estimated_total_cost_usd']:.6f}",
        f"Estimated cost per episode: ${report['estimated_cost_per_episode_usd']:.6f}",
        f"Wall-clock total: {report['wall_clock_seconds_total']:.2f}s",
        f"Wall-clock per episode: {report['wall_clock_seconds_per_episode']:.2f}s",
        f"Quality distribution: {report['quality_distribution']}",
        "Token totals:",
        f"- input: {report['token_totals']['input_tokens']}",
        f"- cached input: {report['token_totals']['cached_input_tokens']}",
        f"- output: {report['token_totals']['output_tokens']}",
        f"- total: {report['token_totals']['total_tokens']}",
        f"Backend/model: {report['pricing']['backend']} / {report['pricing']['model']}",
        "Projected serial labeling:",
    ]
    for row in report["projections"]:
        lines.append(
            f"- {row['episodes']} episodes: ${row['estimated_cost_usd']:.2f}, "
            f"{row['estimated_wall_clock_hours_serial']:.2f} hours"
        )
    lines.append(f"Pricing source: {report['pricing']['source']}")
    lines.append(report["stop_rule"])
    return "\n".join(lines)


def _raw_token_totals(raw_root: Path) -> dict[str, Any]:
    totals = {
        "request_count": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    if not raw_root.exists():
        return totals
    for path in raw_root.glob("*/*.json"):
        data = _read_json(path)
        response = data.get("response_json", {})
        backend = str(data.get("backend", ""))
        model = str(data.get("model", ""))
        if backend:
            totals.setdefault("backends", {})
            totals["backends"][backend] = int(totals["backends"].get(backend, 0)) + 1
        if model:
            totals.setdefault("models", {})
            totals["models"][model] = int(totals["models"].get(model, 0)) + 1
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        gemini_usage = response.get("usageMetadata", {}) if isinstance(response, dict) else {}
        if usage:
            totals["request_count"] += 1
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
            reasoning = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
            totals["input_tokens"] += input_tokens
            totals["cached_input_tokens"] += cached
            totals["output_tokens"] += output_tokens
            totals["reasoning_tokens"] += reasoning
            totals["total_tokens"] += int(usage.get("total_tokens") or input_tokens + output_tokens)
            continue
        if not gemini_usage:
            continue
        totals["request_count"] += 1
        input_tokens = int(gemini_usage.get("promptTokenCount") or 0)
        output_tokens = int(gemini_usage.get("candidatesTokenCount") or 0)
        reasoning = int(gemini_usage.get("thoughtsTokenCount") or 0)
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["reasoning_tokens"] += reasoning
        totals["total_tokens"] += int(gemini_usage.get("totalTokenCount") or input_tokens + output_tokens + reasoning)
    return totals


def _pricing_for_totals(totals: dict[str, Any]) -> dict[str, Any]:
    models = totals.get("models", {}) if isinstance(totals.get("models"), dict) else {}
    backends = totals.get("backends", {}) if isinstance(totals.get("backends"), dict) else {}
    model = max(models, key=models.get) if models else "unknown"
    backend = max(backends, key=backends.get) if backends else "unknown"
    lower_model = str(model).lower()
    if "gemini" in lower_model or "gemini" in str(backend).lower():
        if "flash-lite" in lower_model:
            return {
                "backend": backend,
                "model": model,
                "input_usd_per_1m": GEMINI_FLASH_LITE_INPUT_PER_M,
                "output_usd_per_1m": GEMINI_FLASH_LITE_OUTPUT_PER_M,
                "bill_reasoning_separately": True,
                "source": GEMINI_PRICING_SOURCE,
            }
        return {
            "backend": backend,
            "model": model,
            "input_usd_per_1m": GEMINI_FLASH_INPUT_PER_M,
            "output_usd_per_1m": GEMINI_FLASH_OUTPUT_PER_M,
            "bill_reasoning_separately": True,
            "source": GEMINI_PRICING_SOURCE,
        }
    return {
        "backend": backend,
        "model": model or "gpt-5.5",
        "input_usd_per_1m": GPT55_INPUT_PER_M,
        "cached_input_usd_per_1m": GPT55_CACHED_INPUT_PER_M,
        "output_usd_per_1m": GPT55_OUTPUT_PER_M,
        "bill_reasoning_separately": False,
        "source": GPT55_PRICING_SOURCE,
    }


def _estimate_cost(totals: dict[str, Any], pricing: dict[str, Any]) -> float:
    cached = int(totals.get("cached_input_tokens") or 0)
    input_tokens = int(totals.get("input_tokens") or 0)
    billable_input = max(0, input_tokens - cached)
    output_tokens = int(totals.get("output_tokens") or 0)
    reasoning_tokens = int(totals.get("reasoning_tokens") or 0) if pricing.get("bill_reasoning_separately") else 0
    cached_price = pricing.get("cached_input_usd_per_1m")
    return (
        billable_input / 1_000_000 * float(pricing["input_usd_per_1m"])
        + cached / 1_000_000 * float(cached_price if cached_price is not None else pricing["input_usd_per_1m"])
        + (output_tokens + reasoning_tokens) / 1_000_000 * float(pricing["output_usd_per_1m"])
    )


def _quality_distribution(labels: pd.DataFrame) -> dict[int, int]:
    values: list[int] = []
    if labels.empty or "metadata_payload_json" not in labels.columns:
        return {}
    for value in labels.loc[labels["labeler_name"] == "episode_metadata", "metadata_payload_json"].dropna().tolist():
        try:
            data = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            continue
        quality = metadata_quality(data)
        if quality is not None:
            values.append(int(quality))
    if not values:
        return {}
    return {int(k): int(v) for k, v in pd.Series(values, dtype="int64").value_counts().sort_index().to_dict().items()}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BridgeEngine VLM label cost from raw Responses usage.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--projection", type=int, action="append", default=None)
    args = parser.parse_args()
    report = summarize_cost(
        args.snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        projection_sizes=tuple(args.projection or [200, 1000, 60000]),
    )
    print(format_cost_report(report))


if __name__ == "__main__":
    main()
