from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FAMILIES = (
    "baseline",
    "rich_text",
    "rich_text_metadata",
    "rich_text_metadata_subgoal",
)

FAMILY_LABELERS = {
    "baseline": (),
    "rich_text": ("subtask_segmenter",),
    "rich_text_metadata": ("subtask_segmenter", "episode_metadata"),
    "rich_text_metadata_subgoal": ("subtask_segmenter", "episode_metadata", "subgoal_images"),
}

HISTORY_SIZE = 3
WINDOW_SIZE = HISTORY_SIZE + 1
HASH_FEATURE_DIM = 128
METADATA_FEATURE_DIM = 8
FEATURE_DIM = HASH_FEATURE_DIM + METADATA_FEATURE_DIM
IMAGE_SIZE = 224
DEFAULT_EPOCHS = int(os.environ.get("BRIDGEENGINE_LEWM_EPOCHS", "8"))
DEFAULT_BATCH_SIZE = int(os.environ.get("BRIDGEENGINE_LEWM_BATCH_SIZE", "12"))
DEFAULT_LR = float(os.environ.get("BRIDGEENGINE_LEWM_LR", "3e-4"))
SUBTASK_DROPOUT = 0.30
METADATA_DROPOUT = 0.15
METADATA_COMPONENT_DROPOUT = 0.05
SUBGOAL_TRAIN_KEEP = 0.25
SPLIT_FILE = Path(__file__).with_name("fixed_split_13.json")
DEFAULT_STABLE_WORLDMODEL_PATH = Path(r"C:\Users\Kevin\projects\upstream\stable-worldmodel")
DEFAULT_PRETRAINED_PATH = Path(
    r"D:\hf_cache\models--quentinll--lewm-cube\snapshots"
    r"\7d05e023b3c1114cc8e803ec23fb0177d688598b\weights.pt"
)


@dataclass(frozen=True)
class WindowRecord:
    episode_id: str
    start_idx: int
    task: str
    subtask_text: str
    segment_idx: int | None
    metadata: dict[str, Any]
    subgoal_image_path: str | None


class LeWMBenchmarkError(RuntimeError):
    pass


def run_family_seed(
    cut_path: Path,
    family: str,
    seed: int,
    scale: int = 13,
    *,
    contract_smoke: bool = False,
) -> dict[str, Any]:
    """Run one BridgeEngine family/seed benchmark cell.

    The default path is a real LeWM latent-prediction run: load Kevin's local
    pretrained LeWM, freeze the visual encoder/projector, train the action
    encoder, predictor, prediction projection, and a small conditioning adapter
    on the fixed training split, then report held-out latent MSE.

    ``contract_smoke=True`` is reserved for tests and CI plumbing. It preserves
    the output shape without claiming a scientific result.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}")
    cut_path = Path(cut_path)
    if contract_smoke:
        return _run_contract_smoke(cut_path, family, seed, scale)
    return _run_real_lewm(cut_path, family, seed)


def _run_real_lewm(cut_path: Path, family: str, seed: int) -> dict[str, Any]:
    start = time.perf_counter()
    torch = _import_torch()
    _seed_everything(torch, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, heldout_ids, split_id = _load_split(cut_path)
    train_set = BridgeWindowDataset(cut_path, train_ids)
    heldout_set = BridgeWindowDataset(cut_path, heldout_ids)
    if not train_set.records:
        raise LeWMBenchmarkError("Fixed split produced zero training windows.")
    if not heldout_set.records:
        raise LeWMBenchmarkError("Fixed split produced zero held-out windows.")

    model, latent_dim, pretrained_path = _load_lewm(torch, device)
    conditioner = PromptConditioner(FEATURE_DIM, latent_dim).to(device)
    _configure_trainable_parameters(model)
    params = [p for p in model.parameters() if p.requires_grad] + list(conditioner.parameters())
    optimizer = torch.optim.AdamW(params, lr=DEFAULT_LR, weight_decay=1e-4)

    train_rng = random.Random(seed)
    for epoch in range(DEFAULT_EPOCHS):
        model.train()
        _keep_frozen_modules_eval(model)
        conditioner.train()
        epoch_indices = list(range(len(train_set.records)))
        train_rng.shuffle(epoch_indices)
        for batch_indices in _chunks(epoch_indices, DEFAULT_BATCH_SIZE):
            batch = train_set.batch(batch_indices, torch)
            loss = _batch_loss(
                model,
                conditioner,
                batch,
                family,
                torch,
                device,
                rng=random.Random(seed * 1_000_003 + epoch * 997 + batch_indices[0]),
                train=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

    latent_mse = _evaluate(model, conditioner, heldout_set, family, torch, device)
    seconds = time.perf_counter() - start
    return {
        "family": family,
        "seed": seed,
        "latent_mse": round(float(latent_mse), 8),
        "wall_clock_seconds_labeling": round(_estimate_label_cost(cut_path, family, len(train_ids) + len(heldout_ids)), 6),
        "wall_clock_seconds_training": round(seconds, 6),
        "benchmark_backend": "real_lewm_frozen_adapter",
        "split_id": split_id,
        "train_episode_count": len(train_ids),
        "heldout_episode_count": len(heldout_ids),
        "train_windows": len(train_set.records),
        "heldout_windows": len(heldout_set.records),
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LR,
        "device": str(device),
        "pretrained_path": str(pretrained_path),
        "conditioning_note": _conditioning_note(family),
    }


class BridgeWindowDataset:
    def __init__(self, cut_path: Path, episode_ids: list[str]):
        self.cut_path = Path(cut_path)
        self.episode_ids = list(episode_ids)
        self.episode_sources = _read_json(self.cut_path / "episode_sources.json")
        self.label_paths = _read_json(self.cut_path / "label_paths.json")
        self.frame_cache: dict[str, np.ndarray] = {}
        self.action_cache: dict[str, np.ndarray] = {}
        self.records: list[WindowRecord] = []
        for episode_id in self.episode_ids:
            self._add_episode(episode_id)

    def _add_episode(self, episode_id: str) -> None:
        sources = self.episode_sources[episode_id]
        frames = np.load(sources["frames"], mmap_mode="r")
        actions = np.load(sources["actions"], mmap_mode="r")
        steps = min(int(frames.shape[0]), int(actions.shape[0]))
        if steps < WINDOW_SIZE:
            return
        self.frame_cache[episode_id] = frames
        self.action_cache[episode_id] = actions
        task = _read_episode_task(sources.get("metadata"), episode_id)
        segments = _subtask_segments(self.label_paths.get(episode_id, {}))
        metadata = _episode_metadata(self.label_paths.get(episode_id, {}))
        subgoals = _subgoal_paths(self.label_paths.get(episode_id, {}))
        for start_idx in range(0, steps - WINDOW_SIZE + 1):
            active_step = start_idx + HISTORY_SIZE - 1
            segment = _active_segment(segments, active_step)
            segment_idx = _safe_int(segment.get("segment_idx"))
            self.records.append(
                WindowRecord(
                    episode_id=episode_id,
                    start_idx=start_idx,
                    task=task,
                    subtask_text=str(segment.get("subtask_text") or ""),
                    segment_idx=segment_idx,
                    metadata=metadata,
                    subgoal_image_path=subgoals.get(segment_idx),
                )
            )

    def batch(self, indices: list[int], torch) -> dict[str, Any]:
        records = [self.records[i] for i in indices]
        pixel_tensors = []
        action_tensors = []
        subgoal_tensors = []
        subgoal_mask = []
        for record in records:
            frames = self.frame_cache[record.episode_id][record.start_idx : record.start_idx + WINDOW_SIZE]
            actions = self.action_cache[record.episode_id][record.start_idx : record.start_idx + WINDOW_SIZE]
            pixel_tensors.append(_preprocess_frames(frames, torch))
            action_tensors.append(torch.as_tensor(np.asarray(actions).copy(), dtype=torch.float32))
            if record.subgoal_image_path and Path(record.subgoal_image_path).exists():
                subgoal_tensors.append(_preprocess_image(Path(record.subgoal_image_path), torch))
                subgoal_mask.append(1.0)
            else:
                subgoal_tensors.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32))
                subgoal_mask.append(0.0)
        return {
            "records": records,
            "pixels": torch.stack(pixel_tensors, dim=0),
            "action": torch.stack(action_tensors, dim=0),
            "subgoal_pixels": torch.stack(subgoal_tensors, dim=0),
            "subgoal_mask": torch.as_tensor(subgoal_mask, dtype=torch.float32),
        }


class PromptConditioner:
    def __new__(cls, feature_dim: int, latent_dim: int):
        torch = _import_torch()
        import torch.nn as nn

        class _PromptConditioner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.text = nn.Sequential(
                    nn.Linear(feature_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.GELU(),
                    nn.Linear(latent_dim, latent_dim),
                )
                self.subgoal = nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, latent_dim),
                    nn.GELU(),
                    nn.Linear(latent_dim, latent_dim),
                )
                nn.init.zeros_(self.text[-1].weight)
                nn.init.zeros_(self.text[-1].bias)
                nn.init.zeros_(self.subgoal[-1].weight)
                nn.init.zeros_(self.subgoal[-1].bias)

            def forward(self, features, subgoal_latent=None, subgoal_mask=None):
                condition = self.text(features)
                if subgoal_latent is not None and subgoal_mask is not None:
                    condition = condition + self.subgoal(subgoal_latent) * subgoal_mask[:, None]
                return condition

        return _PromptConditioner()


def _batch_loss(model, conditioner, batch: dict[str, Any], family: str, torch, device, rng: random.Random, train: bool):
    pixels = batch["pixels"].to(device, non_blocking=True)
    actions = batch["action"].to(device, non_blocking=True)
    encoded = model.encode({"pixels": pixels, "action": actions})
    emb = encoded["emb"]
    act_emb = encoded["act_emb"]
    subgoal_latent, subgoal_mask = _subgoal_condition(model, batch, family, torch, device, rng, train)
    features = _condition_features(batch["records"], family, rng, train, torch).to(device)
    condition = conditioner(features, subgoal_latent, subgoal_mask)
    pred = model.predict(emb[:, :HISTORY_SIZE] + condition[:, None, :], act_emb[:, :HISTORY_SIZE])
    target = emb[:, 1 : HISTORY_SIZE + 1].detach()
    return torch.nn.functional.mse_loss(pred, target)


def _evaluate(model, conditioner, dataset: BridgeWindowDataset, family: str, torch, device) -> float:
    model.eval()
    conditioner.eval()
    losses = []
    with torch.no_grad():
        for batch_indices in _chunks(list(range(len(dataset.records))), DEFAULT_BATCH_SIZE):
            batch = dataset.batch(batch_indices, torch)
            loss = _batch_loss(model, conditioner, batch, family, torch, device, rng=random.Random(0), train=False)
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def _subgoal_condition(model, batch: dict[str, Any], family: str, torch, device, rng: random.Random, train: bool):
    mask = batch["subgoal_mask"].to(device)
    if family != "rich_text_metadata_subgoal":
        return None, torch.zeros_like(mask)
    if train:
        keep = torch.as_tensor([1.0 if rng.random() < SUBGOAL_TRAIN_KEEP else 0.0 for _ in range(mask.shape[0])], device=device)
        mask = mask * keep
    if float(mask.sum().item()) == 0.0:
        dim = int(getattr(model.action_encoder, "emb_dim", 192))
        return torch.zeros(mask.shape[0], dim, device=device), mask
    subgoal_pixels = batch["subgoal_pixels"].to(device, non_blocking=True).unsqueeze(1)
    encoded = model.encode({"pixels": subgoal_pixels})
    return encoded["emb"][:, 0].detach(), mask


def _condition_features(records: list[WindowRecord], family: str, rng: random.Random, train: bool, torch):
    features = []
    for record in records:
        include_subtask = family in {"rich_text", "rich_text_metadata", "rich_text_metadata_subgoal"}
        if train and include_subtask and rng.random() < SUBTASK_DROPOUT:
            include_subtask = False
        include_metadata = family in {"rich_text_metadata", "rich_text_metadata_subgoal"}
        metadata = dict(record.metadata) if include_metadata else {}
        if train and include_metadata and rng.random() < METADATA_DROPOUT:
            metadata = {}
        elif train and include_metadata:
            metadata = _drop_metadata_components(metadata, rng)

        text_parts = [f"Task: {record.task}."]
        if include_subtask and record.subtask_text:
            text_parts.append(f"Subtask: {record.subtask_text}.")
        if metadata:
            text_parts.extend(_metadata_text_parts(metadata))
        text_vec = _hash_text(" ".join(text_parts), HASH_FEATURE_DIM)
        meta_vec = _metadata_vector(metadata)
        features.append(np.concatenate([text_vec, meta_vec]).astype(np.float32))
    return torch.as_tensor(np.stack(features, axis=0), dtype=torch.float32)


def _drop_metadata_components(metadata: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    kept = dict(metadata)
    for key in ("speed", "quality", "mistake"):
        if rng.random() < METADATA_COMPONENT_DROPOUT:
            kept.pop(key, None)
    return kept


def _metadata_text_parts(metadata: dict[str, Any]) -> list[str]:
    parts = []
    if metadata.get("speed") is not None:
        parts.append(f"Speed: {metadata['speed']}.")
    if metadata.get("quality") is not None:
        parts.append(f"Quality: {metadata['quality']}/5.")
    if metadata.get("mistake") is not None:
        parts.append(f"Mistake: {str(bool(metadata['mistake'])).lower()}.")
    if metadata.get("control_mode") is not None:
        parts.append(f"Control: {metadata['control_mode']}.")
    return parts


def _metadata_vector(metadata: dict[str, Any]) -> np.ndarray:
    speed = _safe_float(metadata.get("speed"))
    quality = _safe_float(metadata.get("quality"))
    mistake = metadata.get("mistake")
    control = str(metadata.get("control_mode") or "")
    return np.asarray(
        [
            0.0 if speed is None else min(speed / 100.0, 2.0),
            0.0 if quality is None else quality / 5.0,
            0.0 if mistake is None else float(bool(mistake)),
            1.0 if control == "joint" else 0.0,
            1.0 if control == "end_effector" else 0.0,
            1.0 if speed is not None else 0.0,
            1.0 if quality is not None else 0.0,
            1.0 if mistake is not None else 0.0,
        ],
        dtype=np.float32,
    )


def _hash_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 else -1.0
        vec[idx] += sign
    vec /= max(len(tokens) ** 0.5, 1.0)
    return vec


def _preprocess_frames(frames: np.ndarray, torch):
    arr = np.asarray(frames).copy()
    tensor = torch.as_tensor(arr, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    tensor = torch.nn.functional.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def _preprocess_image(path: Path, torch):
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(image)
    return _preprocess_frames(arr[None, ...], torch)[0]


def _load_lewm(torch, device):
    _prepare_stable_worldmodel_import()
    from stable_worldmodel.wm.lewm.module import Embedder
    from stable_worldmodel.wm.utils import load_pretrained

    pretrained_path = Path(os.environ.get("LEWM_PRETRAINED_PATH", str(DEFAULT_PRETRAINED_PATH)))
    if not pretrained_path.exists():
        raise LeWMBenchmarkError(
            "LEWM pretrained checkpoint is missing. Set LEWM_PRETRAINED_PATH to weights.pt. "
            f"Default checked path: {pretrained_path}"
        )
    model = load_pretrained(str(pretrained_path))
    latent_dim = int(getattr(model.action_encoder, "emb_dim", 192))
    if int(getattr(model.action_encoder, "input_dim", 7)) != 7:
        model.action_encoder = Embedder(input_dim=7, emb_dim=latent_dim)
    model.to(device)
    return model, latent_dim, pretrained_path


def _configure_trainable_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in ("action_encoder", "predictor", "pred_proj"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
    _keep_frozen_modules_eval(model)


def _keep_frozen_modules_eval(model) -> None:
    for module_name in ("encoder", "projector"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.eval()


def _import_torch():
    spec = importlib.util.find_spec("torch")
    if spec is None:
        raise LeWMBenchmarkError(
            "PyTorch is not installed in this venv. Install torch/torchvision and stable-worldmodel "
            "before running the real benchmark."
        )
    import torch

    return torch


def _prepare_stable_worldmodel_import() -> None:
    if importlib.util.find_spec("stable_worldmodel") is None:
        local = Path(os.environ.get("STABLE_WORLDMODEL_PATH", str(DEFAULT_STABLE_WORLDMODEL_PATH)))
        if local.exists():
            sys.path.insert(0, str(local))
    if importlib.util.find_spec("stable_worldmodel") is None:
        raise LeWMBenchmarkError(
            "stable_worldmodel is not importable. Install Kevin's local source with "
            "`pip install -e C:\\Users\\Kevin\\projects\\upstream\\stable-worldmodel[train]` "
            "or set STABLE_WORLDMODEL_PATH."
        )


def _seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_split(cut_path: Path) -> tuple[list[str], list[str], str]:
    cut_episode_ids = [
        line.strip()
        for line in (cut_path / "episode_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not SPLIT_FILE.exists():
        raise LeWMBenchmarkError(f"Missing checked-in fixed split: {SPLIT_FILE}")
    split = _read_json(SPLIT_FILE)
    train_ids = list(split["train_episode_ids"])
    heldout_ids = list(split["heldout_episode_ids"])
    missing = sorted((set(train_ids) | set(heldout_ids)) - set(cut_episode_ids))
    extras = sorted(set(cut_episode_ids) - (set(train_ids) | set(heldout_ids)))
    if missing or extras:
        raise LeWMBenchmarkError(
            "The real benchmark expects the checked-in 13-episode split. "
            f"Missing={missing}; extras={extras}. Use --allow-scaffolding-labels only for plumbing tests."
        )
    return train_ids, heldout_ids, str(split.get("split_id", "fixed_13_v1"))


def _subtask_segments(label_groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = label_groups.get("subtask_segmenter", [])
    if not rows:
        return []
    payload = _read_json(Path(rows[0]["payload_path"]))
    return list(payload.get("segments", []))


def _episode_metadata(label_groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = label_groups.get("episode_metadata", [])
    if not rows:
        return {}
    metadata_json = rows[0].get("metadata_payload_json")
    if metadata_json:
        return _parse_json(metadata_json)
    payload = _read_json(Path(rows[0]["payload_path"]))
    return dict(payload.get("metadata", {}))


def _subgoal_paths(label_groups: dict[str, list[dict[str, Any]]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in label_groups.get("subgoal_images", []):
        segment_idx = _safe_int(row.get("segment_idx"))
        path = row.get("subgoal_image_path")
        if segment_idx is not None and path:
            result[segment_idx] = str(path)
    return result


def _active_segment(segments: list[dict[str, Any]], step_idx: int) -> dict[str, Any]:
    for segment in segments:
        start = _safe_int(segment.get("start_step"))
        end = _safe_int(segment.get("end_step"))
        if start is not None and end is not None and start <= step_idx <= end:
            return segment
    return segments[-1] if segments else {"segment_idx": None, "subtask_text": "", "start_step": 0, "end_step": 10**9}


def _read_episode_task(metadata_path: str | None, episode_id: str) -> str:
    if metadata_path and Path(metadata_path).exists():
        data = _read_json(Path(metadata_path))
        for key in ("language_instruction", "task", "instruction"):
            if data.get(key):
                return str(data[key])
    return episode_id


def _estimate_label_cost(cut_path: Path, family: str, n_episodes: int) -> float:
    manifest = _read_json(cut_path / "manifest.json")
    labelers = FAMILY_LABELERS[family]
    if not labelers:
        return 0.0
    snapshot_root = cut_path.parents[1] / "snapshots" / manifest["snapshot_id"]
    snapshot_manifest = snapshot_root / "manifest.json"
    if snapshot_manifest.exists():
        data = _read_json(snapshot_manifest)
        runtimes = data.get("labeler_runtime_seconds", {})
        return sum(float(runtimes.get(name, 0.0)) for name in labelers)
    return 0.0


def _conditioning_note(family: str) -> str:
    notes = {
        "baseline": "task text hashed into a learned conditioning adapter",
        "rich_text": "task plus active VLM-derived subtask text with 30% subtask dropout during training",
        "rich_text_metadata": "rich text plus speed/quality/mistake/control metadata with pi0.7-style dropout",
        "rich_text_metadata_subgoal": "metadata prompt plus actual end-of-segment subgoal frame encoded by frozen LeWM",
    }
    return notes[family]


def _run_contract_smoke(cut_path: Path, family: str, seed: int, scale: int) -> dict[str, Any]:
    manifest = _read_json(Path(cut_path) / "manifest.json")
    label_paths = _read_json(Path(cut_path) / "label_paths.json")
    n_episodes = int(manifest.get("episode_count", len(label_paths)))
    noise = _stable_noise(f"{manifest['transform_hash']}:{family}:{seed}")
    latent_mse = round(float(0.2 + noise), 6)
    return {
        "family": family,
        "seed": seed,
        "latent_mse": latent_mse,
        "wall_clock_seconds_labeling": round(_estimate_label_cost(Path(cut_path), family, n_episodes), 6),
        "wall_clock_seconds_training": 0.0,
        "benchmark_backend": "contract_smoke_no_science",
        "split_id": "none",
        "train_episode_count": n_episodes,
        "heldout_episode_count": 0,
        "train_windows": 0,
        "heldout_windows": 0,
        "epochs": 0,
        "batch_size": 0,
        "learning_rate": 0.0,
        "device": "cpu",
        "pretrained_path": "",
        "conditioning_note": "CI/plumbing result only",
    }


def _stable_noise(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / 2**32
    return (value - 0.5) * 0.006


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
