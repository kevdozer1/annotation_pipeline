from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from bridgeengine.benchmark.head_to_head_runner import (
    DEFAULT_LEWM_ROOT,
    DEFAULT_PLAN_DIR,
    DEFAULT_PRETRAINED,
    DEFAULT_SEEDS,
    DEFAULT_SIZES,
    _lewm_config,
)
from bridgeengine.benchmark.train_lewm import (
    FEATURE_DIM,
    HISTORY_SIZE,
    METADATA_COMPONENT_DROPOUT,
    SUBGOAL_TRAIN_KEEP,
    SUBTASK_DROPOUT,
    PromptConditioner,
    WindowRecord,
    _active_segment,
    _chunks,
    _condition_features,
    _episode_metadata,
    _preprocess_image,
    _read_episode_task,
    _read_json,
    _safe_int,
    _subgoal_paths,
    _subtask_segments,
)
from bridgeengine.export.cut import export_cut
from bridgeengine.paths import data_root as resolve_data_root


PI07_CONDITIONS: dict[str, dict[str, Any]] = {
    "P0_pi07_baseline": {
        "family": "baseline",
        "label": "canonical native LeWM baseline",
        "note": (
            "No pi0.7 conditioning and no auxiliary heads. This intentionally "
            "uses the native LeWM CV trainer/evaluator so the baseline is "
            "comparable to A_baseline instead of a separate BridgeEngine adapter model."
        ),
    },
    "P_adapter_null": {
        "family": "adapter_null",
        "label": "conditioning adapter null control",
        "note": (
            "The BridgeEngine conditioning adapter is active, but the conditioning "
            "feature vector is zeroed and subgoal input is disabled. This isolates "
            "adapter overhead from useful pi0.7 annotation content."
        ),
    },
    "P1_pi07_subtask_text": {
        "family": "rich_text",
        "label": "pi0.7 subtask text",
        "note": "Task text plus active human-reviewed subtask text.",
    },
    "P2_pi07_metadata": {
        "family": "rich_text_metadata",
        "label": "pi0.7 metadata",
        "note": "Subtask text plus calibrated speed/quality/mistake/control metadata.",
    },
    "P3_pi07_subgoal": {
        "family": "rich_text_subgoal",
        "label": "pi0.7 subgoal",
        "note": "Subtask text plus the human-boundary-derived subgoal frame.",
    },
    "P4_pi07_full_stack": {
        "family": "rich_text_metadata_subgoal",
        "label": "pi0.7 full stack",
        "note": "Subtask text, metadata, and subgoal frame.",
    },
}


def prepare_pi07_manifest(
    snapshot_id: str,
    *,
    output_dir: str | Path = "D:/lewm_runs/bridgeengine_head_to_head/run_100",
    plan_dir: str | Path = DEFAULT_PLAN_DIR,
    data_root: str | Path | None = None,
    pretrained_path: str | Path = DEFAULT_PRETRAINED,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 5e-5,
    sigreg_weight: float = 0.09,
) -> dict[str, Any]:
    out = Path(output_dir).resolve()
    plan_dir = Path(plan_dir).resolve()
    root = resolve_data_root(data_root)
    cuts_root = out / "pi07_cuts"
    runs_root = out / "runs"
    configs_root = out / "configs"
    out.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    cuts: list[str] = []
    for size in sizes:
        split_file = plan_dir / "splits" / f"scale_{int(size)}_split.json"
        split = json.loads(split_file.read_text(encoding="utf-8"))
        episode_ids = sorted(set(split["train_episode_ids"]) | set(split["heldout_episode_ids"]))
        filter_sql = _episode_filter_sql(episode_ids)
        cut_name = f"pi07_scale_{int(size)}"
        export_cut(snapshot_id, filter_sql, cuts_root, cut_name, data_root=root)
        cut_path = cuts_root / cut_name
        cuts.append(str(cut_path))
        heldout_name = f"be_h2h_scale_{int(size)}_heldout"
        for condition_name, condition in PI07_CONDITIONS.items():
            for seed in seeds:
                run_dir = runs_root / f"scale_{int(size)}" / f"{condition_name}_seed{int(seed)}"
                if condition_name == "P0_pi07_baseline":
                    baseline_cfg = _lewm_config(
                        condition={"condition_name": condition_name, "auxiliary_heads": {}},
                        dataset_name=f"be_h2h_scale_{int(size)}_train",
                        data_cache_dir=out,
                        pretrained_path=str(pretrained_path),
                        max_epochs=int(max_epochs),
                        batch_size=int(batch_size),
                        lr=float(lr),
                    )
                    baseline_cfg["sigreg_weight"] = float(sigreg_weight)
                    cfg_path = configs_root / f"scale_{int(size)}_{condition_name}.yaml"
                    cfg_path.write_text(yaml.safe_dump(baseline_cfg, sort_keys=False), encoding="utf-8")
                    train_cmd = [
                        sys.executable,
                        str(Path(DEFAULT_LEWM_ROOT) / "scripts" / "finetune_with_aux.py"),
                        "--config",
                        str(cfg_path),
                        "--seed",
                        str(int(seed)),
                        "--output-dir",
                        str(run_dir),
                    ]
                    paradigm = "shared_native_baseline"
                    canonical_baseline = True
                else:
                    train_cmd = [
                        sys.executable,
                        "-m",
                        "bridgeengine.benchmark.pi07_fixed",
                        "train",
                        "--cut-path",
                        str(cut_path),
                        "--split-file",
                        str(split_file),
                        "--family",
                        condition["family"],
                        "--condition-name",
                        condition_name,
                        "--run-dir",
                        str(run_dir),
                        "--pretrained-path",
                        str(pretrained_path),
                        "--seed",
                        str(int(seed)),
                        "--max-epochs",
                        str(int(max_epochs)),
                        "--batch-size",
                        str(int(batch_size)),
                        "--lr",
                        str(float(lr)),
                        "--sigreg-weight",
                        str(float(sigreg_weight)),
                    ]
                    paradigm = "bridgeengine_pi07"
                    canonical_baseline = False
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "bridgeengine.benchmark.lewm_fixed_eval",
                    "--run-dir",
                    str(run_dir),
                    "--dataset-name",
                    heldout_name,
                    "--data-cache-dir",
                    str(out),
                    "--split-file",
                    str(split_file),
                    "--output-json",
                    str(run_dir / "fixed_eval.json"),
                ]
                commands.append(
                    {
                        "scale_n": int(size),
                        "condition": condition_name,
                        "family": condition["family"],
                        "label": condition["label"],
                        "paradigm": paradigm,
                        "canonical_baseline": canonical_baseline,
                        "seed": int(seed),
                        "run_dir": str(run_dir),
                        "cut_path": str(cut_path),
                        "split_file": str(split_file),
                        "train_cmd": train_cmd,
                        "eval_cmd": eval_cmd,
                    }
                )
    manifest = {
        "snapshot_id": snapshot_id,
        "output_dir": str(out),
        "plan_dir": str(plan_dir),
        "data_root": str(root),
        "pretrained_path": str(pretrained_path),
        "sizes": [int(x) for x in sizes],
        "seeds": [int(x) for x in seeds],
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "sigreg_weight": float(sigreg_weight),
        "conditions": PI07_CONDITIONS,
        "cuts": cuts,
        "commands": commands,
        "metric": "held-out next-latent MSE through bridgeengine.benchmark.lewm_fixed_eval",
        "mechanism_disclosure": (
            "BridgeEngine pi0.7 signals are conditioning inputs. The LeWM CV signals "
            "are auxiliary prediction targets. This runner compares annotation strategies, "
            "not a pure signal-content injection. The P0 baseline is a shared native LeWM "
            "baseline with no conditioning and no auxiliary heads."
        ),
    }
    manifest_path = out / "pi07_command_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def train_pi07_cell(
    *,
    cut_path: str | Path,
    split_file: str | Path,
    family: str,
    condition_name: str,
    run_dir: str | Path,
    pretrained_path: str | Path = DEFAULT_PRETRAINED,
    seed: int = 42,
    max_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 5e-5,
    sigreg_weight: float = 0.09,
    device: str = "cuda",
) -> dict[str, Any]:
    if family not in {
        "adapter_null",
        "baseline",
        "rich_text",
        "rich_text_metadata",
        "rich_text_subgoal",
        "rich_text_metadata_subgoal",
    }:
        raise ValueError(f"Unknown pi0.7 family: {family}")
    start = time.perf_counter()
    _seed_everything(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "benchmark_paradigm": "bridgeengine_pi07",
        "condition_name": condition_name,
        "pi07_family": family,
        "cut_path": str(Path(cut_path).resolve()),
        "split_file": str(Path(split_file).resolve()),
        "pretrained_path": str(pretrained_path),
        "action_dim": 7,
        "frameskip": 1,
        "history_size": HISTORY_SIZE,
        "num_preds": 1,
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "sigreg_weight": float(sigreg_weight),
        "gradient_clip": 1.0,
        "precision": "bf16-mixed",
        "seed": int(seed),
        "freeze": "none",
        "native_training_loop": True,
        "conditioning_mechanism": "add learned prompt/subgoal adapter to context latent before predict",
    }
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    train_ids, _, _ = _load_split_ids(cut_path, split_file)
    train_data = Pi07NativeWindowDataset.from_split(
        cut_path=Path(cut_path),
        split_file=Path(split_file),
        split_key="train_episode_ids",
        dataset_name=_dataset_name_from_split(split_file, "train"),
        data_cache_dir=_infer_data_cache_dir(split_file, run_dir),
        cfg=cfg,
    )
    if not train_data.records:
        raise ValueError("No train windows for pi0.7 fixed-split cell.")

    import lightning as pl
    import stable_pretraining as spt
    from stable_worldmodel.wm.loss import SIGReg

    pl.seed_everything(seed, workers=True)
    model, latent_dim = _load_base_lewm(pretrained_path, device_obj)
    conditioner = PromptConditioner(FEATURE_DIM, latent_dim)
    conditioned_model = Pi07ConditionedLeWM(
        base_model=model,
        conditioner=conditioner,
        family=family,
        seed=seed,
    ).to(device_obj)

    split_rng = torch.Generator().manual_seed(int(seed))
    train_set, val_set = spt.data.random_split(train_data, lengths=[0.9, 0.1], generator=split_rng)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        generator=split_rng,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    sigreg = SIGReg(knots=17, num_proj=1024)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {"type": "AdamW", "lr": float(lr), "weight_decay": 1e-3},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    forward_fn = partial(
        pi07_native_forward,
        history_size=HISTORY_SIZE,
        num_preds=1,
        sigreg_weight=float(sigreg_weight),
    )
    lit_module = spt.Module(
        model=conditioned_model,
        sigreg=sigreg,
        forward=forward_fn,
        optim=optimizers,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    trainer = pl.Trainer(
        max_epochs=int(max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        callbacks=[],
        num_sanity_val_steps=1,
        enable_checkpointing=False,
        logger=False,
        default_root_dir=str(run_dir),
    )
    trainer.fit(lit_module, data_module)
    ckpt_dir = run_dir / "checkpoints" / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": conditioned_model.base_model.state_dict(),
            "conditioner_state_dict": conditioned_model.conditioner.state_dict(),
            "latent_dim": int(latent_dim),
            "feature_dim": int(FEATURE_DIM),
            "family": family,
            "condition_name": condition_name,
        },
        ckpt_dir / "full_weights.pt",
    )
    elapsed = time.perf_counter() - start
    metadata = {
        "condition": condition_name,
        "family": family,
        "seed": int(seed),
        "train_episode_count": len(train_ids),
        "train_windows": len(train_data),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "training_time_sec": round(elapsed, 3),
        "device": str(device_obj),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "sigreg_weight": float(sigreg_weight),
        "native_training_loop": True,
        "optimizer": "AdamW",
        "weight_decay": 1e-3,
        "scheduler": "LinearWarmupCosineAnnealingLR",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def evaluate_pi07_run(
    *,
    run_dir: str | Path,
    dataset_name: str,
    data_cache_dir: str | Path,
    split_file: str | Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model, latent_dim = _load_base_lewm(cfg["pretrained_path"], device_obj)
    conditioner = PromptConditioner(FEATURE_DIM, latent_dim).to(device_obj)
    state = torch.load(run_dir / "checkpoints" / "final" / "full_weights.pt", map_location=device_obj, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    conditioner.load_state_dict(state["conditioner_state_dict"])
    model.eval()
    conditioner.eval()
    from stable_worldmodel.wm.loss import SIGReg

    sigreg = SIGReg(knots=17, num_proj=1024).to(device_obj)
    split_file_path = Path(split_file) if split_file else Path(cfg["split_file"])
    heldout_data = Pi07H5WindowDataset.from_split(
        cut_path=Path(cfg["cut_path"]),
        split_file=split_file_path,
        split_key="heldout_episode_ids",
        dataset_name=dataset_name,
        data_cache_dir=Path(data_cache_dir),
        cfg=cfg,
    )
    losses: list[float] = []
    sigreg_losses: list[float] = []
    total_losses: list[float] = []
    batch_size = int(cfg.get("batch_size", 16))
    with torch.no_grad():
        for batch_indices in _chunks(list(range(len(heldout_data))), batch_size):
            batch = heldout_data.batch(batch_indices)
            total, parts = pi07_batch_loss(
                model=model,
                conditioner=conditioner,
                batch=batch,
                family=cfg["pi07_family"],
                sigreg=sigreg,
                sigreg_weight=float(cfg.get("sigreg_weight", 0.09)),
                device=device_obj,
                rng=random.Random(0),
                train=False,
            )
            losses.append(float(parts["pred_loss"]))
            sigreg_losses.append(float(parts["sigreg_loss"]))
            total_losses.append(float(total.item()))
    split_payload = json.loads(split_file_path.read_text(encoding="utf-8"))
    return {
        "condition": str(cfg.get("condition_name", run_dir.name)),
        "family": str(cfg.get("pi07_family")),
        "paradigm": "bridgeengine_pi07",
        "run_dir": str(run_dir),
        "dataset_name": dataset_name,
        "data_cache_dir": str(data_cache_dir),
        "split_file": str(split_file_path),
        "split_id": split_payload.get("split_id"),
        "heldout_episode_count": len(split_payload.get("heldout_episode_ids", [])),
        "heldout_windows": len(heldout_data),
        "latent_mse": float(np.mean(losses)) if losses else float("nan"),
        "pred_mse": float(np.mean(losses)) if losses else float("nan"),
        "sigreg_loss": float(np.mean(sigreg_losses)) if sigreg_losses else float("nan"),
        "total_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
        "weights": str(run_dir / "checkpoints" / "final" / "full_weights.pt"),
        "device": str(device_obj),
        "conditioning_mechanism": cfg.get("conditioning_mechanism"),
    }


class Pi07H5WindowDataset:
    def __init__(
        self,
        *,
        dataset_name: str,
        data_cache_dir: str | Path,
        episode_ids: list[str],
        cut_path: Path,
        cfg: dict[str, Any],
    ) -> None:
        from stable_worldmodel.data.dataset import HDF5Dataset
        import stable_pretraining as spt

        self.dataset_name = dataset_name
        self.data_cache_dir = Path(data_cache_dir)
        self.cut_path = cut_path
        self.cfg = dict(cfg)
        self.dataset = HDF5Dataset(
            name=dataset_name,
            cache_dir=self.data_cache_dir,
            num_steps=int(cfg.get("history_size", HISTORY_SIZE)) + int(cfg.get("num_preds", 1)),
            frameskip=int(cfg.get("frameskip", 1)),
            keys_to_load=["pixels", "action", "observation"],
            keys_to_cache=["action", "observation"],
        )
        self.dataset.transform = _build_transform(self.dataset)
        self.records = _records_for_h5_order(cut_path, episode_ids, dataset_name, self.data_cache_dir)
        if len(self.records) != len(self.dataset):
            raise ValueError(
                f"Conditioning records do not match HDF5Dataset windows for {dataset_name}: "
                f"records={len(self.records)} dataset={len(self.dataset)}"
            )

    @classmethod
    def from_split(
        cls,
        *,
        cut_path: Path,
        split_file: Path,
        split_key: str,
        dataset_name: str,
        data_cache_dir: str | Path,
        cfg: dict[str, Any],
    ) -> "Pi07H5WindowDataset":
        split = json.loads(split_file.read_text(encoding="utf-8"))
        return cls(
            dataset_name=dataset_name,
            data_cache_dir=data_cache_dir,
            episode_ids=[str(x) for x in split[split_key]],
            cut_path=cut_path,
            cfg=cfg,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def batch(self, indices: list[int]) -> dict[str, Any]:
        from torch.utils.data._utils.collate import default_collate

        samples = [self.dataset[int(i)] for i in indices]
        batch = default_collate(samples)
        records = [self.records[int(i)] for i in indices]
        subgoal_tensors = []
        subgoal_mask = []
        for record in records:
            if record.subgoal_image_path and Path(record.subgoal_image_path).exists():
                subgoal_tensors.append(_preprocess_image(Path(record.subgoal_image_path), torch))
                subgoal_mask.append(1.0)
            else:
                subgoal_tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))
                subgoal_mask.append(0.0)
        batch["records"] = records
        batch["subgoal_pixels"] = torch.stack(subgoal_tensors, dim=0)
        batch["subgoal_mask"] = torch.as_tensor(subgoal_mask, dtype=torch.float32)
        return batch


class Pi07NativeWindowDataset:
    """HDF5Dataset wrapper that carries BridgeEngine conditioning fields.

    This preserves the native LeWM data path used by ``finetune_with_aux.py``:
    the base pixels/action/observation tensors come directly from
    ``stable_worldmodel.data.dataset.HDF5Dataset`` with the same transforms. The
    wrapper only appends small conditioning metadata and optional subgoal images.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        data_cache_dir: str | Path,
        episode_ids: list[str],
        cut_path: Path,
        cfg: dict[str, Any],
    ) -> None:
        from stable_worldmodel.data.dataset import HDF5Dataset

        self.dataset_name = dataset_name
        self.data_cache_dir = Path(data_cache_dir)
        self.cut_path = cut_path
        self.cfg = dict(cfg)
        self.dataset = HDF5Dataset(
            name=dataset_name,
            cache_dir=self.data_cache_dir,
            num_steps=int(cfg.get("history_size", HISTORY_SIZE)) + int(cfg.get("num_preds", 1)),
            frameskip=int(cfg.get("frameskip", 1)),
            keys_to_load=["pixels", "action", "observation"],
            keys_to_cache=["action", "observation"],
        )
        self.dataset.transform = _build_transform(self.dataset)
        self.records = _records_for_h5_order(cut_path, episode_ids, dataset_name, self.data_cache_dir)
        if len(self.records) != len(self.dataset):
            raise ValueError(
                f"Conditioning records do not match HDF5Dataset windows for {dataset_name}: "
                f"records={len(self.records)} dataset={len(self.dataset)}"
            )

    @classmethod
    def from_split(
        cls,
        *,
        cut_path: Path,
        split_file: Path,
        split_key: str,
        dataset_name: str,
        data_cache_dir: str | Path,
        cfg: dict[str, Any],
    ) -> "Pi07NativeWindowDataset":
        split = json.loads(split_file.read_text(encoding="utf-8"))
        return cls(
            dataset_name=dataset_name,
            data_cache_dir=data_cache_dir,
            episode_ids=[str(x) for x in split[split_key]],
            cut_path=cut_path,
            cfg=cfg,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[int(index)])
        record = self.records[int(index)]
        sample["_be_episode_id"] = record.episode_id
        sample["_be_start_idx"] = torch.as_tensor(record.start_idx, dtype=torch.long)
        sample["_be_task"] = record.task
        sample["_be_subtask_text"] = record.subtask_text
        sample["_be_segment_idx"] = torch.as_tensor(-1 if record.segment_idx is None else record.segment_idx, dtype=torch.long)
        sample["_be_metadata_json"] = json.dumps(record.metadata, sort_keys=True)
        sample["_be_subgoal_image_path"] = record.subgoal_image_path or ""
        if record.subgoal_image_path and Path(record.subgoal_image_path).exists():
            sample["subgoal_pixels"] = _preprocess_image(Path(record.subgoal_image_path), torch)
            sample["subgoal_mask"] = torch.as_tensor(1.0, dtype=torch.float32)
        else:
            sample["subgoal_pixels"] = torch.zeros(3, 224, 224, dtype=torch.float32)
            sample["subgoal_mask"] = torch.as_tensor(0.0, dtype=torch.float32)
        return sample


class Pi07ConditionedLeWM(torch.nn.Module):
    def __init__(self, *, base_model, conditioner, family: str, seed: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.conditioner = conditioner
        self.family = family
        self.condition_rng = random.Random(int(seed))

    def encode(self, info: dict[str, Any]) -> dict[str, Any]:
        return self.base_model.encode(info)

    def predict(self, emb, act_emb):
        return self.base_model.predict(emb, act_emb)


def pi07_native_forward(self, batch, stage, history_size, num_preds, sigreg_weight):
    """Native LeWM Lightning forward with one conditioning-adapter insertion."""

    conditioned_model: Pi07ConditionedLeWM = self.model
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = conditioned_model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    records = _records_from_collated_batch(batch)
    train = str(stage).lower().startswith("train")
    rng = conditioned_model.condition_rng if train else random.Random(0)
    features = _pi07_condition_features(records, conditioned_model.family, rng, train).to(emb.device)
    subgoal_latent, subgoal_mask = _pi07_subgoal_condition(
        conditioned_model.base_model,
        batch,
        conditioned_model.family,
        emb.device,
        rng,
        train,
    )
    condition = conditioned_model.conditioner(features, subgoal_latent, subgoal_mask)
    ctx_emb = emb[:, :history_size] + condition[:, None, :]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = conditioned_model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["base_loss"] = output["pred_loss"] + float(sigreg_weight) * output["sigreg_loss"]
    output["loss"] = output["base_loss"]
    losses_dict = {f"{stage}/{key}": value.detach() for key, value in output.items() if "loss" in key}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def pi07_batch_loss(
    *,
    model,
    conditioner,
    batch: dict[str, Any],
    family: str,
    sigreg,
    sigreg_weight: float,
    device,
    rng: random.Random,
    train: bool,
):
    pixels = batch["pixels"].to(device, non_blocking=True)
    actions = torch.nan_to_num(batch["action"].to(device, non_blocking=True), 0.0)
    output = model.encode({"pixels": pixels, "action": actions})
    emb = output["emb"]
    act_emb = output["act_emb"]
    features = _pi07_condition_features(batch["records"], family, rng, train).to(device)
    subgoal_latent, subgoal_mask = _pi07_subgoal_condition(model, batch, family, device, rng, train)
    condition = conditioner(features, subgoal_latent, subgoal_mask)
    history_size = HISTORY_SIZE
    pred_emb = model.predict(emb[:, :history_size] + condition[:, None, :], act_emb[:, :history_size])
    tgt_emb = emb[:, 1 : history_size + 1].detach()
    pred_loss = torch.nn.functional.mse_loss(pred_emb, tgt_emb)
    sigreg_loss = sigreg(emb.transpose(0, 1))
    total = pred_loss + float(sigreg_weight) * sigreg_loss
    return total, {"pred_loss": float(pred_loss.item()), "sigreg_loss": float(sigreg_loss.item())}


def run_pi07_manifest(
    manifest_path: str | Path,
    *,
    skip_existing: bool = True,
    max_cells: int | None = None,
    cleanup_epoch_checkpoints: bool = True,
    clean_stale_runs: bool = True,
) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cells = [
        cmd
        for cmd in manifest["commands"]
        if cmd.get("paradigm") in {"bridgeengine_pi07", "shared_native_baseline"}
    ]
    if max_cells is not None:
        cells = cells[: int(max_cells)]
    for cell in cells:
        run_dir = Path(cell["run_dir"])
        eval_json = run_dir / "fixed_eval.json"
        if cell.get("canonical_baseline") and eval_json.exists():
            existing = json.loads(eval_json.read_text(encoding="utf-8"))
            if existing.get("paradigm") == "bridgeengine_pi07":
                print(f"[clean invalid adapter baseline] {run_dir}")
                _safe_rmtree(run_dir)
                eval_json = run_dir / "fixed_eval.json"
        if cell.get("paradigm") == "bridgeengine_pi07" and eval_json.exists():
            cfg_path = run_dir / "config_snapshot.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
            if not cfg.get("native_training_loop"):
                print(f"[clean invalid custom-loop pi07 result] {run_dir}")
                _safe_rmtree(run_dir)
                eval_json = run_dir / "fixed_eval.json"
        if skip_existing and eval_json.exists():
            print(f"[skip] {eval_json}")
            continue
        if clean_stale_runs and run_dir.exists() and not eval_json.exists():
            print(f"[clean stale] {run_dir}")
            _safe_rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[train {cell.get('paradigm')}] "
            f"scale={cell['scale_n']} condition={cell['condition']} seed={cell['seed']}"
        )
        subprocess.run(cell["train_cmd"], check=True)
        print(f"[eval {cell.get('paradigm')}] {run_dir}")
        subprocess.run(cell["eval_cmd"], check=True)
        if cleanup_epoch_checkpoints:
            _cleanup_epoch_checkpoints(run_dir)


def summarize_pi07_results(output_dir: str | Path = "D:/lewm_runs/bridgeengine_head_to_head/run_100") -> dict[str, Any]:
    out = Path(output_dir)
    runs_root = out / "runs"
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("scale_*/*/fixed_eval.json")):
        run_name = path.parent.name
        if "_seed" not in run_name:
            continue
        condition_name, seed_text = run_name.rsplit("_seed", 1)
        if condition_name not in PI07_CONDITIONS:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "scale_n": int(path.parent.parent.name.replace("scale_", "")),
                "condition": condition_name,
                "family": payload.get("family") or PI07_CONDITIONS[condition_name]["family"],
                "label": PI07_CONDITIONS[condition_name]["label"],
                "paradigm": payload.get("paradigm") or "shared_native_baseline",
                "seed": int(seed_text),
                "latent_mse": float(payload["latent_mse"]),
                "heldout_windows": int(payload.get("heldout_windows", 0) or 0),
                "run_dir": str(path.parent),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No pi0.7 fixed_eval.json files found under {runs_root}")

    results = pd.DataFrame(rows).sort_values(["scale_n", "condition", "seed"]).reset_index(drop=True)
    for column in (
        "delta_vs_native_baseline",
        "delta_pct_vs_native_baseline",
        "delta_vs_adapter_null",
        "delta_pct_vs_adapter_null",
    ):
        results[column] = np.nan

    for (_, _), index_values in results.groupby(["scale_n", "seed"]).groups.items():
        idx = list(index_values)
        group = results.loc[idx]
        native = group[group["condition"] == "P0_pi07_baseline"]
        adapter = group[group["condition"] == "P_adapter_null"]
        native_mse = float(native["latent_mse"].iloc[0]) if not native.empty else float("nan")
        adapter_mse = float(adapter["latent_mse"].iloc[0]) if not adapter.empty else float("nan")
        if np.isfinite(native_mse):
            results.loc[idx, "delta_vs_native_baseline"] = results.loc[idx, "latent_mse"] - native_mse
            results.loc[idx, "delta_pct_vs_native_baseline"] = (
                (results.loc[idx, "latent_mse"] - native_mse) / native_mse * 100.0
            )
        if np.isfinite(adapter_mse):
            results.loc[idx, "delta_vs_adapter_null"] = results.loc[idx, "latent_mse"] - adapter_mse
            results.loc[idx, "delta_pct_vs_adapter_null"] = (
                (results.loc[idx, "latent_mse"] - adapter_mse) / adapter_mse * 100.0
            )

    detailed_csv = out / "pi07_results_with_deltas.csv"
    results.to_csv(detailed_csv, index=False)
    grouped = (
        results.groupby(["scale_n", "condition", "family", "label"], dropna=False)
        .agg(
            mean_latent_mse=("latent_mse", "mean"),
            std_latent_mse=("latent_mse", "std"),
            mean_delta_vs_native_baseline=("delta_vs_native_baseline", "mean"),
            mean_delta_pct_vs_native_baseline=("delta_pct_vs_native_baseline", "mean"),
            mean_delta_vs_adapter_null=("delta_vs_adapter_null", "mean"),
            mean_delta_pct_vs_adapter_null=("delta_pct_vs_adapter_null", "mean"),
            seeds_completed=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["scale_n", "mean_latent_mse"])
    )
    grouped_csv = out / "pi07_summary_by_condition.csv"
    grouped.to_csv(grouped_csv, index=False)

    missing: list[dict[str, Any]] = []
    observed = {(int(row.scale_n), str(row.condition), int(row.seed)) for row in results.itertuples(index=False)}
    manifest_path = out / "pi07_command_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for cell in manifest.get("commands", []):
            if cell.get("condition") not in PI07_CONDITIONS:
                continue
            key = (int(cell["scale_n"]), str(cell["condition"]), int(cell["seed"]))
            if key not in observed:
                missing.append({"scale_n": key[0], "condition": key[1], "seed": key[2]})
    return {
        "detailed_csv": str(detailed_csv),
        "grouped_csv": str(grouped_csv),
        "rows": int(len(results)),
        "missing_cells": missing,
    }


def _build_transform(dataset):
    import torch
    import stable_pretraining as spt

    imagenet_stats = spt.data.dataset_stats.ImageNet
    transforms = [
        spt.data.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels"),
        spt.data.transforms.Resize(224, source="pixels", target="pixels"),
    ]
    col_data = dataset.get_col_data("observation")
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    transforms.append(spt.data.transforms.WrapTorchTransform(norm_fn, source="observation", target="observation"))
    return spt.data.transforms.Compose(*transforms)


def _records_for_h5_order(cut_path: Path, expected_ids: list[str], dataset_name: str, data_cache_dir: Path) -> list[WindowRecord]:
    import h5py

    h5_path = data_cache_dir / "datasets" / f"{dataset_name}.h5"
    with h5py.File(h5_path, "r") as f:
        h5_ids = json.loads(f.attrs["bridgeengine_selected_episode_ids_json"])
        ep_lens = [int(x) for x in f["ep_len"][:]]
    expected_set = set(str(x) for x in expected_ids)
    if set(h5_ids) != expected_set:
        raise ValueError(f"HDF5 ids for {dataset_name} do not match split ids.")
    label_paths = _read_json(cut_path / "label_paths.json")
    episode_sources = _read_json(cut_path / "episode_sources.json")
    records: list[WindowRecord] = []
    for episode_id, ep_len in zip(h5_ids, ep_lens):
        task = _read_episode_task(episode_sources[episode_id].get("metadata"), episode_id)
        groups = label_paths.get(episode_id, {})
        segments = _subtask_segments(groups)
        metadata = _episode_metadata(groups)
        subgoals = _subgoal_paths(groups)
        for start_idx in range(0, ep_len - (HISTORY_SIZE + 1) + 1):
            active_step = start_idx + HISTORY_SIZE - 1
            segment = _active_segment(segments, active_step)
            segment_idx = _safe_int(segment.get("segment_idx"))
            records.append(
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
    return records


def _pi07_condition_features(records: list[WindowRecord], family: str, rng: random.Random, train: bool):
    if family == "adapter_null":
        return torch.zeros(len(records), FEATURE_DIM, dtype=torch.float32)
    mapped = "rich_text_metadata_subgoal" if family == "rich_text_subgoal" else family
    features = _condition_features(records, mapped, rng, train, torch)
    if family == "rich_text_subgoal":
        # The helper maps this through full-stack to include the subtask field;
        # remove metadata numerics by rebuilding with metadata-free records.
        metadata_free = [
            WindowRecord(
                episode_id=r.episode_id,
                start_idx=r.start_idx,
                task=r.task,
                subtask_text=r.subtask_text,
                segment_idx=r.segment_idx,
                metadata={},
                subgoal_image_path=r.subgoal_image_path,
            )
            for r in records
        ]
        features = _condition_features(metadata_free, "rich_text", rng, train, torch)
    return features


def _records_from_collated_batch(batch: dict[str, Any]) -> list[WindowRecord]:
    episode_ids = list(batch["_be_episode_id"])
    tasks = list(batch["_be_task"])
    subtasks = list(batch["_be_subtask_text"])
    metadata_json = list(batch["_be_metadata_json"])
    start_indices = batch["_be_start_idx"].detach().cpu().tolist()
    segment_indices = batch["_be_segment_idx"].detach().cpu().tolist()
    subgoal_paths = list(batch["_be_subgoal_image_path"])
    records: list[WindowRecord] = []
    for idx, episode_id in enumerate(episode_ids):
        segment_idx = int(segment_indices[idx])
        records.append(
            WindowRecord(
                episode_id=str(episode_id),
                start_idx=int(start_indices[idx]),
                task=str(tasks[idx]),
                subtask_text=str(subtasks[idx]),
                segment_idx=None if segment_idx < 0 else segment_idx,
                metadata=json.loads(metadata_json[idx]) if metadata_json[idx] else {},
                subgoal_image_path=str(subgoal_paths[idx]) if subgoal_paths[idx] else None,
            )
        )
    return records


def _pi07_subgoal_condition(model, batch: dict[str, Any], family: str, device, rng: random.Random, train: bool):
    mask = batch["subgoal_mask"].to(device)
    if family not in {"rich_text_subgoal", "rich_text_metadata_subgoal"}:
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


def _load_base_lewm(pretrained_path: str | Path, device):
    from stable_worldmodel.wm.lewm.module import Embedder
    from stable_worldmodel.wm.utils import load_pretrained

    model = load_pretrained(str(pretrained_path))
    latent_dim = int(getattr(model.action_encoder, "emb_dim", 192))
    if int(getattr(model.action_encoder, "input_dim", 7)) != 7:
        model.action_encoder = Embedder(input_dim=7, emb_dim=latent_dim)
    model.to(device)
    return model, latent_dim


def _load_split_ids(cut_path: str | Path, split_file: str | Path) -> tuple[list[str], list[str], str]:
    cut_ids = [
        line.strip()
        for line in (Path(cut_path) / "episode_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    train_ids = [str(x) for x in split["train_episode_ids"]]
    heldout_ids = [str(x) for x in split["heldout_episode_ids"]]
    missing = sorted((set(train_ids) | set(heldout_ids)) - set(cut_ids))
    extras = sorted(set(cut_ids) - (set(train_ids) | set(heldout_ids)))
    if missing or extras:
        raise ValueError(f"Cut/split mismatch. missing={missing} extras={extras}")
    return train_ids, heldout_ids, str(split.get("split_id"))


def _dataset_name_from_split(split_file: str | Path, kind: str) -> str:
    stem = Path(split_file).stem
    size = stem.split("_")[1]
    return f"be_h2h_scale_{size}_{kind}"


def _infer_data_cache_dir(split_file: str | Path, run_dir: str | Path) -> Path:
    # The preregistered split lives in the repo plan directory; the HDF5 exports
    # live under the shared run_100 output root. The run dir is
    # <out>/runs/scale_N/condition_seed.
    return Path(run_dir).resolve().parents[2]


def _episode_filter_sql(episode_ids: list[str]) -> str:
    quoted = ", ".join("'" + episode_id.replace("'", "''") + "'" for episode_id in sorted(episode_ids))
    return f"e.episode_id IN ({quoted})"


def _cleanup_epoch_checkpoints(run_dir: Path) -> None:
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.exists():
        return
    for epoch_dir in checkpoints.glob("epoch_*"):
        if epoch_dir.is_dir():
            _safe_rmtree(epoch_dir)


def _safe_rmtree(path: Path) -> None:
    target = path.resolve()
    allowed_roots = [Path.cwd().resolve(), Path("D:/lewm_runs").resolve(), Path(tempfile.gettempdir()).resolve()]
    if not any(str(target).lower().startswith(str(root).lower()) for root in allowed_roots):
        raise ValueError(f"Refusing to remove unexpected path: {target}")
    shutil.rmtree(target)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BridgeEngine pi0.7 conditions on fixed head-to-head splits.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--snapshot", required=True)
    prepare.add_argument("--output-dir", default="D:/lewm_runs/bridgeengine_head_to_head/run_100")
    prepare.add_argument("--plan-dir", default=str(DEFAULT_PLAN_DIR))
    prepare.add_argument("--data-root", default=None)
    prepare.add_argument("--pretrained-path", default=DEFAULT_PRETRAINED)
    prepare.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    prepare.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    prepare.add_argument("--max-epochs", type=int, default=20)
    prepare.add_argument("--batch-size", type=int, default=16)
    prepare.add_argument("--lr", type=float, default=5e-5)
    prepare.add_argument("--sigreg-weight", type=float, default=0.09)

    train = sub.add_parser("train")
    train.add_argument("--cut-path", required=True)
    train.add_argument("--split-file", required=True)
    train.add_argument("--family", required=True)
    train.add_argument("--condition-name", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--pretrained-path", default=DEFAULT_PRETRAINED)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--sigreg-weight", type=float, default=0.09)
    train.add_argument("--device", default="cuda")

    run = sub.add_parser("run-manifest")
    run.add_argument("--manifest", required=True)
    run.add_argument("--no-skip-existing", action="store_true")
    run.add_argument("--max-cells", type=int, default=None)
    run.add_argument("--keep-epoch-checkpoints", action="store_true")
    run.add_argument("--keep-stale-runs", action="store_true")

    summary = sub.add_parser("summarize")
    summary.add_argument("--output-dir", default="D:/lewm_runs/bridgeengine_head_to_head/run_100")

    args = parser.parse_args()
    if args.cmd == "prepare":
        manifest = prepare_pi07_manifest(
            snapshot_id=args.snapshot,
            output_dir=args.output_dir,
            plan_dir=args.plan_dir,
            data_root=Path(args.data_root) if args.data_root else None,
            pretrained_path=args.pretrained_path,
            sizes=tuple(args.sizes),
            seeds=tuple(args.seeds),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigreg_weight=args.sigreg_weight,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.cmd == "train":
        result = train_pi07_cell(
            cut_path=args.cut_path,
            split_file=args.split_file,
            family=args.family,
            condition_name=args.condition_name,
            run_dir=args.run_dir,
            pretrained_path=args.pretrained_path,
            seed=args.seed,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigreg_weight=args.sigreg_weight,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.cmd == "run-manifest":
        run_pi07_manifest(
            args.manifest,
            skip_existing=not args.no_skip_existing,
            max_cells=args.max_cells,
            cleanup_epoch_checkpoints=not args.keep_epoch_checkpoints,
            clean_stale_runs=not args.keep_stale_runs,
        )
    elif args.cmd == "summarize":
        print(json.dumps(summarize_pi07_results(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
