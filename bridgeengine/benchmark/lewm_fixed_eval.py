from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_LEWM_ROOT = Path("C:/Users/Kevin/projects/LeWM_testbed")


def evaluate_fixed_heldout(
    run_dir: str | Path,
    dataset_name: str,
    data_cache_dir: str | Path,
    split_file: str | Path | None = None,
    device: str = "cuda",
    lewm_root: str | Path = DEFAULT_LEWM_ROOT,
) -> dict[str, Any]:
    """Evaluate one native LeWM aux-head checkpoint on an explicit held-out HDF5.

    This intentionally avoids LeWM_testbed/scripts/evaluate_boring3d.py's
    per-seed random 90/10 split. The dataset passed here should already be the
    preregistered fixed held-out HDF5.
    """
    _attach_lewm_src(lewm_root)
    import torch
    import stable_pretraining as spt
    from lewm_testbed.auxiliary.heads import AuxiliaryLeWM
    from stable_worldmodel.data.dataset import HDF5Dataset
    from stable_worldmodel.wm.lewm.module import Embedder
    from stable_worldmodel.wm.loss import SIGReg
    from stable_worldmodel.wm.utils import load_pretrained

    run_dir = Path(run_dir)
    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_snapshot.yaml not found under {run_dir}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    base = load_pretrained(cfg["pretrained_path"])
    action_dim = int(cfg.get("action_dim", 7))
    frameskip = int(cfg.get("frameskip", 1))
    effective_action_dim = action_dim * frameskip
    embed_dim = base.projector.net[0].in_features if hasattr(base.projector, "net") else 192
    if base.action_encoder.input_dim != effective_action_dim:
        base.action_encoder = Embedder(input_dim=effective_action_dim, emb_dim=embed_dim)

    heads_config = cfg.get("auxiliary_heads", {})
    model = AuxiliaryLeWM(base, heads_config=heads_config, freeze=cfg.get("freeze", "none"))
    weights = _final_weights(run_dir)
    state = torch.load(weights, map_location=device_obj, weights_only=True)
    model.load_state_dict(state)
    model.to(device_obj)
    model.eval()

    history_size = int(cfg.get("history_size", 3))
    num_preds = int(cfg.get("num_preds", 1))
    num_steps = history_size + num_preds
    keys_to_load, keys_to_cache = _keys_for_heads(heads_config)
    dataset = HDF5Dataset(
        name=dataset_name,
        cache_dir=data_cache_dir,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
    )
    dataset.transform = _build_transform(dataset, keys_to_load, cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    sigreg = SIGReg(knots=17, num_proj=1024).to(device_obj)
    sigreg_weight = float(cfg.get("sigreg_weight", 0.09))
    pred_losses = []
    sigreg_losses = []
    total_losses = []
    aux_losses: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            output = model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            tgt_emb = emb[:, num_preds:]
            pred_emb = model.predict(ctx_emb, ctx_act)
            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            sigreg_loss = sigreg(emb.transpose(0, 1))
            base_loss = pred_loss + sigreg_weight * sigreg_loss
            aux = model.forward_aux(emb, batch)
            total = base_loss + aux["aux_total_loss"]
            pred_losses.append(float(pred_loss.item()))
            sigreg_losses.append(float(sigreg_loss.item()))
            total_losses.append(float(total.item()))
            for key, value in aux.items():
                if key != "aux_total_loss" and "loss" in key:
                    aux_losses.setdefault(key, []).append(float(value.item()))

    split_payload = None
    if split_file:
        split_payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "condition": str(cfg.get("condition_name", run_dir.name)),
        "run_dir": str(run_dir),
        "dataset_name": dataset_name,
        "data_cache_dir": str(data_cache_dir),
        "split_file": str(split_file) if split_file else None,
        "split_id": split_payload.get("split_id") if isinstance(split_payload, dict) else None,
        "heldout_episode_count": len(split_payload.get("heldout_episode_ids", [])) if isinstance(split_payload, dict) else None,
        "heldout_windows": len(dataset),
        "latent_mse": float(np.mean(pred_losses)) if pred_losses else float("nan"),
        "pred_mse": float(np.mean(pred_losses)) if pred_losses else float("nan"),
        "sigreg_loss": float(np.mean(sigreg_losses)) if sigreg_losses else float("nan"),
        "total_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
        "weights": str(weights),
        "device": str(device_obj),
    }
    for key, values in aux_losses.items():
        result[key] = float(np.mean(values)) if values else float("nan")
    return result


def _attach_lewm_src(lewm_root: str | Path) -> None:
    root = Path(lewm_root)
    src = root / "src"
    for path in (root, src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _final_weights(run_dir: Path) -> Path:
    final = run_dir / "checkpoints" / "final" / "full_weights.pt"
    if final.exists():
        return final
    epoch_dirs = sorted((run_dir / "checkpoints").glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    if epoch_dirs:
        candidate = epoch_dirs[-1] / "full_weights.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No full_weights.pt found under {run_dir}")


def _keys_for_heads(heads_config: dict[str, Any]) -> tuple[list[str], list[str]]:
    keys_to_load = ["pixels", "action", "observation"]
    keys_to_cache = ["action", "observation"]
    if heads_config.get("depth", {}).get("enabled"):
        keys_to_load.append("depth")
    if heads_config.get("tracks", {}).get("enabled"):
        keys_to_load.extend(["tracks", "track_visibility"])
    if heads_config.get("masked_depth", {}).get("enabled") or heads_config.get("weighted_depth", {}).get("enabled"):
        keys_to_load.extend(["depth", "object_mask"])
    if heads_config.get("object_centroid", {}).get("enabled"):
        keys_to_load.append("object_centroid")
        keys_to_cache.append("object_centroid")
    if heads_config.get("object_shape", {}).get("enabled"):
        keys_to_load.append("object_shape")
        keys_to_cache.append("object_shape")
    return list(dict.fromkeys(keys_to_load)), list(dict.fromkeys(keys_to_cache))


def _build_transform(dataset, keys_to_load: list[str], cfg: dict[str, Any]):
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
    mask_dilation = int(cfg.get("mask_dilation", 0) or 0)
    if mask_dilation > 0 and "object_mask" in keys_to_load:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mask_dilation + 1, 2 * mask_dilation + 1))

        def dilate_mask(mask_tensor):
            mask_np = mask_tensor.numpy().astype(np.uint8)
            dilated = np.stack([cv2.dilate(m, kernel) for m in mask_np])
            return torch.from_numpy(dilated.astype(bool))

        transforms.append(spt.data.transforms.WrapTorchTransform(dilate_mask, source="object_mask", target="object_mask"))
    return spt.data.transforms.Compose(*transforms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LeWM aux-head run on an explicit fixed held-out HDF5.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-cache-dir", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lewm-root", default=str(DEFAULT_LEWM_ROOT))
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    result = evaluate_fixed_heldout(
        run_dir=args.run_dir,
        dataset_name=args.dataset_name,
        data_cache_dir=args.data_cache_dir,
        split_file=args.split_file,
        device=args.device,
        lewm_root=args.lewm_root,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
