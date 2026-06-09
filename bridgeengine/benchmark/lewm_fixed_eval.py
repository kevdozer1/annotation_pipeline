from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bridgeengine.benchmark.train_lewm import WindowRecord
from bridgeengine.benchmark.window_eval import mean_or_nan, weighted_mean_or_nan, write_fixed_eval_windows


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
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config_snapshot.yaml"
    if cfg_path.exists():
        cfg_probe = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if cfg_probe.get("benchmark_paradigm") == "bridgeengine_pi07":
            from bridgeengine.benchmark.pi07_fixed import evaluate_pi07_run

            return evaluate_pi07_run(
                run_dir=run_dir,
                dataset_name=dataset_name,
                data_cache_dir=data_cache_dir,
                split_file=split_file,
                device=device,
            )
    _attach_lewm_src(lewm_root)
    import torch
    import stable_pretraining as spt
    from lewm_testbed.auxiliary.heads import AuxiliaryLeWM
    from stable_worldmodel.data.dataset import HDF5Dataset
    from stable_worldmodel.wm.lewm.module import Embedder
    from stable_worldmodel.wm.loss import SIGReg
    from stable_worldmodel.wm.utils import load_pretrained

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
    per_window_losses: list[float] = []
    sigreg_losses = []
    total_losses = []
    batch_weights: list[int] = []
    aux_losses: dict[str, list[float]] = {}
    records = _records_for_eval_order(
        dataset_name=dataset_name,
        data_cache_dir=data_cache_dir,
        split_file=split_file,
        history_size=history_size,
        num_preds=num_preds,
    )
    if len(records) != len(dataset):
        raise ValueError(
            f"Window identity record count does not match held-out dataset: "
            f"records={len(records)} dataset={len(dataset)}"
        )
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            batch_count = int(batch["pixels"].shape[0])
            output = model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]
            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            tgt_emb = emb[:, num_preds:]
            pred_emb = model.predict(ctx_emb, ctx_act)
            per_window = (pred_emb - tgt_emb).pow(2).mean(dim=(1, 2))
            pred_loss = per_window.mean()
            sigreg_loss = sigreg(emb.transpose(0, 1))
            base_loss = pred_loss + sigreg_weight * sigreg_loss
            aux = model.forward_aux(emb, batch)
            total = base_loss + aux["aux_total_loss"]
            pred_losses.append(float(pred_loss.item()))
            per_window_losses.extend(float(x) for x in per_window.detach().cpu().tolist())
            sigreg_losses.append(float(sigreg_loss.item()))
            total_losses.append(float(total.item()))
            batch_weights.append(batch_count)
            for key, value in aux.items():
                if key != "aux_total_loss" and "loss" in key:
                    aux_losses.setdefault(key, []).append(float(value.item()))
    windows_csv = write_fixed_eval_windows(
        run_dir,
        records,
        per_window_losses,
        history_size=history_size,
    )

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
        "latent_mse": mean_or_nan(per_window_losses),
        "pred_mse": mean_or_nan(per_window_losses),
        "sigreg_loss": weighted_mean_or_nan(sigreg_losses, batch_weights),
        "total_loss": weighted_mean_or_nan(total_losses, batch_weights),
        "fixed_eval_windows_csv": str(windows_csv),
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


def _records_for_eval_order(
    *,
    dataset_name: str,
    data_cache_dir: str | Path,
    split_file: str | Path | None,
    history_size: int,
    num_preds: int,
) -> list[WindowRecord]:
    data_cache_dir = Path(data_cache_dir)
    if split_file:
        cut_path = _infer_pi07_cut_path(dataset_name, data_cache_dir)
        if cut_path.exists():
            try:
                from bridgeengine.benchmark.pi07_fixed import _records_for_h5_order

                split_payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
                heldout_ids = [str(x) for x in split_payload.get("heldout_episode_ids", [])]
                return _records_for_h5_order(cut_path, heldout_ids, dataset_name, data_cache_dir)
            except Exception:
                pass
    return _basic_records_for_h5_order(dataset_name, data_cache_dir, history_size, num_preds)


def _infer_pi07_cut_path(dataset_name: str, data_cache_dir: Path) -> Path:
    # dataset names are be_h2h_scale_25_heldout / train.
    parts = str(dataset_name).split("_")
    scale = None
    for idx, part in enumerate(parts):
        if part == "scale" and idx + 1 < len(parts):
            scale = parts[idx + 1]
            break
    if scale is None:
        scale = next((part for part in parts if part.isdigit()), "")
    return data_cache_dir / "pi07_cuts" / f"pi07_scale_{scale}"


def _basic_records_for_h5_order(
    dataset_name: str,
    data_cache_dir: Path,
    history_size: int,
    num_preds: int,
) -> list[WindowRecord]:
    import h5py

    h5_path = data_cache_dir / "datasets" / f"{dataset_name}.h5"
    records: list[WindowRecord] = []
    with h5py.File(h5_path, "r") as f:
        ids = json.loads(f.attrs["bridgeengine_selected_episode_ids_json"])
        ep_lens = [int(x) for x in f["ep_len"][:]]
    window_size = int(history_size) + int(num_preds)
    for episode_id, ep_len in zip(ids, ep_lens):
        for start_idx in range(0, int(ep_len) - window_size + 1):
            records.append(
                WindowRecord(
                    episode_id=str(episode_id),
                    start_idx=int(start_idx),
                    task=str(episode_id),
                    subtask_text="",
                    segment_idx=None,
                    metadata={},
                    subgoal_image_path=None,
                )
            )
    return records


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
