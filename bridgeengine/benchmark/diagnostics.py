"""Eval-only diagnostics for the head-to-head grid.

These four diagnostics were added to answer "why are the effect sizes small?".
They train nothing and change no split/seed/checkpoint; every function loads an
existing trained checkpoint and runs forward passes on the fixed splits.

1. Trivial-baseline floor (`diagnose_run` -> floor fields): per-window MSE of a
   copy baseline (z_hat_{t+1} = z_t), a constant baseline (z_hat = mean target),
   and the trained predictor. Ratios trained/copy and trained/constant, and the
   implied R^2 = 1 - trained_MSE / Var(target), say what fraction of the one-step
   latent-prediction problem is solved by doing nothing.

2. Motion-conditioned error: bin held-out windows by frame-to-frame latent
   displacement ||z_{t+1} - z_t|| (the copy MSE is exactly that displacement
   energy) and report per-bin trained MSE and per-bin paired deltas.

3. Conditioning-channel magnitude (pi0.7 only): ||condition|| / ||ctx_emb||, the
   number of distinct conditioning feature vectors actually seen, and subgoal
   coverage at eval time.

4. Training-dynamics audit: per-epoch losses were not persisted (both trainers run
   with logger=False), so this reports the available substitute -- the trained
   predictor's train-split MSE vs held-out MSE (a generalization gap) per run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from bridgeengine.benchmark.idm import (
    ALL_CONDITIONS,
    _build_window_dataset,
    _chunks,
    _encode_batch,
    _scale_from_run_dir,
    _split_file_for_scale,
)
from bridgeengine.benchmark.train_lewm import FEATURE_DIM, HISTORY_SIZE


DEFAULT_OUTPUT_DIR = Path("D:/lewm_runs/bridgeengine_head_to_head/run_100")
PI07_CONDITIONS = {
    "P0_pi07_baseline",
    "P_adapter_null",
    "P1_pi07_subtask_text",
    "P2_pi07_metadata",
    "P3_pi07_subgoal",
    "P4_pi07_full_stack",
}
MOTION_QUANTILE_EDGES = (0.2, 0.4, 0.6, 0.8)
DIAG_WINDOWS_FILENAME = "diagnostics_windows.csv"
DIAG_RESULT_FILENAME = "diagnostics.json"


def diagnose_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda",
) -> dict[str, Any]:
    import torch

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    cfg = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    scale_n = _scale_from_run_dir(run_dir)
    seed = int(cfg.get("seed", 42))
    condition = str(cfg.get("condition_name", run_dir.name.rsplit("_seed", 1)[0]))
    history_size = int(cfg.get("history_size", HISTORY_SIZE))
    split_file = _split_file_for_scale(scale_n)
    cut_path = output_dir / "pi07_cuts" / f"pi07_scale_{scale_n}"
    batch_size = int(cfg.get("batch_size", 16))

    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model, latent_dim, predict_fn, condition_fn = _load_run(run_dir, cfg, device_obj)

    heldout_ds = _build_window_dataset(
        cut_path=cut_path,
        split_file=split_file,
        split_key="heldout_episode_ids",
        dataset_name=f"be_h2h_scale_{scale_n}_heldout",
        data_cache_dir=output_dir,
        cfg=cfg,
    )
    train_ds = _build_window_dataset(
        cut_path=cut_path,
        split_file=split_file,
        split_key="train_episode_ids",
        dataset_name=f"be_h2h_scale_{scale_n}_train",
        data_cache_dir=output_dir,
        cfg=cfg,
    )

    # ---- Held-out pass: trained / copy MSE, motion, conditioning, target store ----
    trained_mse: list[float] = []
    copy_mse: list[float] = []
    tgt_store: list[np.ndarray] = []  # per-window (T, D) targets for constant baseline
    episode_ids: list[str] = []
    start_idxs: list[int] = []
    cond_ctx_ratio: list[float] = []
    subgoal_masks: list[float] = []
    feature_hashes: list[str] = []
    with torch.no_grad():
        for chunk in _chunks(list(range(len(heldout_ds))), batch_size):
            batch = heldout_ds.batch(chunk)
            emb, act_emb, _actions = _encode_batch(model, batch, device_obj)
            tgt = emb[:, 1 : history_size + 1]  # (B, T, D)
            ctx = emb[:, :history_size]
            pred = predict_fn(emb, act_emb, batch)
            trained = (pred - tgt).pow(2).mean(dim=(1, 2))
            copy = (ctx - tgt).pow(2).mean(dim=(1, 2))
            trained_mse.extend(float(x) for x in trained.detach().cpu().tolist())
            copy_mse.extend(float(x) for x in copy.detach().cpu().tolist())
            tgt_store.extend(np.asarray(t) for t in tgt.float().cpu().numpy())
            for record in batch["records"]:
                episode_ids.append(str(record.episode_id))
                start_idxs.append(int(record.start_idx))
            subgoal_masks.extend(float(x) for x in batch["subgoal_mask"].detach().cpu().tolist())
            if condition_fn is not None:
                cond, features = condition_fn(emb, batch)
                cnorm = cond.norm(dim=-1)  # (B,)
                ctxnorm = ctx.norm(dim=-1).mean(dim=1)  # (B,)
                ratio = (cnorm / ctxnorm.clamp_min(1e-8))
                cond_ctx_ratio.extend(float(x) for x in ratio.detach().cpu().tolist())
                for row in features.detach().cpu().numpy():
                    feature_hashes.append(hashlib.sha1(np.ascontiguousarray(row).tobytes()).hexdigest())

    # constant baseline: z_hat = mean over all held-out target vectors (per dim)
    all_tgt = np.concatenate([t.reshape(-1, latent_dim) for t in tgt_store], axis=0)
    global_mean = all_tgt.mean(axis=0)  # (D,)
    target_variance = float(np.mean(all_tgt.var(axis=0)))
    constant_mse = [float(((t - global_mean) ** 2).mean()) for t in tgt_store]

    # ---- Train-split pass: trained MSE only (generalization gap) ----
    train_trained: list[float] = []
    with torch.no_grad():
        for chunk in _chunks(list(range(len(train_ds))), batch_size):
            batch = train_ds.batch(chunk)
            emb, act_emb, _actions = _encode_batch(model, batch, device_obj)
            tgt = emb[:, 1 : history_size + 1]
            pred = predict_fn(emb, act_emb, batch)
            train_trained.extend(
                float(x) for x in (pred - tgt).pow(2).mean(dim=(1, 2)).detach().cpu().tolist()
            )

    windows_df = pd.DataFrame(
        {
            "episode_id": episode_ids,
            "start_idx": start_idxs,
            "trained_mse": trained_mse,
            "copy_mse": copy_mse,
            "constant_mse": constant_mse,
            "motion": copy_mse,  # frame-to-frame latent displacement energy
        }
    )
    windows_df.to_csv(run_dir / DIAG_WINDOWS_FILENAME, index=False)

    mean_trained = float(np.mean(trained_mse))
    mean_copy = float(np.mean(copy_mse))
    mean_constant = float(np.mean(constant_mse))
    result = {
        "condition": condition,
        "scale_n": scale_n,
        "seed": seed,
        "paradigm": str(cfg.get("benchmark_paradigm", "lewm_cv_aux")),
        "heldout_windows": len(trained_mse),
        "mean_trained_mse": mean_trained,
        "mean_copy_mse": mean_copy,
        "mean_constant_mse": mean_constant,
        "target_variance": target_variance,
        "ratio_trained_over_copy": mean_trained / mean_copy if mean_copy else float("nan"),
        "ratio_trained_over_constant": mean_trained / mean_constant if mean_constant else float("nan"),
        "r2_vs_constant": 1.0 - (mean_trained / mean_constant) if mean_constant else float("nan"),
        "train_trained_mse": float(np.mean(train_trained)) if train_trained else float("nan"),
        "train_windows": len(train_trained),
        "generalization_gap": (float(np.mean(train_trained)) - mean_trained) if train_trained else float("nan"),
        "subgoal_coverage": float(np.mean(subgoal_masks)) if subgoal_masks else float("nan"),
    }
    if condition_fn is not None and cond_ctx_ratio:
        arr = np.asarray(cond_ctx_ratio, dtype=np.float64)
        result.update(
            {
                "cond_ctx_ratio_mean": float(arr.mean()),
                "cond_ctx_ratio_p50": float(np.quantile(arr, 0.5)),
                "cond_ctx_ratio_p90": float(np.quantile(arr, 0.9)),
                "distinct_feature_vectors": int(len(set(feature_hashes))),
            }
        )
    (run_dir / DIAG_RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_diagnostics_grid(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    scales: tuple[int, ...] = (25, 50),
    conditions: tuple[str, ...] | None = None,
    skip_existing: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    condition_filter = set(conditions or ALL_CONDITIONS)
    done: list[dict[str, Any]] = []
    skipped: list[str] = []
    missing: list[str] = []
    for scale_n in scales:
        scale_dir = output_dir / "runs" / f"scale_{int(scale_n)}"
        if not scale_dir.is_dir():
            continue
        for run_dir in sorted(scale_dir.glob("*_seed*")):
            condition = run_dir.name.rsplit("_seed", 1)[0]
            if condition not in condition_filter:
                continue
            if not (run_dir / "checkpoints" / "final" / "full_weights.pt").exists():
                missing.append(str(run_dir))
                continue
            if skip_existing and (run_dir / DIAG_RESULT_FILENAME).exists():
                skipped.append(str(run_dir))
                continue
            res = diagnose_run(run_dir, output_dir=output_dir, device=device)
            done.append({k: res.get(k) for k in ("scale_n", "condition", "seed", "ratio_trained_over_constant")})
            print(
                f"[diag] scale={res['scale_n']} {res['condition']} seed={res['seed']} "
                f"r2={res['r2_vs_constant']:.4f} trained/copy={res['ratio_trained_over_copy']:.3f}"
            )
    report = {"output_dir": str(output_dir), "scales": [int(x) for x in scales], "completed": done, "skipped": skipped, "missing": missing}
    (output_dir / "diagnostics_run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def summarize_diagnostics(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    scales: tuple[int, ...] = (25, 50),
    motion_conditions: tuple[str, ...] = ("E_depth_tracks", "P4_pi07_full_stack"),
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    scale_set = {int(x) for x in scales}
    run_rows = []
    window_frames = []
    for path in sorted((output_dir / "runs").glob(f"scale_*/*/{DIAG_RESULT_FILENAME}")):
        scale_n = int(path.parent.parent.name.replace("scale_", ""))
        if scale_n not in scale_set:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_rows.append(payload)
        wpath = path.parent / DIAG_WINDOWS_FILENAME
        if wpath.exists():
            wf = pd.read_csv(wpath)
            wf["scale_n"] = scale_n
            wf["condition"] = payload["condition"]
            wf["seed"] = int(payload["seed"])
            wf["episode_id"] = wf["episode_id"].astype(str)
            wf["start_idx"] = wf["start_idx"].astype(int)
            window_frames.append(wf)
    if not run_rows:
        raise FileNotFoundError(f"No {DIAG_RESULT_FILENAME} files found under {output_dir}")
    runs = pd.DataFrame(run_rows)

    # ---- trivial floor table (mean over seeds) ----
    floor = (
        runs.groupby(["scale_n", "condition"])
        .agg(
            seeds=("seed", "nunique"),
            mean_trained_mse=("mean_trained_mse", "mean"),
            mean_copy_mse=("mean_copy_mse", "mean"),
            mean_constant_mse=("mean_constant_mse", "mean"),
            ratio_trained_over_copy=("ratio_trained_over_copy", "mean"),
            ratio_trained_over_constant=("ratio_trained_over_constant", "mean"),
            r2_vs_constant=("r2_vs_constant", "mean"),
        )
        .reset_index()
        .sort_values(["scale_n", "condition"])
    )
    floor_csv = output_dir / "diagnostics_trivial_floor.csv"
    floor.to_csv(floor_csv, index=False)

    # ---- conditioning table (pi0.7 only) ----
    cond = runs[runs["condition"].isin(PI07_CONDITIONS)].copy()
    cond_cols = [
        "scale_n", "condition", "seed", "cond_ctx_ratio_mean", "cond_ctx_ratio_p50",
        "cond_ctx_ratio_p90", "distinct_feature_vectors", "subgoal_coverage",
    ]
    cond_cols = [c for c in cond_cols if c in cond.columns]
    conditioning = (
        cond[cond_cols]
        .groupby(["scale_n", "condition"])
        .agg(
            {
                **{c: "mean" for c in ("cond_ctx_ratio_mean", "cond_ctx_ratio_p50", "cond_ctx_ratio_p90", "subgoal_coverage") if c in cond_cols},
                **({"distinct_feature_vectors": "mean"} if "distinct_feature_vectors" in cond_cols else {}),
            }
        )
        .reset_index()
        .sort_values(["scale_n", "condition"])
    )
    conditioning_csv = output_dir / "diagnostics_conditioning.csv"
    conditioning.to_csv(conditioning_csv, index=False)

    # ---- training-dynamics table ----
    dyn = (
        runs.groupby(["scale_n", "condition"])
        .agg(
            mean_train_mse=("train_trained_mse", "mean"),
            mean_heldout_mse=("mean_trained_mse", "mean"),
            mean_gap=("generalization_gap", "mean"),
        )
        .reset_index()
    )
    dyn["gap_pct_of_heldout"] = dyn["mean_gap"] / dyn["mean_heldout_mse"] * 100.0
    dyn = dyn.sort_values(["scale_n", "condition"])
    dyn_csv = output_dir / "diagnostics_training_dynamics.csv"
    dyn.to_csv(dyn_csv, index=False)

    # ---- motion bins + paired delta vs P0 ----
    motion_csv = output_dir / "diagnostics_motion_bins.csv"
    motion_table = pd.DataFrame()
    low_motion_fraction: dict[int, float] = {}
    if window_frames:
        allw = pd.concat(window_frames, ignore_index=True)
        motion_rows = []
        for scale_n, scale_grp in allw.groupby("scale_n"):
            ref = scale_grp[scale_grp["condition"] == "P0_pi07_baseline"][
                ["seed", "episode_id", "start_idx", "motion", "trained_mse"]
            ].rename(columns={"motion": "ref_motion", "trained_mse": "ref_trained_mse"})
            if ref.empty:
                continue
            edges = list(np.quantile(ref["ref_motion"], MOTION_QUANTILE_EDGES))
            bounds = [-np.inf] + edges + [np.inf]
            ref = ref.copy()
            ref["motion_bin"] = pd.cut(ref["ref_motion"], bins=bounds, labels=False, include_lowest=True)
            low_motion_fraction[int(scale_n)] = float((ref["motion_bin"] == 0).mean())
            for condition in ["P0_pi07_baseline", *motion_conditions]:
                cgrp = scale_grp[scale_grp["condition"] == condition][
                    ["seed", "episode_id", "start_idx", "trained_mse"]
                ]
                merged = cgrp.merge(ref, on=["seed", "episode_id", "start_idx"], how="inner")
                if merged.empty:
                    continue
                merged["paired_delta_vs_p0"] = merged["trained_mse"] - merged["ref_trained_mse"]
                for b, bgrp in merged.groupby("motion_bin"):
                    motion_rows.append(
                        {
                            "scale_n": int(scale_n),
                            "condition": condition,
                            "motion_bin": int(b),
                            "windows": int(len(bgrp)),
                            "mean_motion": float(bgrp["ref_motion"].mean()),
                            "mean_trained_mse": float(bgrp["trained_mse"].mean()),
                            "mean_paired_delta_vs_p0": float(bgrp["paired_delta_vs_p0"].mean()),
                        }
                    )
        if motion_rows:
            motion_table = pd.DataFrame(motion_rows).sort_values(["scale_n", "condition", "motion_bin"])
            motion_table.to_csv(motion_csv, index=False)

    return {
        "trivial_floor_csv": str(floor_csv),
        "conditioning_csv": str(conditioning_csv),
        "training_dynamics_csv": str(dyn_csv),
        "motion_bins_csv": str(motion_csv) if not motion_table.empty else None,
        "low_motion_fraction_bin0": low_motion_fraction,
        "runs": int(len(runs)),
    }


# ---------------------------------------------------------------------------
# Loading: model + predict_fn + (pi0.7) condition_fn
# ---------------------------------------------------------------------------
def _load_run(
    run_dir: Path, cfg: dict[str, Any], device
) -> tuple[Any, int, Callable, Callable | None]:
    import random

    import torch

    weights = run_dir / "checkpoints" / "final" / "full_weights.pt"
    paradigm = cfg.get("benchmark_paradigm")
    history_size = int(cfg.get("history_size", HISTORY_SIZE))
    if paradigm == "bridgeengine_pi07":
        from bridgeengine.benchmark.pi07_fixed import (
            _load_base_lewm,
            _pi07_condition_features,
            _pi07_subgoal_condition,
        )
        from bridgeengine.benchmark.train_lewm import PromptConditioner

        model, latent_dim = _load_base_lewm(cfg["pretrained_path"], device)
        state = torch.load(weights, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        conditioner = PromptConditioner(FEATURE_DIM, latent_dim).to(device)
        conditioner.load_state_dict(state["conditioner_state_dict"])
        model.eval()
        conditioner.eval()
        family = str(cfg["pi07_family"])

        def _condition(emb, batch):
            features = _pi07_condition_features(batch["records"], family, random.Random(0), False).to(emb.device)
            subgoal_latent, subgoal_mask = _pi07_subgoal_condition(model, batch, family, emb.device, random.Random(0), False)
            cond = conditioner(features, subgoal_latent, subgoal_mask)
            return cond, features

        def predict_fn(emb, act_emb, batch):
            cond, _ = _condition(emb, batch)
            return model.predict(emb[:, :history_size] + cond[:, None, :], act_emb[:, :history_size])

        return model, latent_dim, predict_fn, _condition

    from bridgeengine.benchmark.lewm_fixed_eval import DEFAULT_LEWM_ROOT, _attach_lewm_src

    _attach_lewm_src(DEFAULT_LEWM_ROOT)
    from lewm_testbed.auxiliary.heads import AuxiliaryLeWM
    from stable_worldmodel.wm.lewm.module import Embedder
    from stable_worldmodel.wm.utils import load_pretrained

    base = load_pretrained(cfg["pretrained_path"])
    action_dim = int(cfg.get("action_dim", 7))
    frameskip = int(cfg.get("frameskip", 1))
    effective_action_dim = action_dim * frameskip
    embed_dim = base.projector.net[0].in_features if hasattr(base.projector, "net") else 192
    if base.action_encoder.input_dim != effective_action_dim:
        base.action_encoder = Embedder(input_dim=effective_action_dim, emb_dim=embed_dim)
    model = AuxiliaryLeWM(base, heads_config=cfg.get("auxiliary_heads", {}), freeze=cfg.get("freeze", "none"))
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    latent_dim = int(getattr(model.base_model.action_encoder, "emb_dim", 192))

    def predict_fn(emb, act_emb, batch):
        return model.predict(emb[:, :history_size], act_emb[:, :history_size])

    return model, latent_dim, predict_fn, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-only diagnostics for the head-to-head grid.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run.add_argument("--scales", type=int, nargs="+", default=[25, 50])
    run.add_argument("--conditions", nargs="+", default=None)
    run.add_argument("--device", default="cuda")
    run.add_argument("--force", action="store_true")

    one = sub.add_parser("diagnose-run")
    one.add_argument("--run-dir", required=True)
    one.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    one.add_argument("--device", default="cuda")

    summ = sub.add_parser("summarize")
    summ.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    summ.add_argument("--scales", type=int, nargs="+", default=[25, 50])

    args = parser.parse_args()
    if args.cmd == "run":
        print(json.dumps(run_diagnostics_grid(args.output_dir, scales=tuple(args.scales), conditions=tuple(args.conditions) if args.conditions else None, skip_existing=not args.force, device=args.device), indent=2, sort_keys=True))
    elif args.cmd == "diagnose-run":
        print(json.dumps(diagnose_run(args.run_dir, output_dir=args.output_dir, device=args.device), indent=2, sort_keys=True))
    elif args.cmd == "summarize":
        print(json.dumps(summarize_diagnostics(args.output_dir, scales=tuple(args.scales)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
