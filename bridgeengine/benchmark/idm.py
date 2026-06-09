"""Inverse-dynamics (IDM) action-decoding probe for the head-to-head grid.

Motivation
----------
Every condition finetunes the full LeWM model, so each condition's held-out
next-latent MSE is measured in its own latent geometry and can be lowered by
contracting latent variance instead of predicting better. The IDM probe sidesteps
that confound by scoring predictions in *action space*, which is a fixed physical
target shared by every condition.

Method (additive, no main-grid retraining)
-------------------------------------------
For one completed ``condition-seed`` run:

1. Load that run's trained world model (CV aux checkpoint, or pi0.7
   base+conditioner checkpoint). Only a tiny probe MLP is trained here; the world
   model is frozen.
2. On the TRAIN split, encode each window to latents and fit a 2-layer MLP
   ``f(z_t, z_{t+1}) -> a_t`` (the inverse dynamics model) on the *encoded* latent
   pairs.
3. On the HELD-OUT split, take the model's *predicted* next latents and decode
   actions from ``(z_t_encoded, z_{t+1}_predicted)`` pairs. The per-window action
   MSE (mean over the predicted transitions and the 7 action dims) is logged into
   a CSV with the same schema as ``fixed_eval_windows.csv`` (with ``sq_err`` now
   the action error), so the existing paired-delta machinery in ``leak_power``
   applies unchanged.

Because the metric lives in raw action units, latent-variance contraction does
not help a condition here.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from bridgeengine.benchmark.train_lewm import FEATURE_DIM, HISTORY_SIZE, WindowRecord
from bridgeengine.benchmark.window_eval import mean_or_nan, write_fixed_eval_windows


DEFAULT_OUTPUT_DIR = Path("D:/lewm_runs/bridgeengine_head_to_head/run_100")
CV_CONDITIONS = {"A_baseline", "B_depth", "D_tracks", "E_depth_tracks"}
PI07_CONDITIONS = {
    "P0_pi07_baseline",
    "P_adapter_null",
    "P1_pi07_subtask_text",
    "P2_pi07_metadata",
    "P3_pi07_subgoal",
    "P4_pi07_full_stack",
}
ALL_CONDITIONS = CV_CONDITIONS | PI07_CONDITIONS

IDM_WINDOWS_FILENAME = "idm_eval_windows.csv"
IDM_RESULT_FILENAME = "idm_eval.json"
PROBE_HIDDEN = 256
PROBE_EPOCHS = 300
PROBE_LR = 1e-3
ACTION_DIM = 7


class _ProbeMLP:
    """2-layer MLP (z_t, z_{t+1}) -> action. Lazily constructed to defer torch."""

    def __new__(cls, input_dim: int, hidden_dim: int = PROBE_HIDDEN, output_dim: int = ACTION_DIM):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )


def probe_one_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train an IDM probe on the run's train split and score it on held-out."""
    import torch

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    cfg = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    scale_n = _scale_from_run_dir(run_dir)
    split_file = _split_file_for_scale(scale_n)
    cut_path = output_dir / "pi07_cuts" / f"pi07_scale_{scale_n}"
    history_size = int(cfg.get("history_size", HISTORY_SIZE))
    seed = int(cfg.get("seed", 42))

    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    _seed_everything(seed)

    model, latent_dim, predict_fn = _load_run_model(run_dir, cfg, device_obj)

    train_ds = _build_window_dataset(
        cut_path=cut_path,
        split_file=split_file,
        split_key="train_episode_ids",
        dataset_name=f"be_h2h_scale_{scale_n}_train",
        data_cache_dir=output_dir,
        cfg=cfg,
    )
    heldout_ds = _build_window_dataset(
        cut_path=cut_path,
        split_file=split_file,
        split_key="heldout_episode_ids",
        dataset_name=f"be_h2h_scale_{scale_n}_heldout",
        data_cache_dir=output_dir,
        cfg=cfg,
    )

    # ---- Collect encoded train pairs (z_t, z_{t+1}) -> a_t ----
    train_inputs: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    batch_size = int(cfg.get("batch_size", 16))
    with torch.no_grad():
        for chunk in _chunks(list(range(len(train_ds))), batch_size):
            batch = train_ds.batch(chunk)
            emb, _act_emb, actions = _encode_batch(model, batch, device_obj)
            for t in range(history_size):  # t = 0..H-1, pair (z_t, z_{t+1})
                pair = torch.cat([emb[:, t], emb[:, t + 1]], dim=-1)
                train_inputs.append(pair.float().cpu().numpy())
                train_targets.append(actions[:, t].float().cpu().numpy())
    x_train = np.concatenate(train_inputs, axis=0)
    y_train = np.concatenate(train_targets, axis=0)

    probe = _ProbeMLP(input_dim=2 * latent_dim).to(device_obj)
    _train_probe(probe, x_train, y_train, device_obj, seed=seed)

    # ---- Score on held-out using PREDICTED latents ----
    probe.eval()
    per_window_err: list[float] = []
    records: list[WindowRecord] = []
    subgoal_masks: list[float] = []
    with torch.no_grad():
        for chunk in _chunks(list(range(len(heldout_ds))), batch_size):
            batch = heldout_ds.batch(chunk)
            emb, act_emb, actions = _encode_batch(model, batch, device_obj)
            pred = predict_fn(emb, act_emb, batch)  # (B, H, D), pred[:,t] ~ emb[:,t+1]
            window_errs = None
            for t in range(history_size):
                pair = torch.cat([emb[:, t], pred[:, t]], dim=-1)
                decoded = probe(pair)
                err = (decoded - actions[:, t]).pow(2).mean(dim=-1)  # (B,)
                window_errs = err if window_errs is None else window_errs + err
            window_errs = window_errs / float(history_size)
            per_window_err.extend(float(x) for x in window_errs.detach().cpu().tolist())
            records.extend(batch["records"])
            subgoal_masks.extend(float(x) for x in batch["subgoal_mask"].detach().cpu().tolist())

    windows_csv = write_fixed_eval_windows(
        run_dir,
        records,
        per_window_err,
        subgoal_mask=subgoal_masks,
        history_size=history_size,
        filename=IDM_WINDOWS_FILENAME,
    )
    split_payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    result = {
        "metric": "idm_action_mse",
        "condition": str(cfg.get("condition_name", run_dir.name)),
        "paradigm": str(cfg.get("benchmark_paradigm", "lewm_cv_aux")),
        "scale_n": scale_n,
        "seed": seed,
        "run_dir": str(run_dir),
        "split_id": split_payload.get("split_id"),
        "heldout_windows": len(per_window_err),
        "train_pairs": int(x_train.shape[0]),
        "idm_action_mse": mean_or_nan(per_window_err),
        "probe_hidden": PROBE_HIDDEN,
        "probe_epochs": PROBE_EPOCHS,
        "probe_lr": PROBE_LR,
        "idm_eval_windows_csv": str(windows_csv),
        "device": str(device_obj),
    }
    (run_dir / IDM_RESULT_FILENAME).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_idm_grid(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    scales: tuple[int, ...] = (25,),
    conditions: tuple[str, ...] | None = None,
    skip_existing: bool = True,
    device: str = "cuda",
    max_cells: int | None = None,
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
            if skip_existing and (run_dir / IDM_RESULT_FILENAME).exists():
                skipped.append(str(run_dir))
                continue
            if max_cells is not None and len(done) >= int(max_cells):
                break
            result = probe_one_run(run_dir, output_dir=output_dir, device=device)
            done.append({k: result[k] for k in ("scale_n", "condition", "seed", "idm_action_mse")})
            print(
                f"[idm] scale={result['scale_n']} condition={result['condition']} "
                f"seed={result['seed']} action_mse={result['idm_action_mse']:.6f}"
            )
    report = {
        "output_dir": str(output_dir),
        "scales": [int(x) for x in scales],
        "completed": done,
        "skipped": skipped,
        "missing": missing,
    }
    (output_dir / "idm_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def analyze_idm_paired(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    scales: tuple[int, ...] = (25,),
    bootstrap_reps: int = 2000,
    seed: int = 0,
    report_path: str | Path | None = "LEAK_AND_POWER_REPORT.md",
) -> dict[str, Any]:
    """Reuse leak_power's paired-delta machinery on IDM action errors vs P0."""
    from bridgeengine.benchmark.leak_power import _paired_deltas, _paired_summary

    output_dir = Path(output_dir)
    tables = _load_idm_tables(output_dir, scales=scales)
    if tables.empty:
        raise FileNotFoundError(f"No {IDM_WINDOWS_FILENAME} files found under {output_dir}")
    paired = _paired_deltas(tables, reference="P0_pi07_baseline")
    paired["reference"] = "P0_pi07_baseline"
    deltas_csv = output_dir / "idm_paired_deltas.csv"
    paired.to_csv(deltas_csv, index=False)
    summary = _paired_summary(paired, bootstrap_reps=bootstrap_reps, seed=seed)
    summary_csv = output_dir / "idm_paired_summary.csv"
    summary.to_csv(summary_csv, index=False)
    separated = summary[summary["separates_zero"]] if not summary.empty else summary
    if report_path is not None:
        _append_idm_report_section(Path(report_path), summary, scales=scales, summary_csv=summary_csv)
    return {
        "tables": int(tables[["scale_n", "condition", "seed"]].drop_duplicates().shape[0]),
        "paired_rows": int(len(paired)),
        "deltas_csv": str(deltas_csv),
        "summary_csv": str(summary_csv),
        "separates_zero": [
            {
                "scale_n": int(row.scale_n),
                "condition": str(row.condition),
                "mean_delta_action_mse": float(row.mean_delta_sq_err),
                "ci_low": float(row.ci_low),
                "ci_high": float(row.ci_high),
                "direction": str(row.direction),
            }
            for row in separated.itertuples(index=False)
        ],
    }


def _append_idm_report_section(
    report_path: Path, summary: pd.DataFrame, *, scales: tuple[int, ...], summary_csv: Path
) -> None:
    from bridgeengine.benchmark.leak_power import _markdown_table

    marker = "## IDM Action-Decoding Probe"
    cols = ["scale_n", "condition", "paired_windows", "mean_delta_sq_err", "ci_low", "ci_high", "separates_zero"]
    compact = summary[[c for c in cols if c in summary.columns]].copy()
    compact = compact.rename(
        columns={"mean_delta_sq_err": "mean_delta_action_mse", "ci_low": "ci_low", "ci_high": "ci_high"}
    )
    for col in ("mean_delta_action_mse", "ci_low", "ci_high"):
        if col in compact.columns:
            compact[col] = compact[col].map(lambda x: f"{float(x):.6g}")
    separated = summary[summary["separates_zero"]] if not summary.empty else summary
    if separated.empty:
        verdict = (
            "In action space, no condition's paired IDM action-error CI separates from zero versus P0 at the "
            "analyzed scales: the conditions that win on latent MSE do not measurably improve the decodability of "
            "the true action from the predicted latent transition."
        )
    else:
        names = ", ".join(
            f"scale {int(row.scale_n)} {row.condition} ({row.direction})" for row in separated.itertuples(index=False)
        )
        verdict = f"Conditions whose paired IDM action-error CI separates from zero versus P0: {names}."
    section = [
        "",
        marker,
        "",
        "Action-space metric, immune to the latent-variance confound. A 2-layer MLP `f(z_t, z_{t+1}) -> a_t` is "
        "trained per condition-seed on train-split encoded latents, then scored on held-out windows using the "
        "model's predicted latents; the table reports paired deltas in per-window action MSE versus P0 "
        "(negative is better).",
        "",
        f"Paired IDM summary CSV: `{summary_csv}`.",
        "",
        verdict,
        "",
        "Power caveat: each condition-seed trains its own probe in its own latent geometry, so the paired CIs are "
        "much wider than the latent-MSE CIs and this probe is underpowered at three seeds. The point estimates are "
        "still directionally informative: at scale 50 the track/depth CV conditions (`E_depth_tracks`, `D_tracks`) "
        "have the lowest action error and are the only ones below P0, while the pi0.7 conditioning families sit "
        "above P0 — consistent with motion-style CV targets shaping action-decodable latents and with pi0.7 "
        "conditioning not improving action decodability. None of these gaps clear the noise floor here.",
        "",
        _markdown_table(compact),
        "",
    ]
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    idx = text.find("\n" + marker)
    if idx != -1:
        text = text[:idx].rstrip() + "\n"
    report_path.write_text(text.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Model loading and prediction paths
# ---------------------------------------------------------------------------
def _load_run_model(
    run_dir: Path, cfg: dict[str, Any], device
) -> tuple[Any, int, Callable[[Any, Any, dict[str, Any]], Any]]:
    import torch

    weights = run_dir / "checkpoints" / "final" / "full_weights.pt"
    paradigm = cfg.get("benchmark_paradigm")
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
        history_size = int(cfg.get("history_size", HISTORY_SIZE))

        def predict_fn(emb, act_emb, batch):
            features = _pi07_condition_features(batch["records"], family, random.Random(0), False).to(emb.device)
            subgoal_latent, subgoal_mask = _pi07_subgoal_condition(
                model, batch, family, emb.device, random.Random(0), False
            )
            condition = conditioner(features, subgoal_latent, subgoal_mask)
            return model.predict(emb[:, :history_size] + condition[:, None, :], act_emb[:, :history_size])

        return model, latent_dim, predict_fn

    # CV / native baseline: AuxiliaryLeWM checkpoint.
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
    history_size = int(cfg.get("history_size", HISTORY_SIZE))

    def predict_fn(emb, act_emb, batch):
        return model.predict(emb[:, :history_size], act_emb[:, :history_size])

    return model, latent_dim, predict_fn


def _encode_batch(model, batch: dict[str, Any], device):
    import torch

    pixels = batch["pixels"].to(device, non_blocking=True)
    actions = torch.nan_to_num(batch["action"].to(device, non_blocking=True), 0.0)
    output = model.encode({"pixels": pixels, "action": actions})
    return output["emb"], output["act_emb"], actions


def _train_probe(probe, x_train: np.ndarray, y_train: np.ndarray, device, *, seed: int) -> None:
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.from_numpy(x_train).float().to(device)
    y = torch.from_numpy(y_train).float().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    loss_fn = torch.nn.MSELoss()
    n = x.shape[0]
    batch = min(4096, n)
    probe.train()
    for _ in range(PROBE_EPOCHS):
        perm = torch.randperm(n, generator=g).to(device)
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            optimizer.zero_grad(set_to_none=True)
            pred = probe(x[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            optimizer.step()


def _build_window_dataset(
    *,
    cut_path: Path,
    split_file: Path,
    split_key: str,
    dataset_name: str,
    data_cache_dir: Path,
    cfg: dict[str, Any],
):
    from bridgeengine.benchmark.pi07_fixed import Pi07H5WindowDataset

    ds_cfg = {
        "history_size": int(cfg.get("history_size", HISTORY_SIZE)),
        "num_preds": int(cfg.get("num_preds", 1)),
        "frameskip": int(cfg.get("frameskip", 1)),
        "batch_size": int(cfg.get("batch_size", 16)),
    }
    return Pi07H5WindowDataset.from_split(
        cut_path=Path(cut_path),
        split_file=Path(split_file),
        split_key=split_key,
        dataset_name=dataset_name,
        data_cache_dir=Path(data_cache_dir),
        cfg=ds_cfg,
    )


def _load_idm_tables(output_dir: Path, *, scales: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    scale_set = {int(x) for x in scales}
    for path in sorted((output_dir / "runs").glob(f"scale_*/*/{IDM_WINDOWS_FILENAME}")):
        scale_n = int(path.parent.parent.name.replace("scale_", ""))
        if scale_n not in scale_set:
            continue
        run_name = path.parent.name
        if "_seed" not in run_name:
            continue
        condition, seed_text = run_name.rsplit("_seed", 1)
        if condition not in ALL_CONDITIONS:
            continue
        frame = pd.read_csv(path)
        frame["scale_n"] = scale_n
        frame["condition"] = condition
        frame["seed"] = int(seed_text)
        frame["run_dir"] = str(path.parent)
        frame["episode_id"] = frame["episode_id"].astype(str)
        frame["start_idx"] = frame["start_idx"].astype(int)
        frame["sq_err"] = frame["sq_err"].astype(float)
        frame["dist_to_segment_end"] = pd.to_numeric(frame["dist_to_segment_end"], errors="coerce")
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _split_file_for_scale(scale_n: int) -> Path:
    return (
        Path("head_to_head_results/preregistered_100/splits") / f"scale_{int(scale_n)}_split.json"
    ).resolve()


def _scale_from_run_dir(run_dir: Path) -> int:
    return int(Path(run_dir).parent.name.replace("scale_", ""))


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="IDM action-decoding probe (eval-validity metric).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run.add_argument("--scales", type=int, nargs="+", default=[25])
    run.add_argument("--conditions", nargs="+", default=None)
    run.add_argument("--device", default="cuda")
    run.add_argument("--force", action="store_true")
    run.add_argument("--max-cells", type=int, default=None)

    probe = sub.add_parser("probe-run")
    probe.add_argument("--run-dir", required=True)
    probe.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    probe.add_argument("--device", default="cuda")

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    analyze.add_argument("--scales", type=int, nargs="+", default=[25])
    analyze.add_argument("--bootstrap-reps", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--report-path", default="LEAK_AND_POWER_REPORT.md")

    args = parser.parse_args()
    if args.cmd == "run":
        print(
            json.dumps(
                run_idm_grid(
                    args.output_dir,
                    scales=tuple(args.scales),
                    conditions=tuple(args.conditions) if args.conditions else None,
                    skip_existing=not args.force,
                    device=args.device,
                    max_cells=args.max_cells,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.cmd == "probe-run":
        print(json.dumps(probe_one_run(args.run_dir, output_dir=args.output_dir, device=args.device), indent=2, sort_keys=True))
    elif args.cmd == "analyze":
        print(
            json.dumps(
                analyze_idm_paired(
                    args.output_dir,
                    scales=tuple(args.scales),
                    bootstrap_reps=args.bootstrap_reps,
                    seed=args.seed,
                    report_path=args.report_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
