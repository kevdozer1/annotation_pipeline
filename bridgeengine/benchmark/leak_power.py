from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def reevaluate_completed(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    scales: tuple[int, ...] = (25,),
    conditions: tuple[str, ...] | None = None,
    skip_existing: bool = True,
    max_cells: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    condition_filter = set(conditions or ALL_CONDITIONS)
    cells = [
        cell
        for cell in _manifest_cells(output_dir)
        if int(cell["scale_n"]) in set(scales)
        and str(cell["condition"]) in condition_filter
        and cell.get("eval_cmd")
    ]
    completed = []
    skipped = []
    missing = []
    for cell in cells:
        run_dir = Path(cell["run_dir"])
        windows_csv = run_dir / "fixed_eval_windows.csv"
        if not _final_weights(run_dir).exists():
            missing.append(_cell_id(cell, "missing_final_checkpoint"))
            continue
        if skip_existing and windows_csv.exists():
            skipped.append(_cell_id(cell, "windows_csv_exists"))
            continue
        if max_cells is not None and len(completed) >= int(max_cells):
            break
        cmd = [str(x) for x in cell["eval_cmd"]]
        cmd[0] = sys.executable
        subprocess.run(cmd, check=True)
        completed.append(_cell_id(cell, "reevaluated"))
    report = {
        "output_dir": str(output_dir),
        "scales": [int(x) for x in scales],
        "conditions": sorted(condition_filter),
        "reevaluated": completed,
        "skipped": skipped,
        "missing": missing,
    }
    report_path = output_dir / "leak_power_reeval_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def analyze_paired_windows(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    scales: tuple[int, ...] = (25,),
    bootstrap_reps: int = 2000,
    seed: int = 0,
    report_path: str | Path = "LEAK_AND_POWER_REPORT.md",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tables = _load_window_tables(output_dir, scales=scales)
    if tables.empty:
        raise FileNotFoundError(f"No fixed_eval_windows.csv files found under {output_dir}")
    paired = _paired_deltas(tables, reference="P0_pi07_baseline")
    paired_adapter = _paired_deltas(tables, reference="P_adapter_null")
    paired_adapter = paired_adapter[paired_adapter["condition"].str.startswith("P", na=False)]
    paired_adapter["reference"] = "P_adapter_null"
    paired["reference"] = "P0_pi07_baseline"
    all_deltas = pd.concat([paired, paired_adapter], ignore_index=True)
    deltas_csv = output_dir / "paired_window_deltas.csv"
    all_deltas.to_csv(deltas_csv, index=False)

    summary = _paired_summary(all_deltas, bootstrap_reps=bootstrap_reps, seed=seed)
    summary_csv = output_dir / "paired_window_summary.csv"
    summary.to_csv(summary_csv, index=False)

    leak = _subgoal_leak_bins(paired)
    leak_csv = output_dir / "subgoal_leak_bins.csv"
    leak.to_csv(leak_csv, index=False)

    variance = _target_variance_table(output_dir, scales=scales)
    variance_csv = output_dir / "target_variance_table.csv"
    if not variance.empty:
        variance.to_csv(variance_csv, index=False)

    figures = _write_figures(output_dir, summary, leak, scales=scales)
    report = _write_report(
        Path(report_path),
        summary=summary,
        leak=leak,
        variance=variance,
        scales=scales,
        deltas_csv=deltas_csv,
        summary_csv=summary_csv,
        leak_csv=leak_csv,
        variance_csv=variance_csv,
        figures=figures,
    )
    return {
        "window_tables": int(tables[["scale_n", "condition", "seed"]].drop_duplicates().shape[0]),
        "paired_rows": int(len(all_deltas)),
        "normalized_metric_available": bool("delta_norm_sq_err" in all_deltas.columns),
        "deltas_csv": str(deltas_csv),
        "summary_csv": str(summary_csv),
        "leak_csv": str(leak_csv),
        "variance_csv": str(variance_csv) if not variance.empty else None,
        "figures": {key: str(value) for key, value in figures.items()},
        "report_path": str(report),
    }


def _manifest_cells(output_dir: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for manifest_name in ("command_manifest.json", "pi07_command_manifest.json"):
        manifest_path = output_dir / manifest_name
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for cell in payload.get("commands", []):
            if cell.get("run_dir") and cell.get("eval_cmd"):
                cells.append(cell)
    return cells


def _final_weights(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "final" / "full_weights.pt"


def _cell_id(cell: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "scale_n": int(cell["scale_n"]),
        "condition": str(cell["condition"]),
        "seed": int(cell["seed"]),
        "run_dir": str(cell["run_dir"]),
        "status": status,
    }


def _load_window_tables(output_dir: Path, *, scales: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    scale_set = {int(x) for x in scales}
    for path in sorted((output_dir / "runs").glob("scale_*/*/fixed_eval_windows.csv")):
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
        if "norm_sq_err" in frame.columns:
            frame["norm_sq_err"] = pd.to_numeric(frame["norm_sq_err"], errors="coerce")
        frame["dist_to_segment_end"] = pd.to_numeric(frame["dist_to_segment_end"], errors="coerce")
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    # Only treat the normalized metric as available if every loaded run carries it,
    # so a partially re-evaluated tree doesn't silently drop conditions from the
    # normalized paired join.
    if "norm_sq_err" in combined.columns and combined["norm_sq_err"].isna().any():
        combined = combined.drop(columns=["norm_sq_err"])
    return combined


def _paired_deltas(tables: pd.DataFrame, *, reference: str) -> pd.DataFrame:
    pairs = []
    key_cols = ["scale_n", "seed", "episode_id", "start_idx"]
    has_norm = "norm_sq_err" in tables.columns
    ref_cols = key_cols + ["sq_err"] + (["norm_sq_err"] if has_norm else [])
    ref = tables[tables["condition"] == reference][ref_cols].rename(
        columns={"sq_err": "ref_sq_err", "norm_sq_err": "ref_norm_sq_err"}
    )
    if ref.empty:
        return pd.DataFrame()
    for condition, group in tables.groupby("condition"):
        if condition == reference:
            continue
        merged = group.merge(ref, on=key_cols, how="inner")
        if merged.empty:
            continue
        merged["reference"] = reference
        merged["delta_sq_err"] = merged["sq_err"] - merged["ref_sq_err"]
        merged["advantage_sq_err"] = -merged["delta_sq_err"]
        if has_norm:
            merged["delta_norm_sq_err"] = merged["norm_sq_err"] - merged["ref_norm_sq_err"]
            merged["advantage_norm_sq_err"] = -merged["delta_norm_sq_err"]
        pairs.append(merged)
    if not pairs:
        return pd.DataFrame()
    keep = [
        "scale_n",
        "seed",
        "condition",
        "reference",
        "episode_id",
        "start_idx",
        "segment_idx",
        "subgoal_mask",
        "dist_to_segment_end",
        "sq_err",
        "ref_sq_err",
        "delta_sq_err",
        "advantage_sq_err",
    ]
    if has_norm:
        keep += ["norm_sq_err", "ref_norm_sq_err", "delta_norm_sq_err", "advantage_norm_sq_err"]
    return pd.concat(pairs, ignore_index=True)[keep]


def _paired_summary(deltas: pd.DataFrame, *, bootstrap_reps: int, seed: int) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(int(seed))
    has_norm = "delta_norm_sq_err" in deltas.columns
    for (scale_n, reference, condition), group in deltas.groupby(["scale_n", "reference", "condition"]):
        values = group["delta_sq_err"].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        cluster_means = (
            group.groupby(["seed", "episode_id"], dropna=False)["delta_sq_err"]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        ci_low, ci_high = _bootstrap_ci(cluster_means, reps=bootstrap_reps, rng=rng)
        ref_mean = float(group["ref_sq_err"].mean())
        row = {
            "scale_n": int(scale_n),
            "reference": str(reference),
            "condition": str(condition),
            "paired_windows": int(len(values)),
            "bootstrap_units": int(len(cluster_means)),
            "seeds": int(group["seed"].nunique()),
            "mean_delta_sq_err": mean,
            "mean_delta_pct": float(mean / ref_mean * 100.0) if ref_mean else float("nan"),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "separates_zero": bool(ci_high < 0.0 or ci_low > 0.0),
            "direction": "better" if mean < 0.0 else "worse",
        }
        if has_norm:
            norm_values = group["delta_norm_sq_err"].to_numpy(dtype=np.float64)
            norm_mean = float(np.nanmean(norm_values)) if norm_values.size else float("nan")
            norm_cluster = (
                group.groupby(["seed", "episode_id"], dropna=False)["delta_norm_sq_err"]
                .mean()
                .to_numpy(dtype=np.float64)
            )
            norm_cluster = norm_cluster[np.isfinite(norm_cluster)]
            norm_ci_low, norm_ci_high = _bootstrap_ci(norm_cluster, reps=bootstrap_reps, rng=rng)
            ref_norm_mean = float(group["ref_norm_sq_err"].mean())
            row.update(
                {
                    "mean_delta_norm_sq_err": norm_mean,
                    "mean_delta_norm_pct": float(norm_mean / ref_norm_mean * 100.0)
                    if ref_norm_mean
                    else float("nan"),
                    "ci_low_norm": norm_ci_low,
                    "ci_high_norm": norm_ci_high,
                    "separates_zero_norm": bool(norm_ci_high < 0.0 or norm_ci_low > 0.0),
                    "direction_norm": "better" if norm_mean < 0.0 else "worse",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scale_n", "reference", "mean_delta_sq_err"])


def _bootstrap_ci(values: np.ndarray, *, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    idx = rng.integers(0, values.size, size=(int(reps), values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _subgoal_leak_bins(deltas_vs_p0: pd.DataFrame) -> pd.DataFrame:
    if deltas_vs_p0.empty:
        return pd.DataFrame()
    probe_conditions = [
        "B_depth",
        "D_tracks",
        "E_depth_tracks",
        "P3_pi07_subgoal",
        "P4_pi07_full_stack",
    ]
    rows = []
    bins = [(-10**9, 0, "at_boundary"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 10**9, ">10")]
    for (scale_n, condition), group in deltas_vs_p0[deltas_vs_p0["condition"].isin(probe_conditions)].groupby(
        ["scale_n", "condition"]
    ):
        valid = group[np.isfinite(group["dist_to_segment_end"].to_numpy(dtype=np.float64))]
        if valid.empty:
            continue
        dist = valid["dist_to_segment_end"].to_numpy(dtype=np.float64)
        advantage = valid["advantage_sq_err"].to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = (
                float(np.corrcoef(dist, advantage)[0, 1])
                if len(valid) > 1 and np.std(dist) > 0 and np.std(advantage) > 0
                else float("nan")
            )
            # Relative advantage divides each window's advantage by the P0 baseline's
            # squared error on that same window, so the leak diagnostic is not dominated
            # by absolute error scale and is comparable across distance bins.
            ref_err = valid["ref_sq_err"].to_numpy(dtype=np.float64)
            rel_adv = np.divide(advantage, ref_err, out=np.full_like(advantage, np.nan), where=ref_err > 0)
            rel_mask = np.isfinite(rel_adv)
            rel_corr = (
                float(np.corrcoef(dist[rel_mask], rel_adv[rel_mask])[0, 1])
                if rel_mask.sum() > 1 and np.std(dist[rel_mask]) > 0 and np.std(rel_adv[rel_mask]) > 0
                else float("nan")
            )
        near_df = valid[valid["dist_to_segment_end"] <= 2]
        far_df = valid[valid["dist_to_segment_end"] > 10]
        near = near_df["advantage_sq_err"]
        far = far_df["advantage_sq_err"]
        near_ref = float(near_df["ref_sq_err"].mean()) if len(near_df) else float("nan")
        far_ref = float(far_df["ref_sq_err"].mean()) if len(far_df) else float("nan")
        near_adv = float(near.mean()) if len(near) else float("nan")
        far_adv = float(far.mean()) if len(far) else float("nan")
        near_rel = near_adv / near_ref if len(near) and near_ref else float("nan")
        far_rel = far_adv / far_ref if len(far) and far_ref else float("nan")
        rows.append(
            {
                "scale_n": int(scale_n),
                "condition": str(condition),
                "bin": "near_vs_far",
                "windows": int(len(valid)),
                "mean_advantage": float(valid["advantage_sq_err"].mean()),
                "mean_delta": float(valid["delta_sq_err"].mean()),
                "dist_advantage_corr": corr,
                "dist_advantage_rel_corr": rel_corr,
                "near_advantage": near_adv,
                "far_advantage": far_adv,
                "near_minus_far_advantage": (near_adv - far_adv) if len(near) and len(far) else float("nan"),
                "near_advantage_rel": near_rel,
                "far_advantage_rel": far_rel,
                "near_minus_far_advantage_rel": (near_rel - far_rel)
                if np.isfinite(near_rel) and np.isfinite(far_rel)
                else float("nan"),
            }
        )
        for low, high, label in bins:
            subset = valid[(valid["dist_to_segment_end"] >= low) & (valid["dist_to_segment_end"] <= high)]
            if subset.empty:
                continue
            rows.append(
                {
                    "scale_n": int(scale_n),
                    "condition": str(condition),
                    "bin": label,
                    "windows": int(len(subset)),
                    "mean_advantage": float(subset["advantage_sq_err"].mean()),
                    "mean_delta": float(subset["delta_sq_err"].mean()),
                    "dist_advantage_corr": corr,
                    "near_advantage": float("nan"),
                    "far_advantage": float("nan"),
                    "near_minus_far_advantage": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _target_variance_table(output_dir: Path, *, scales: tuple[int, ...]) -> pd.DataFrame:
    """Per-condition held-out target-latent variance, read from fixed_eval.json.

    This is the direct test of the reviewer's confound: if depth conditions (B/E)
    lower raw MSE mainly by contracting the latent space, their mean per-dim
    target variance will sit materially below the no-aux anchors (A_baseline for
    the CV trainer, P0 for the native baseline) at the same scale.
    """
    scale_set = {int(x) for x in scales}
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "runs").glob("scale_*/*/fixed_eval.json")):
        run_name = path.parent.name
        if "_seed" not in run_name:
            continue
        condition, seed_text = run_name.rsplit("_seed", 1)
        if condition not in ALL_CONDITIONS:
            continue
        scale_n = int(path.parent.parent.name.replace("scale_", ""))
        if scale_n not in scale_set:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "heldout_target_variance" not in payload:
            continue
        rows.append(
            {
                "scale_n": scale_n,
                "condition": condition,
                "seed": int(seed_text),
                "heldout_target_variance": float(payload.get("heldout_target_variance", float("nan"))),
                "heldout_target_mean_sq_norm": float(payload.get("heldout_target_mean_sq_norm", float("nan"))),
                "latent_mse": float(payload.get("latent_mse", float("nan"))),
                "norm_latent_mse": float(payload.get("norm_latent_mse", float("nan"))),
            }
        )
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["scale_n", "condition"])
        .agg(
            seeds=("seed", "nunique"),
            mean_target_variance=("heldout_target_variance", "mean"),
            std_target_variance=("heldout_target_variance", "std"),
            mean_latent_mse=("latent_mse", "mean"),
            mean_norm_latent_mse=("norm_latent_mse", "mean"),
        )
        .reset_index()
    )
    grouped["std_target_variance"] = grouped["std_target_variance"].fillna(0.0)
    anchors_a = grouped[grouped["condition"] == "A_baseline"].set_index("scale_n")["mean_target_variance"].to_dict()
    anchors_p0 = (
        grouped[grouped["condition"] == "P0_pi07_baseline"].set_index("scale_n")["mean_target_variance"].to_dict()
    )
    grouped["var_ratio_vs_A_baseline"] = grouped.apply(
        lambda r: float(r["mean_target_variance"] / anchors_a[r["scale_n"]])
        if anchors_a.get(r["scale_n"])
        else float("nan"),
        axis=1,
    )
    grouped["var_ratio_vs_P0"] = grouped.apply(
        lambda r: float(r["mean_target_variance"] / anchors_p0[r["scale_n"]])
        if anchors_p0.get(r["scale_n"])
        else float("nan"),
        axis=1,
    )
    # Flag a depth/track aux condition whose latent variance is contracted by more
    # than 10% relative to the no-aux native anchor at the same scale.
    def _flag(row: pd.Series) -> bool:
        if row["condition"] not in {"B_depth", "D_tracks", "E_depth_tracks"}:
            return False
        ratio = row["var_ratio_vs_A_baseline"]
        return bool(np.isfinite(ratio) and ratio < 0.90)

    grouped["variance_contracted_flag"] = grouped.apply(_flag, axis=1)
    return grouped.sort_values(["scale_n", "condition"]).reset_index(drop=True)


def _write_figures(output_dir: Path, summary: pd.DataFrame, leak: pd.DataFrame, *, scales: tuple[int, ...]) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    forest_path = figure_dir / "paired_delta_forest.png"
    norm_forest_path = figure_dir / "paired_delta_forest_normalized.png"
    leak_path = figure_dir / "subgoal_advantage_by_boundary_distance.png"

    p0 = summary[summary["reference"] == "P0_pi07_baseline"].copy()
    p0 = p0[p0["scale_n"].isin([int(x) for x in scales])]
    if not p0.empty:
        labels = [f"{int(row.scale_n)} {row.condition}" for row in p0.itertuples(index=False)]
        y = np.arange(len(p0))
        x = p0["mean_delta_sq_err"].to_numpy()
        xerr = np.vstack([x - p0["ci_low"].to_numpy(), p0["ci_high"].to_numpy() - x])
        fig_h = max(4.0, 0.32 * len(labels) + 1.0)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.errorbar(x, y, xerr=xerr, fmt="o", color="#1B4965", ecolor="#5FA8D3", capsize=3)
        ax.axvline(0.0, color="#333333", linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Paired delta in RAW per-window latent MSE vs P0 native baseline; negative is better")
        ax.set_title("Paired Window Delta Forest Plot (raw)")
        fig.tight_layout()
        fig.savefig(forest_path, dpi=180)
        plt.close(fig)

    # Normalized forest: paired delta in variance-normalized per-window error.
    if not p0.empty and "mean_delta_norm_sq_err" in p0.columns and p0["mean_delta_norm_sq_err"].notna().any():
        pn = p0[p0["mean_delta_norm_sq_err"].notna()].copy()
        labels = [f"{int(row.scale_n)} {row.condition}" for row in pn.itertuples(index=False)]
        y = np.arange(len(pn))
        x = pn["mean_delta_norm_sq_err"].to_numpy()
        xerr = np.vstack([x - pn["ci_low_norm"].to_numpy(), pn["ci_high_norm"].to_numpy() - x])
        fig_h = max(4.0, 0.32 * len(labels) + 1.0)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.errorbar(x, y, xerr=xerr, fmt="o", color="#7A4419", ecolor="#D38B5F", capsize=3)
        ax.axvline(0.0, color="#333333", linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Paired delta in VARIANCE-NORMALIZED per-window error vs P0; negative is better")
        ax.set_title("Paired Window Delta Forest Plot (variance-normalized)")
        fig.tight_layout()
        fig.savefig(norm_forest_path, dpi=180)
        plt.close(fig)

    leak_bins = leak[(leak["bin"] != "near_vs_far") & leak["condition"].isin(["P3_pi07_subgoal", "P4_pi07_full_stack", "B_depth", "D_tracks", "E_depth_tracks"])]
    if not leak_bins.empty:
        order = ["at_boundary", "1-2", "3-5", "6-10", ">10"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for condition, group in leak_bins.groupby("condition"):
            group = group.copy()
            group["bin_order"] = group["bin"].map({label: idx for idx, label in enumerate(order)})
            group = group.sort_values(["scale_n", "bin_order"])
            # Keep the first requested scale visually clean for the leak diagnostic.
            first_scale = int(scales[0])
            scale_group = group[group["scale_n"] == first_scale]
            if scale_group.empty:
                continue
            ax.plot(
                scale_group["bin"],
                scale_group["mean_advantage"],
                marker="o",
                linewidth=2,
                label=condition,
            )
        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.set_ylabel("Mean advantage vs P0 per-window MSE; positive is better")
        ax.set_xlabel("Distance from active window to subtask segment end")
        ax.set_title(f"Boundary-Distance Leak Audit, Scale {int(scales[0])}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(leak_path, dpi=180)
        plt.close(fig)
    figures = {"paired_delta_forest": forest_path, "subgoal_leak": leak_path}
    if norm_forest_path.exists():
        figures["paired_delta_forest_normalized"] = norm_forest_path
    return figures


def _write_report(
    report_path: Path,
    *,
    summary: pd.DataFrame,
    leak: pd.DataFrame,
    variance: pd.DataFrame,
    scales: tuple[int, ...],
    deltas_csv: Path,
    summary_csv: Path,
    leak_csv: Path,
    variance_csv: Path,
    figures: dict[str, Path],
) -> Path:
    p0 = summary[summary["reference"] == "P0_pi07_baseline"].copy()
    separated = p0[p0["separates_zero"]]
    has_norm = "separates_zero_norm" in p0.columns
    separated_norm = p0[p0["separates_zero_norm"]] if has_norm else p0.iloc[0:0]
    subgoal_rows = leak[(leak["bin"] == "near_vs_far") & leak["condition"].isin(["P3_pi07_subgoal", "P4_pi07_full_stack"])]
    leak_verdict = _leak_verdict(subgoal_rows)
    lines = [
        "# Leak And Power Report",
        "",
        "Eval-only report. No checkpoints were trained or modified in this pass.",
        "",
        f"Scales analyzed: {', '.join(str(int(x)) for x in scales)}.",
        "Scale 100 is intentionally excluded here because its checkpoint set is incomplete; this report only "
        "analyzes scales with all CV and pi0.7 cells completed.",
        f"Paired deltas CSV: `{deltas_csv}`.",
        f"Paired summary CSV: `{summary_csv}`.",
        f"Boundary leak CSV: `{leak_csv}`.",
        f"Target-variance table CSV: `{variance_csv}`.",
        "",
        "## Eval-Validity Framing",
        "",
        "Every condition finetunes the full LeWM model, so each condition's held-out next-latent MSE is measured "
        "in its own latent geometry. A condition can therefore lower raw MSE by contracting target-latent variance "
        "rather than predicting better. This report reads every paired delta in two ways: the raw per-window MSE "
        "delta, and a variance-normalized delta where each run's per-window error is divided by that run's own mean "
        "per-dimension target-latent variance (`norm_sq_err = sq_err / heldout_target_variance`). The normalized "
        "delta is unitless (fraction of target variance unexplained) and is the leakage-robust read.",
        "",
        "## Paired Power Verdict (raw)",
        "",
        "CIs are paired bootstraps over episode-seed clusters, not individual adjacent windows. "
        "That is intentionally conservative because window errors are temporally correlated.",
        "",
    ]
    if separated.empty:
        lines.append("No analyzed condition has a raw paired bootstrap CI that separates cleanly from zero versus P0 native baseline.")
    else:
        names = ", ".join(
            f"scale {int(row.scale_n)} {row.condition} ({row.direction})" for row in separated.itertuples(index=False)
        )
        lines.append(f"Conditions with RAW paired CIs separating from zero versus P0: {names}.")
    lines.extend(["", _markdown_table(_compact_summary(p0)), ""])

    if has_norm:
        lines.extend(["## Paired Power Verdict (variance-normalized)", ""])
        if separated_norm.empty:
            lines.append(
                "After normalizing each run by its own target-latent variance, no condition's paired CI separates "
                "from zero versus P0. This is the conservative, leakage-robust verdict."
            )
        else:
            names = ", ".join(
                f"scale {int(row.scale_n)} {row.condition} ({row.direction_norm})"
                for row in separated_norm.itertuples(index=False)
            )
            lines.append(f"Conditions with NORMALIZED paired CIs separating from zero versus P0: {names}.")
        lines.extend(["", _markdown_table(_compact_norm_summary(p0)), ""])

    lines.extend(["## Per-Condition Target-Latent Variance", ""])
    if variance.empty:
        lines.append("No target-variance fields were found in fixed_eval.json; re-evaluate with the updated evaluator.")
    else:
        contracted = variance[variance.get("variance_contracted_flag", False) == True]  # noqa: E712
        if not contracted.empty:
            flagged = ", ".join(
                f"scale {int(row.scale_n)} {row.condition} (var ratio vs A_baseline "
                f"{float(row.var_ratio_vs_A_baseline):.3f})"
                for row in contracted.itertuples(index=False)
            )
            lines.append(
                f"**Variance-contraction flag:** {flagged}. For these conditions, part of the raw MSE advantage is "
                "explained by a smaller target-latent variance, not better prediction; trust the normalized delta."
            )
        else:
            lines.append(
                "No depth/track aux condition contracts mean per-dim target variance by more than 10% versus the "
                "A_baseline native anchor at the same scale, so the raw advantages are not primarily a variance artifact."
            )
        lines.extend(["", _markdown_table(_compact_variance(variance)), ""])

    lines.extend(["## Subgoal-Leak Audit", "", leak_verdict, ""])
    if not subgoal_rows.empty:
        lines.append(_markdown_table(_compact_leak(subgoal_rows)))
        lines.append("")
        lines.append(
            "Relative advantage divides each window's advantage by the P0 baseline error on the same window, so the "
            "near-vs-far comparison is not dominated by absolute error scale. Honest reading: the absence of "
            "near-boundary concentration rules out crude target-copying of the supplied subgoal frame, but it does "
            "**not** rule out privileged-future-information leakage. A same-episode subgoal frame is drawn from the "
            "future of the very trajectory being predicted; it can carry globally useful future context that lowers "
            "error uniformly across distance bins rather than only at the boundary. Deployability still requires "
            "replacing the same-episode subgoal with a retrieved or generated one."
        )
        lines.append("")
    figure_lines = [
        "## Figures",
        "",
        f"- Paired delta forest (raw): `{figures['paired_delta_forest']}`",
    ]
    if "paired_delta_forest_normalized" in figures:
        figure_lines.append(f"- Paired delta forest (normalized): `{figures['paired_delta_forest_normalized']}`")
    figure_lines.append(f"- Boundary-distance audit: `{figures['subgoal_leak']}`")
    lines.extend(figure_lines)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Negative paired delta means the condition has lower held-out next-latent MSE than P0 on the same windows. "
            "Where the raw and normalized verdicts disagree, the normalized verdict governs: a raw win that vanishes "
            "after variance normalization is a latent-geometry artifact, not a prediction gain. The boundary-distance "
            "audit is a diagnostic, not a proof.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    cols = [
        "scale_n",
        "condition",
        "paired_windows",
        "bootstrap_units",
        "seeds",
        "mean_delta_sq_err",
        "mean_delta_pct",
        "ci_low",
        "ci_high",
        "separates_zero",
    ]
    out = summary[cols].copy()
    for col in ("mean_delta_sq_err", "mean_delta_pct", "ci_low", "ci_high"):
        out[col] = out[col].map(lambda x: f"{float(x):.6g}")
    return out


def _compact_norm_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "mean_delta_norm_sq_err" not in summary.columns:
        return summary.iloc[0:0]
    cols = [
        "scale_n",
        "condition",
        "paired_windows",
        "mean_delta_norm_sq_err",
        "mean_delta_norm_pct",
        "ci_low_norm",
        "ci_high_norm",
        "separates_zero_norm",
    ]
    out = summary[cols].copy()
    for col in ("mean_delta_norm_sq_err", "mean_delta_norm_pct", "ci_low_norm", "ci_high_norm"):
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return out


def _compact_variance(variance: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scale_n",
        "condition",
        "seeds",
        "mean_target_variance",
        "mean_latent_mse",
        "mean_norm_latent_mse",
        "var_ratio_vs_A_baseline",
        "var_ratio_vs_P0",
        "variance_contracted_flag",
    ]
    out = variance[[c for c in cols if c in variance.columns]].copy()
    for col in ("mean_target_variance", "mean_latent_mse", "mean_norm_latent_mse", "var_ratio_vs_A_baseline", "var_ratio_vs_P0"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return out


def _compact_leak(leak: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scale_n",
        "condition",
        "windows",
        "dist_advantage_corr",
        "dist_advantage_rel_corr",
        "near_advantage",
        "far_advantage",
        "near_minus_far_advantage",
        "near_advantage_rel",
        "far_advantage_rel",
        "near_minus_far_advantage_rel",
    ]
    cols = [c for c in cols if c in leak.columns]
    out = leak[cols].copy()
    for col in cols:
        if col in ("scale_n", "condition", "windows"):
            continue
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return out


def _leak_verdict(subgoal_rows: pd.DataFrame) -> str:
    if subgoal_rows.empty:
        return "No P3/P4 subgoal rows were available for the leak audit."
    flags = []
    for row in subgoal_rows.itertuples(index=False):
        near_minus_far = float(row.near_minus_far_advantage) if pd.notna(row.near_minus_far_advantage) else float("nan")
        corr = float(row.dist_advantage_corr) if pd.notna(row.dist_advantage_corr) else float("nan")
        if np.isfinite(near_minus_far) and near_minus_far > 0.0 and (not np.isfinite(corr) or corr < 0.0):
            flags.append(f"{row.condition} at scale {int(row.scale_n)}")
    if flags:
        return (
            "Leak warning: subgoal advantage is larger near segment boundaries for "
            + ", ".join(flags)
            + ". Treat same-episode subgoal conditioning as an oracle-style diagnostic until replaced with retrieval/generated subgoals."
        )
    return "No clear near-boundary concentration was observed for P3/P4 in the analyzed scales."


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(col) for col in frame.columns]
    rows = [[str(value) for value in row] for row in frame.astype(str).itertuples(index=False, name=None)]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-only paired power and subgoal-leak analysis.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    reevaluate = sub.add_parser("reevaluate")
    reevaluate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    reevaluate.add_argument("--scales", type=int, nargs="+", default=[25])
    reevaluate.add_argument("--conditions", nargs="+", default=None)
    reevaluate.add_argument("--force", action="store_true")
    reevaluate.add_argument("--max-cells", type=int, default=None)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    analyze.add_argument("--scales", type=int, nargs="+", default=[25])
    analyze.add_argument("--bootstrap-reps", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--report-path", default="LEAK_AND_POWER_REPORT.md")

    args = parser.parse_args()
    if args.cmd == "reevaluate":
        print(
            json.dumps(
                reevaluate_completed(
                    args.output_dir,
                    scales=tuple(args.scales),
                    conditions=tuple(args.conditions) if args.conditions else None,
                    skip_existing=not args.force,
                    max_cells=args.max_cells,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.cmd == "analyze":
        print(
            json.dumps(
                analyze_paired_windows(
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
