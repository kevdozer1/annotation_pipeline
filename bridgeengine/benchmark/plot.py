from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_bar_chart(results_csv: Path, output_path: Path) -> Path:
    results = pd.read_csv(results_csv)
    grouped = (
        results.groupby("family", as_index=False)
        .agg(latent_mse_mean=("latent_mse", "mean"), latent_mse_std=("latent_mse", "std"))
        .sort_values("latent_mse_mean")
    )
    colors = {
        "baseline": "#5B6770",
        "rich_text": "#247BA0",
        "rich_text_metadata": "#2E7D32",
        "rich_text_metadata_subgoal": "#8C5E2A",
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(
        grouped["family"],
        grouped["latent_mse_mean"],
        yerr=grouped["latent_mse_std"].fillna(0.0),
        color=[colors.get(x, "#444444") for x in grouped["family"]],
        capsize=4,
    )
    ax.set_ylabel("Latent prediction MSE")
    ax.set_xlabel("Annotation family")
    backend = results.get("benchmark_backend", pd.Series(["unknown"])).iloc[0]
    if backend == "contract_smoke_no_science":
        title = "BridgeEngine Benchmark Contract Smoke, Not A Scientific Result"
    else:
        title = "BridgeEngine Real LeWM Smoke Ablation, 13 episodes"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.0,
        -0.32,
        "Caption: fixed 10/3 episode split; bars are seed mean +/- std. "
        "Smoke-scale only.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
