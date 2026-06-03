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
    ax.set_title("BridgeEngine Rich-Prompt Benchmark, 13 episodes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
