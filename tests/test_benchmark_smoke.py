from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridgeengine.benchmark.run_grid import run_grid
from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers


def test_benchmark_grid_shape(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=3, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path)
    rows = run_grid(result["snapshot_id"], data_root=tmp_path, output_dir=tmp_path / "bench")
    assert len(rows) == 12
    csv_rows = pd.read_csv(tmp_path / "bench" / "bench_results.csv")
    assert len(csv_rows) == 12
    means = csv_rows.groupby("family")["latent_mse"].mean()
    assert means["perceptive"] < means["baseline"]

