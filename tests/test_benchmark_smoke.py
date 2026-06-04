from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridgeengine.benchmark.run_grid import run_grid
from bridgeengine.ingest import ingest_bridge_v2
from bridgeengine.orchestrate import run_labelers


def test_benchmark_grid_shape(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=3, data_root=tmp_path)
    run_labelers(result["snapshot_id"], data_root=tmp_path, allow_fallback=True)
    rows = run_grid(
        result["snapshot_id"],
        data_root=tmp_path,
        output_dir=tmp_path / "bench",
        allow_scaffolding_labels=True,
    )
    assert len(rows) == 12
    csv_rows = pd.read_csv(tmp_path / "bench" / "bench_results.csv")
    assert len(csv_rows) == 12
    assert set(csv_rows["benchmark_backend"]) == {"contract_smoke_no_science"}
    assert set(csv_rows["family"]) == {
        "baseline",
        "rich_text",
        "rich_text_metadata",
        "rich_text_metadata_subgoal",
    }
