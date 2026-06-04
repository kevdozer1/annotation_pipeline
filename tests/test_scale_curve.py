from __future__ import annotations

import json
from pathlib import Path

from bridgeengine.benchmark.scale_curve import plan_scale_curve
from bridgeengine.ingest import ingest_bridge_v2


def test_scale_curve_plan_writes_fixed_splits(tmp_path: Path) -> None:
    result = ingest_bridge_v2(source="synthetic", episodes=6, data_root=tmp_path)
    plan = plan_scale_curve(
        result["snapshot_id"],
        sizes=(4, 6, 10),
        data_root=tmp_path,
        output_dir=tmp_path / "scale",
        heldout_count=2,
        seed=7,
    )

    assert [row["size"] for row in plan["available_sizes"]] == [4, 6]
    assert [row["size"] for row in plan["unavailable_sizes"]] == [10]

    first_split = Path(plan["available_sizes"][0]["split_file"])
    second_split = Path(plan["available_sizes"][1]["split_file"])
    split_a = json.loads(first_split.read_text(encoding="utf-8"))
    split_b = json.loads(second_split.read_text(encoding="utf-8"))

    assert split_a["heldout_episode_ids"] == split_b["heldout_episode_ids"]
    assert set(split_a["heldout_episode_ids"]).isdisjoint(split_a["train_episode_ids"])
    assert set(split_b["heldout_episode_ids"]).isdisjoint(split_b["train_episode_ids"])
    assert set(split_a["train_episode_ids"]).issubset(split_b["train_episode_ids"])
