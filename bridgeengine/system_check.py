from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


DEFAULT_BRIDGE_ROOT = Path("D:/bridgedata_v2_subset")
DEFAULT_MODELS_ROOT = Path("D:/extraction_models")


def collect_status(
    bridge_root: str | Path = DEFAULT_BRIDGE_ROOT,
    models_root: str | Path = DEFAULT_MODELS_ROOT,
) -> list[dict[str, Any]]:
    bridge_root = Path(bridge_root)
    models_root = Path(models_root)
    checks = [
        ("BridgeData V2 subset", bridge_root, bridge_root / "episodes"),
        ("BridgeData manifest", bridge_root / "manifest.json", bridge_root / "manifest.json"),
        ("LEWM pilot subset", Path("C:/Users/Kevin/projects/LeWM_testbed/outputs/pilot_subset.json"), Path("C:/Users/Kevin/projects/LeWM_testbed/outputs/pilot_subset.json")),
        ("SAM2 tiny checkpoint", models_root / "sam2_checkpoints" / "sam2.1_hiera_tiny.pt", models_root / "sam2_checkpoints" / "sam2.1_hiera_tiny.pt"),
        ("Video-Depth-Anything repo", models_root / "Video-Depth-Anything", models_root / "Video-Depth-Anything" / "video_depth_anything"),
        ("VDA ViT-S checkpoint", models_root / "Video-Depth-Anything" / "checkpoints" / "video_depth_anything_vits.pth", models_root / "Video-Depth-Anything" / "checkpoints" / "video_depth_anything_vits.pth"),
        ("CoTracker repo", models_root / "co-tracker", models_root / "co-tracker" / "cotracker"),
    ]
    rows = []
    for name, display_path, required_path in checks:
        rows.append(
            {
                "component": name,
                "status": "present" if required_path.exists() else "missing",
                "path": str(display_path),
            }
        )
    imports = {
        "torch": "live GPU inference",
        "sam2": "SAM mask extraction",
        "video_depth_anything": "VDA extraction after repo path is on PYTHONPATH",
        "cotracker": "CoTracker extraction after repo/install",
        "streamlit": "viewer",
    }
    for module, purpose in imports.items():
        rows.append(
            {
                "component": f"Python import: {module}",
                "status": "installed" if importlib.util.find_spec(module) else "missing",
                "path": purpose,
            }
        )
    rows.append(_episode_artifact_status(bridge_root))
    return rows


def _episode_artifact_status(bridge_root: Path) -> dict[str, Any]:
    episodes_root = bridge_root / "episodes"
    if not episodes_root.exists():
        return {"component": "Per-episode artifacts", "status": "missing", "path": str(episodes_root)}
    required = ["frames.npy", "actions.npy", "depth.npy", "object_mask.npy", "tracks.npy", "visibility.npy", "video.mp4"]
    counts = dict.fromkeys(required, 0)
    episode_count = 0
    for ep_dir in episodes_root.iterdir():
        if not ep_dir.is_dir():
            continue
        episode_count += 1
        for name in required:
            if (ep_dir / name).exists():
                counts[name] += 1
    complete = min(counts.values()) if counts else 0
    return {
        "component": "Per-episode artifacts",
        "status": f"{complete}/{episode_count} complete",
        "path": json.dumps(counts, sort_keys=True),
    }


def main() -> None:
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Check local BridgeEngine demo readiness.")
    parser.add_argument("--bridge-root", default=str(DEFAULT_BRIDGE_ROOT))
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    args = parser.parse_args()
    rows = collect_status(args.bridge_root, args.models_root)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

