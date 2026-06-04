from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bridge_v2 import ingest_bridge_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest BridgeData V2 into BridgeEngine.")
    parser.add_argument("--source", default="bridge_v2", help="bridge_v2, synthetic, or a source root path")
    parser.add_argument("--episodes", default="13", help="Episode count to ingest, or 'all'.")
    parser.add_argument("--episode-offset", type=int, default=0, help="Deterministic offset into the selected source episode order.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--copy-raw", action="store_true", help="Copy raw episode files into bridgeengine_data/raw")
    args = parser.parse_args()

    result = ingest_bridge_v2(
        source=args.source,
        episodes=None if str(args.episodes).lower() == "all" else int(args.episodes),
        data_root=Path(args.data_root) if args.data_root else None,
        copy_raw=args.copy_raw,
        episode_offset=args.episode_offset,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
