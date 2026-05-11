from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgeengine.orchestrate.runner import run_labelers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BridgeEngine Mode A labelers.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--labeler", action="append", dest="labelers", default=None)
    args = parser.parse_args()
    result = run_labelers(
        snapshot_id=args.snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        labeler_names=args.labelers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

