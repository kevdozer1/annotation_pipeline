from __future__ import annotations

import argparse
from pathlib import Path

from bridgeengine.paths import data_root as resolve_data_root
from bridgeengine.quality_gate import evaluate_snapshot_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BridgeEngine label quality gates and print a report.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    report = evaluate_snapshot_quality(root / "snapshots" / args.snapshot)
    print(report.to_text())


if __name__ == "__main__":
    main()
