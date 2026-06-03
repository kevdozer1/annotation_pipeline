from __future__ import annotations

import argparse
import json
from pathlib import Path

from bridgeengine.orchestrate.runner import run_labelers, run_perceptive_labelers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BridgeEngine Mode A labelers.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--labeler", action="append", dest="labelers", default=None)
    parser.add_argument("--perceptive", action="store_true", help="Run comparison perception labelers instead of rich-prompt labelers")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow scaffolding-only deterministic fallback labels when the selected VLM is unavailable")
    parser.add_argument("--vlm-backend", default=None, help="VLM backend for semantic labelers: openai, moondream, or mock")
    parser.add_argument("--vlm-model", default=None, help="Optional model name for the selected VLM backend")
    args = parser.parse_args()
    fn = run_perceptive_labelers if args.perceptive else run_labelers
    result = fn(
        snapshot_id=args.snapshot,
        data_root=Path(args.data_root) if args.data_root else None,
        labeler_names=args.labelers,
        **(
            {}
            if args.perceptive
            else {"allow_fallback": args.allow_fallback, "vlm_backend": args.vlm_backend, "vlm_model": args.vlm_model}
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
