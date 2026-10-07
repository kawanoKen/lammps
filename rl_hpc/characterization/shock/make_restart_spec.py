#!/usr/bin/env python3
"""Create a portable one-restart specification with an actual local digest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--physical-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restart = args.restart.resolve()
    if not restart.is_file():
        parser.error("restart does not exist")
    payload = [{
        "path": str(restart),
        "physical_seed": args.physical_seed,
        "sha256": hashlib.sha256(restart.read_bytes()).hexdigest(),
    }]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
