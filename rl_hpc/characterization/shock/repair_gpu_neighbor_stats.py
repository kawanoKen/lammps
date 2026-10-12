#!/usr/bin/env python3
"""Repair GPU trajectory JSON using FullNghs retained in complete LAMMPS logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import shock_counterfactual as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    repaired = 0
    for transitions in sorted(args.input.glob("seed-*/transitions.jsonl")):
        rows = [json.loads(line) for line in transitions.read_text().splitlines()]
        log = transitions.with_name("log.lammps").read_text(errors="replace")
        blocks = [block for block in base.timing_blocks(log) if block["steps"] > 0]
        # Last 121 positive runs are the common 50-step initialization followed
        # by 120 policy intervals; earlier runs belong to restart creation state.
        blocks = blocks[-(len(rows) + 1):]
        if len(blocks) != len(rows) + 1 or blocks[0]["steps"] != 50:
            raise RuntimeError(f"unexpected timing blocks: {transitions}")
        for index, row in enumerate(rows):
            before = blocks[index]
            after = blocks[index + 1]
            if before["neighs"]["mean"] <= 0 or after["neighs"]["mean"] <= 0:
                raise RuntimeError(f"missing FullNghs: {transitions}:{index}")
            row["state_before"]["previous_segment"] = before
            row["state_before"]["atom_imbalance"] = before["nlocal"]["max"] / before["nlocal"]["mean"]
            row["state_after"]["previous_segment"] = after
            row["state_after"]["atom_imbalance"] = after["nlocal"]["max"] / after["nlocal"]["mean"]
        temporary = transitions.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        temporary.replace(transitions)
        repaired += len(rows)
    print(f"repaired {repaired} transitions")


if __name__ == "__main__":
    main()
