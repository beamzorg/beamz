#!/usr/bin/env python3
"""Collect modal throughput JSONs, keeping experimental variants distinguishable."""

import argparse
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("*/*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "kernel_gcups" not in data:
            continue
        frequencies = re.search(r"(\d+) frequencies", " ".join(data["monitors"]))
        rows.append(
            {
                "variant": path.parent.name,
                "artifact": str(path.relative_to(args.directory)),
                "backend": data["backend"],
                "gpus": data["device_count"],
                "shape": "x".join(map(str, data["grid_dimensions"])),
                "material_cells": data["material_cells"],
                "frequencies": int(frequencies.group(1)) if frequencies else "",
                "kernel_gcups": data["kernel_gcups"],
                "end_to_end_gcups": data["end_to_end_gcups"],
                "compile_s": data["compile_s"],
                "peak_memory_bytes": data["peak_memory_bytes"],
                "host_setup": data.get("host_setup", False),
                "extension_sha256": data.get("extension_sha256", ""),
                "harness_sha256": data.get("harness_sha256", ""),
            }
        )
    if not rows:
        raise SystemExit("No throughput measurements found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} measurements to {args.output}")


if __name__ == "__main__":
    main()
