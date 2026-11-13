"""Count executed D4 field expressions using the exact native launch geometry.

Counts are static work, not measured instruction or wall-time fractions. Field
expressions outside a component's support still execute before the output guard.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def analyze(path):
    fixture = json.loads(path.read_text())
    shapes = [v["shape"] for v in fixture["inputs"][:6]]
    limit = tuple(min(shape[a] for shape in shapes[:3]) for a in range(3))
    extent = tuple(max(shape[a] for shape in shapes) for a in range(3))
    masks = []
    for mask in range(8):
        tile = (64, 8 if mask in (0, 1, 2, 4) else (3 if mask == 7 else 4), 25)
        origins = []
        for axis, size in enumerate(tile):
            origins.append(
                [(o, min(o + size, 16)) for o in range(0, 16, size)]
                + [(o, min(o + size, extent[axis])) for o in range(limit[axis] - 16, extent[axis], size)]
                if mask & (1 << axis)
                else [(o, min(o + size, limit[axis] - 16)) for o in range(16, limit[axis] - 16, size)]
            )
        executed = useful_box = barriers = 0
        for box in itertools.product(*origins):
            lengths = [end - start for start, end in box]
            useful_box += 24 * math.prod(lengths)
            executed += 3 * sum((lengths[0] + 7 - k) * (tile[1] + 7 - k) * (tile[2] + 7 - k) for k in range(8))
            barriers += 1 + 8 * (lengths[0] + 7)
        masks.append({"mask": mask, "tile_zyx": tile,
                      "ctas": math.prod(map(len, origins)),
                      "executed_component_updates": executed,
                      "owned_bounding_box_component_updates": useful_box,
                      "block_barrier_instances": barriers})
    useful = 4 * sum(math.prod(s) for s in shapes)
    executed = sum(m["executed_component_updates"] for m in masks)
    for m in masks:
        m["executed_work_fraction"] = m["executed_component_updates"] / executed
    return {"fixture": str(path), "shape_zyx": limit, "component_shapes_native": shapes,
            "useful_four_step_component_updates": useful,
            "executed_component_updates": executed,
            "executed_to_useful_component_updates": executed / useful,
            "masks": masks,
            "scope": "Static geometry; excludes CPML expressions, source loops, publication, DFT and tail steps"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixtures", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = [analyze(path) for path in a.fixtures]
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for r in result:
        print(r["shape_zyx"], round(r["executed_to_useful_component_updates"], 5),
              [m["ctas"] for m in r["masks"]])


if __name__ == "__main__":
    main()
