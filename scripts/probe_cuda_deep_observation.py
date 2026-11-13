"""CPU proof that unique-cell staging preserves native DFT gather operands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.cuda_deep_observation import build_plan
except ModuleNotFoundError:
    from cuda_deep_observation import build_plan


def verify():
    shapes = (
        (35, 49, 66),
        (35, 50, 65),
        (36, 49, 65),
        (36, 50, 65),
        (36, 49, 66),
        (35, 50, 66),
    )
    rng = np.random.default_rng(840112)
    indices = np.empty((6, 6, 137, 8), np.int32)
    for c in range(6):
        count = int(np.prod(shapes[(c + 3) % 6]))
        indices[:, c] = rng.integers(0, count, size=(6, 137, 8), dtype=np.int32)
        indices[0, c, 0] = np.asarray([0, 0, -1, count, count - 1, 0, 1, 2], np.int32)
    plan = build_plan(shapes, indices)
    keys = plan["keys"]
    remapped = plan["remapped_indices"]
    bases = plan["field_base"]
    mask_shape = tuple(map(int, plan["mask_shape_zyx"]))
    words = plan["publication_words"].view(np.uint32)
    fields = []
    for _step in range(4):
        values = [rng.standard_normal(s).astype(np.float32) for s in shapes]
        # Raw operand staging must preserve special values too, including a
        # NaN that a zero-weight original record would propagate through FMA.
        for c in range(6):
            values[c].ravel()[0] = np.asarray(0x7FC01234, np.uint32).view(np.float32)
        fields.append(values)
    staged = np.empty((4, len(keys)), np.float32)
    for c in range(6):
        where = np.flatnonzero(
            (keys >= bases[c]) & (keys < bases[c] + np.prod(shapes[c]))
        )
        local = keys[where] - bases[c]
        z, y, x = np.unravel_index(local, shapes[c])
        linear = (z * mask_shape[1] + y) * mask_shape[2] + x
        byte = (words[linear >> 2] >> (8 * (linear & 3))) & 255
        assert np.all(byte & (1 << c))
        for step in range(4):
            staged[step, where] = fields[step][c].ravel()[local]
    checked = 0
    invalid = 0
    for c in range(6):
        native = (c + 3) % 6
        raw = indices[:, c]
        valid = (raw >= 0) & (raw < np.prod(shapes[native]))
        assert np.all(remapped[:, c][~valid] == -1)
        invalid += int((~valid).sum())
        for step in range(4):
            expected = fields[step][native].ravel()[raw[valid]]
            actual = staged[step, remapped[:, c][valid]]
            assert np.array_equal(expected.view(np.uint32), actual.view(np.uint32))
            checked += expected.size
    return {
        "raw_operands_bitwise_checked": checked,
        "invalid_records_preserve_skip": invalid,
        "zero_weight_operand_nan_payload_preserved": True,
        "key_count": len(keys),
        "raw_record_count": indices.size,
        "staged_bytes_d4": plan["staged_bytes_d4"],
        "publication_mask_bytes": plan["mask_bytes"],
        "shape_zyx": mask_shape,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = verify()
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
