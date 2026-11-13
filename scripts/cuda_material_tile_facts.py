"""Conservative, bit-exact uniform-E facts for moving CUDA tile trajectories.

CPU research certificate only. No generated/native dispatch uses these facts yet.
The union box includes every Q(n), n=1..depth, clipped to the real component
support. Aliased table codes are equivalent only when their float32 bits match;
out-of-table byte codes have the existing PackedMaterialSource +0 semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def bit_lut(table):
    table = np.asarray(table)
    if (
        table.dtype != np.dtype("float32")
        or table.ndim != 1
        or not 1 <= table.size <= 256
    ):
        raise ValueError("Expected FP32 byte-code table of length1..256")
    lut = np.zeros(256, dtype=np.uint32)
    lut[: table.size] = table.view(np.uint32)
    return lut


def uniform_boxes(packed, table, shape, origins, owned=(5, 15, 31), depth=16):
    """Prove equality to the most frequent physical coefficient bit pattern.

    An uncertified tile may still be uniform with a different value. This
    deliberate false-negative is safe; no unsupported tile is admitted.
    """
    shape = tuple(int(x) for x in shape)
    owned = np.asarray(owned, dtype=np.int64)
    origins = np.asarray(origins, dtype=np.int64)
    packed = np.asarray(packed)
    if (
        len(shape) != 3
        or min(shape) < 1
        or origins.ndim != 2
        or origins.shape[1] != 3
        or owned.shape != (3,)
        or np.any(owned < 1)
        or depth < 1
        or packed.dtype != np.dtype("int32")
        or packed.ndim != 1
    ):
        raise ValueError("Invalid packed field/trajectory geometry")
    count = int(np.prod(shape, dtype=np.int64))
    if count >= 2**31 or packed.size != (count + 3) // 4:
        raise ValueError("Packed size or prefix count outside supported bounds")
    # Explicit shifts preserve logical code ordering independent of host endian.
    words = packed.view(np.uint32)
    codes = np.empty(count, dtype=np.uint8)
    for byte in range(4):
        codes[byte::4] = (words[: codes[byte::4].size] >> (8 * byte)) & 255
    lut = bit_lut(table)
    counts = np.bincount(codes, minlength=256)
    choices, inverse = np.unique(lut, return_inverse=True)
    frequencies = np.zeros(choices.size, dtype=np.uint64)
    np.add.at(frequencies, inverse, counts.astype(np.uint64))
    bits = int(choices[np.argmax(frequencies)])
    prefix = np.zeros(tuple(x + 1 for x in shape), dtype=np.int32)
    prefix[1:, 1:, 1:] = (lut[codes] != bits).reshape(shape)
    del codes
    for axis in range(3):
        np.cumsum(prefix, axis=axis, dtype=np.int32, out=prefix)
    lower = np.clip(origins - depth, 0, shape)
    upper = np.clip(origins + owned - 1, 0, shape)  # exclusive, n=1 high+1
    bad = np.zeros(origins.shape[0], dtype=np.int64)
    for corner in range(8):
        coordinates = np.where([(corner >> a) & 1 for a in range(3)], upper, lower)
        sign = 1 if corner.bit_count() % 2 else -1
        bad += sign * prefix[tuple(coordinates.T)].astype(np.int64)
    if np.any(bad < 0):
        raise AssertionError("Invalid inclusion/exclusion count")
    return {
        "bits": bits,
        "admitted": bad == 0,
        "nonuniform_points": bad,
        "lower": lower,
        "upper": upper,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.cuda_fixture_store import ArrayReader

    f = json.loads(a.fixture.read_text())
    reader = ArrayReader(a.fixture.parent / f["data_directory"])
    maximum = np.max([x["shape"] for x in f["inputs"][:6]], axis=0)
    owned = np.asarray((5, 15, 31))
    counts = (maximum + a.depth + owned - 1) // owned
    ids = np.indices(tuple(counts), dtype=np.int64).reshape(3, -1).T
    origins = ids * owned
    all_admitted = np.ones(ids.shape[0], dtype=bool)
    rows = []
    for c in range(3):
        result = uniform_boxes(
            reader[f["inputs"][40 + c]["key"]],
            reader[f["inputs"][37 + c]["key"]],
            f["inputs"][3 + c]["shape"],
            origins,
            depth=a.depth,
        )
        all_admitted &= result["admitted"]
        rows.append(
            {
                "component": c,
                "bits": result["bits"],
                "uniform_tiles": int(result["admitted"].sum()),
            }
        )
    # Weight by actual supported component updates, not padded/ghost slots.
    total = uniform = 0
    for field in f["inputs"][:6]:
        shape = np.asarray(field["shape"])
        for n in range(1, a.depth + 1):
            low = np.clip(origins - n, 0, shape)
            high = np.clip(origins - n + owned, 0, shape)
            updates = np.prod(high - low, axis=1)
            total += int(updates.sum())
            uniform += int(updates[all_admitted].sum())
    result = {
        "scope": __doc__,
        "fixture": str(a.fixture),
        "fixture_sha256": hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
        "depth": a.depth,
        "owned": owned.tolist(),
        "counts": counts.tolist(),
        "tiles": ids.shape[0],
        "all_three_uniform_tiles": int(all_admitted.sum()),
        "uniform_real_component_update_fraction": uniform / total,
        "components": rows,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
