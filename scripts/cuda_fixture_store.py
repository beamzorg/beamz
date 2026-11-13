"""Bounded-memory NPY fixture storage and exact numerical comparisons."""

from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path

import numpy as np

CHUNK_BYTES = 32 << 20


def host_chunks(value, chunk_bytes=CHUNK_BYTES):
    count = int(np.prod(value.shape, dtype=np.int64))
    width = max(1, chunk_bytes // np.dtype(value.dtype).itemsize)
    flat = value.reshape(-1)
    for start in range(0, count, width):
        yield start, np.asarray(flat[start : min(count, start + width)])


def digest_array(value):
    digest = hashlib.sha256()
    digest.update(np.dtype(value.dtype).str.encode())
    digest.update(json.dumps(list(value.shape)).encode())
    for _, block in host_chunks(value):
        digest.update(memoryview(np.ascontiguousarray(block)).cast("B"))
    return digest.hexdigest()


def drop_pages(value):
    mapping = getattr(value, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise"):
        mapping.madvise(mmap.MADV_DONTNEED)


class ArrayWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.arrays = {}

    def put(
        self,
        key,
        value=None,
        *,
        shape=None,
        dtype=None,
        zero=False,
        require_finite=False,
    ):
        shape = tuple(shape if shape is not None else value.shape)
        dtype = np.dtype(dtype if dtype is not None else value.dtype)
        path = self.directory / f"{key}.npy"
        target = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        digest = hashlib.sha256()
        digest.update(dtype.str.encode())
        digest.update(json.dumps(list(shape)).encode())
        count = int(np.prod(shape, dtype=np.int64))
        maximum = 0.0
        finite = True
        if zero:
            # A newly truncated sparse file reads as zero; avoid a giant RAM fill.
            block = np.zeros(min(count, max(1, CHUNK_BYTES // dtype.itemsize)), dtype)
            for start in range(0, count, max(1, block.size)):
                digest.update(
                    memoryview(block[: min(block.size, count - start)]).cast("B")
                )
        else:
            flat = target.reshape(-1)
            for start, block in host_chunks(value):
                flat[start : start + block.size] = block
                digest.update(memoryview(np.ascontiguousarray(block)).cast("B"))
                if dtype.kind == "f" and block.size:
                    valid = np.isfinite(block)
                    finite &= bool(np.all(valid))
                    if not finite and require_finite:
                        raise ValueError(f"Nonfinite fixture values: {key}")
                    maximum = max(
                        maximum, float(np.max(np.abs(block[valid]), initial=0))
                    )
        target.flush()
        drop_pages(target)
        del target
        self.arrays[key] = {
            "file": path.name,
            "shape": list(shape),
            "dtype": dtype.str,
            "sha256": digest.hexdigest(),
            "nbytes": count * dtype.itemsize,
            "max_abs": maximum,
            "finite": finite,
        }
        return self.arrays[key]

    def alias(self, key, existing):
        self.arrays[key] = dict(self.arrays[existing])

    def finish(self):
        path = self.directory / "manifest.json"
        path.write_text(
            json.dumps(
                {"schema": "beamz.npy-array-directory.v1", "arrays": self.arrays},
                indent=2,
            )
            + "\n"
        )
        return path


class ArrayReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "manifest.json").read_text())[
            "arrays"
        ]
        self.cache = {}

    def __getitem__(self, key):
        item = self.metadata[key]
        filename = item["file"]
        if filename not in self.cache:
            self.cache[filename] = np.load(
                self.directory / filename, mmap_mode="r", allow_pickle=False
            )
        result = self.cache[filename]
        if list(result.shape) != item["shape"] or result.dtype.str != item["dtype"]:
            raise ValueError(f"Fixture dtype/shape differs: {key}")
        return result

    def verify(self, key):
        value = self[key]
        passed = digest_array(value) == self.metadata[key]["sha256"]
        drop_pages(value)
        if not passed:
            raise ValueError(f"Fixture hash differs: {key}")


def compare_arrays(actual, expected, chunk_bytes=8 << 20):
    compatible = actual.shape == expected.shape and actual.dtype == expected.dtype
    result = {
        "passed": False,
        "exact": False,
        "bitwise_exact": False,
        "shape": list(actual.shape),
        "dtype": actual.dtype.str,
    }
    if not compatible:
        return result
    exact, bits, finite = True, True, True
    maximum, scale = 0.0, 0.0
    left, right = actual.reshape(-1), expected.reshape(-1)
    width = max(1, chunk_bytes // actual.dtype.itemsize)
    for start in range(0, left.size, width):
        a, b = (
            np.asarray(left[start : start + width]),
            np.asarray(right[start : start + width]),
        )
        exact &= bool(np.array_equal(a, b))
        bits &= bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))
        if a.dtype.kind == "f":
            finite &= bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0
                    )
                ),
            )
            scale = max(scale, float(np.max(np.abs(b), initial=0)))
    result.update(
        passed=exact and finite,
        exact=exact,
        bitwise_exact=bits,
        finite=finite,
        max_abs_error=maximum,
        reference_peak=scale,
    )
    return result
