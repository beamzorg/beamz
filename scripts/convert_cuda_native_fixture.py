#!/usr/bin/env python3
"""Convert an immutable native fixture to audited mmap NPY storage.

An optional existing NPY store supplies byte-identical aliases. Every source
array is still hashed; differing operational arrays get their own new files.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import (
    ArrayReader,
    ArrayWriter,
    digest_array,
    drop_pages,
)
from scripts.export_fdtdx_comparison import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-store", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("New output path required")
    f = copy.deepcopy(json.loads(a.fixture.read_text()))
    original = f.copy()
    if file_digest(a.fixture.parent / f["data_file"]) != f["data_sha256"]:
        raise ValueError("Source data hash differs")
    store = (
        ArrayReader(a.fixture.parent / f["data_directory"])
        if f.get("data_format") == "npy_directory_v1"
        else np.load(a.fixture.parent / f["data_file"], allow_pickle=False)
    )
    keys = list(store.metadata) if isinstance(store, ArrayReader) else store.files
    reuse = ArrayReader(a.reuse_store) if a.reuse_store else None
    directory = a.output.parent / (a.output.stem + ".arrays")
    writer = ArrayWriter(directory)
    aliased = written = 0
    for key in keys:
        value = store[key]
        digest = digest_array(value)
        if reuse and key in reuse.metadata and reuse.metadata[key]["sha256"] == digest:
            entry = dict(reuse.metadata[key])
            entry["file"] = str((reuse.directory / entry["file"]).resolve())
            writer.arrays[key] = entry
            aliased += 1
        else:
            writer.put(key, value)
            written += 1
        drop_pages(value)
    for item in f["inputs"]:
        if writer.arrays[item["key"]]["sha256"] != item["sha256"]:
            raise ValueError(f"Source input hash differs: {item['key']}")
    manifest = writer.finish()
    f.update(
        data_format="npy_directory_v1",
        data_directory=directory.name,
        data_file=str(manifest.relative_to(a.output.parent)),
        data_sha256=file_digest(manifest),
    )
    f["storage_conversion"] = {
        "source_fixture": str(a.fixture.resolve()),
        "source_fixture_sha256": file_digest(a.fixture),
        "source_data_sha256": original["data_sha256"],
        "array_aliases": aliased,
        "arrays_written": written,
        "all_array_bytes_preserved": True,
        "worker_sha256": file_digest(__file__),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(f, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(a.output),
                "array_aliases": aliased,
                "arrays_written": written,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
