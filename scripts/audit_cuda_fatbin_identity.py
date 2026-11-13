"""Compare CUDA device sections without rewriting a loaded shared library.

Read ELF bytes directly. Some binary inspection commands rewrite their input
unless a separate output is supplied; that is unsafe while CUDA/Python has the
library mapped, even when the resulting bytes are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def fatbin(data: bytes) -> bytes:
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected a little-endian ELF64 module")
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, names_index = struct.unpack_from("<HHH", data, 58)
    if stride < 64 or not count or names_index >= count:
        raise ValueError("Unsupported or invalid ELF section table")
    if offset + stride * count > len(data):
        raise ValueError("Truncated ELF section table")
    rows = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride)
        for i in range(count)
    ]

    def contents(row: tuple) -> bytes:
        start, size = row[4:6]
        if start + size > len(data):
            raise ValueError("Truncated ELF section payload")
        return data[start : start + size]

    names = contents(rows[names_index])
    matches = []
    for row in rows:
        if row[0] >= len(names):
            raise ValueError("Invalid ELF section name")
        name = names[row[0] :].split(b"\0", 1)[0]
        if name == b".nv_fatbin":
            matches.append(contents(row))
    if len(matches) != 1 or not matches[0]:
        raise ValueError("Expected one nonempty CUDA fatbin section")
    return matches[0]


def audit(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("At least two modules required")
    rows, sections = [], []
    for path in paths:
        before = path.stat()
        data = path.read_bytes()
        section = fatbin(data)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("Module changed during read-only inspection")
        sections.append(section)
        rows.append(
            {
                "module": str(path.resolve()),
                "module_sha256": hashlib.sha256(data).hexdigest(),
                "fatbin_sha256": hashlib.sha256(section).hexdigest(),
                "fatbin_bytes": len(section),
                "mtime_ns": after.st_mtime_ns,
            }
        )
    return {
        "schema": "beamz.readonly-device-identity.v1",
        "modules": rows,
        "device_code_byte_identical": all(s == sections[0] for s in sections[1:]),
        "method": "Read-only ELF64 parsing; no binary tools or GPU context",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve() == p.resolve() for p in args.module):
        parser.error("Report must not replace an input module")
    if args.output.exists() and any(args.output.samefile(p) for p in args.module):
        parser.error("Report must not alias an input module")
    result = audit(args.module)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"device_code_byte_identical": result["device_code_byte_identical"]})
    )
    if not result["device_code_byte_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
