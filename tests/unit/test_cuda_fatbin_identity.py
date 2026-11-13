"""Inspection must not mutate an ELF backing a live module mapping."""

import struct

import pytest

from scripts.audit_cuda_fatbin_identity import audit, fatbin


def elf(payload=b"GPU instructions"):
    names = b"\0.shstrtab\0.nv_fatbin\0"
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 40, 64)
    struct.pack_into("<HHH", header, 58, 64, 3, 1)
    rows = b"\0" * 64
    rows += struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, 256, len(names), 0, 0, 1, 0)
    rows += struct.pack(
        "<IIQQQQIIQQ", 11, 1, 0, 0, 256 + len(names), len(payload), 0, 0, 1, 0
    )
    return bytes(header) + rows + names + payload


def test_comparison_reads_without_touching_module_bytes_or_mtime(tmp_path):
    paths = [tmp_path / name for name in ("on.so", "off.so")]
    for path in paths:
        path.write_bytes(elf())
        path.chmod(0o444)
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    assert audit(paths)["device_code_byte_identical"]
    assert before == [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    paths[1].chmod(0o644)
    paths[1].write_bytes(elf(b"different"))
    assert not audit(paths)["device_code_byte_identical"]


@pytest.mark.parametrize("bad", [b"", b"not ELF", elf()[:-1], elf()[:160]])
def test_truncated_or_non_elf_data_is_rejected(bad):
    with pytest.raises(ValueError):
        fatbin(bad)
