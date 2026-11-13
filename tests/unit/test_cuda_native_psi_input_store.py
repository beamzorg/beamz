"""Fail-closed source binding for an optional CPML pointer-lowering experiment."""

import hashlib

import pytest

from scripts import patch_cuda_native_psi_input_store as patch


def test_actual_pointer_change_preserves_load_alias_guard_and_inverse(monkeypatch):
    # A synthetic reviewed parent lets this test exercise all actual mutation
    # guards without depending on the lifetime of private benchmark snapshots.
    consumer = (patch.ALIAS_GUARD + "\n" + patch.LOAD + "\n" + patch.OLD).encode()
    extension, cell, descriptor = b"extension", b"cell", b"descriptor"
    def digest(data):
        return hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(patch, "PARENTS", {digest(consumer): digest(extension)})
    monkeypatch.setattr(patch, "CELL", digest(cell))
    monkeypatch.setattr(patch, "DESCRIPTOR", digest(descriptor))
    result = patch.transform(consumer, extension, cell, descriptor)
    assert result.decode().replace(patch.NEW, patch.OLD).encode() == consumer
    assert patch.LOAD in result.decode()
    assert patch.ALIAS_GUARD in result.decode()
    for which in range(4):
        inputs = [consumer, extension, cell, descriptor]
        inputs[which] += b"unreviewed"
        with pytest.raises(ValueError):
            patch.transform(*inputs)


def test_unknown_real_parent_fails_closed():
    with pytest.raises(ValueError, match="Exact reviewed"):
        patch.transform(b"unknown", b"", b"", b"")
