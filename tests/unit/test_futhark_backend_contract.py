"""Host-side contracts of the optional Futhark backend (no GPU required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from beamz.simulation.backend import normalize_backend
from beamz.simulation.futhark.runtime import (
    _storage_indices,
    source_table,
    storage_shape,
)

ROOT = Path(__file__).resolve().parents[2]


def _group(coeffs, starts, waveforms):
    return SimpleNamespace(
        coeffs=np.asarray(coeffs, np.float32),
        starts=np.asarray(starts, np.int32),
        waveforms=np.asarray(waveforms, np.float32),
    )


def test_futhark_is_an_explicit_backend_name():
    assert normalize_backend("futhark") == "futhark"
    assert normalize_backend(" Futhark ") == "futhark"


def test_source_table_drops_outside_and_pec_constrained_cells():
    shape = (4, 5, 6)
    fields = (shape,) * 6
    # Two one-cell sources on Ez after the E update (group 8) and one Hz
    # source after the H update (group 5); every PEC face is metallic.
    ez = _group(np.ones((2, 1, 1, 1)), [[2, 2, 3], [2, 0, 3]], [[1, 2, 3], [4, 5, 6]])
    hz = _group(np.full((1, 1, 1, 2), 0.5), [[0, 1, 5]], [[7, 8]])
    groups = (None,) * 5 + (hz, None, None, ez)
    target, amplitude, offset, length, group_ids, waves = source_table(
        groups, fields, 0b111111
    )
    flat = lambda z, y, x: (z * shape[1] + y) * shape[2] + x  # noqa: E731
    # Hz is normal to z: its z=0 face is constrained; x=6 is outside the field.
    assert target.tolist()[:2] == [-1, -1]
    # Ez at y=0 is tangential to the metallic y face.
    assert target.tolist()[2:] == [flat(2, 2, 3), -1]
    assert group_ids.tolist() == [5, 5, 8, 8]
    assert amplitude.tolist() == [0.5, 0.5, 1.0, 1.0]
    assert offset.tolist() == [0, 0, 2, 5]
    assert length.tolist() == [2, 2, 3, 3]
    assert waves.tolist() == [7, 8, 1, 2, 3, 4, 5, 6]


def test_pre_e_sources_ignore_pec_constraints():
    shape = (3, 3, 3)
    group = _group(np.ones((1, 1, 1, 1)), [[0, 0, 0]], [[1.0]])
    target, *_ = source_table((group,) + (None,) * 8, (shape,) * 6, 0b111111)
    assert target.tolist() == [0]


def test_targets_and_plans_rebase_onto_padded_storage():
    shapes = ((3, 4, 5), (3, 5, 6), (4, 4, 6), (4, 5, 5), (4, 4, 6), (3, 5, 6))
    storage = storage_shape(shapes)
    assert storage == (4, 5, 6)
    group = _group(np.ones((1, 1, 1, 1)), [[2, 3, 4]], [[1.0]])
    target, *_ = source_table((group,) + (None,) * 8, shapes, 0, storage)
    assert target.tolist() == [(2 * 5 + 3) * 6 + 4]
    # [M][6][P][N] plan offsets index each component's own shape; -1 and
    # out-of-range entries stay dropped.
    indices = np.full((1, 6, 3, 1), -1, np.int32)
    indices[0, 0, :, 0] = [(2 * 4 + 3) * 5 + 4, -1, 3 * 4 * 5]
    indices[0, 3, 0, 0] = (1 * 5 + 2) * 5 + 3
    rebased = np.asarray(_storage_indices(indices, shapes, storage))
    assert rebased[0, 0, :, 0].tolist() == [(2 * 5 + 3) * 6 + 4, -1, -1]
    assert rebased[0, 3, 0, 0] == (1 * 5 + 2) * 6 + 3


def test_handler_generator_rejects_consumed_inputs():
    spec = importlib.util.spec_from_file_location(
        "futhark_build", ROOT / "futhark" / "build.py"
    )
    build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build  # dataclasses resolve their module by name
    spec.loader.exec_module(build)
    manifest = {
        "entry_points": {
            "program": {
                "cfun": "futhark_entry_program",
                "inputs": [
                    {"name": "nsteps", "type": "i64", "unique": False},
                    {"name": "field", "type": "[][][]f32", "unique": True},
                ],
                "output": {"type": "([][][]f32)", "unique": True},
            }
        },
        "types": {
            "([][][]f32)": {
                "ctype": "struct futhark_opaque_x *",
                "ops": {"free": "futhark_free_opaque_x"},
                "record": {
                    "fields": [{"name": "0", "project": "p0", "type": "[][][]f32"}]
                },
            }
        },
    }
    with pytest.raises(SystemExit, match="consumed"):
        build.generate_handler(manifest, "fdtd.h")
    manifest["entry_points"]["program"]["inputs"][1]["unique"] = False
    handler = build.generate_handler(manifest, "fdtd.h")
    assert 'Attr<int64_t>("nsteps")' in handler
    assert "futhark_new_raw_f32_3d" in handler
