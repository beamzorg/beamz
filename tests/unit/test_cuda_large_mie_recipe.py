"""CPU work/physics resolution and native data-class admission for large Mie."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.export_cuda_large_fixture import moving_packet_admission, recipe


def options(**changes):
    return SimpleNamespace(**({"recipe": "mie", "shape": (512, 512, 512), "steps": None,
                               "reference_steps": 256, "duration_fs": 30.} | changes))


def test_mie512_short_reference_keeps_full_physical_waveform_horizon():
    args, spec, steps, metadata = recipe(options())
    assert steps == 256 and spec["timesteps"] == 12347
    assert spec["resolution_nm"] == 1.328125
    assert spec["shape_zyx"] == [512] * 3
    assert spec["pml_cells"] == 12
    assert metadata["waveform_horizon_steps"] == 12347
    assert args.radius_nm == 100 and args.particle_index == 2
    assert args.wavelength_nm == 500 and args.courant == .95 and args.monitor_faces == 6
    assert args.steps is None


def test_guide_defaults_and_explicit_step_override_stay_unchanged():
    args, spec, steps, metadata = recipe(options(recipe="guide", reference_steps=None))
    assert spec is None and steps == args.steps == 512
    assert args.pml == 12 and args.monitor_offset_cells == 32
    assert args.monitors == 2 and args.frequencies == 3 and metadata["not_Mie"]
    assert recipe(options(recipe="guide", steps=259))[2] == 259


def test_mie_rejects_rectangles_and_reference_beyond_prepared_waveforms():
    with pytest.raises(ValueError):
        recipe(options(shape=(512, 512, 256)))
    with pytest.raises(ValueError):
        recipe(options(steps=64, reference_steps=256))


def packet():
    values = [np.empty((0,), np.float32) for _ in range(134)]
    for c, delta in enumerate(SUPPORT_DELTA):
        values[c] = np.empty(tuple(32 + d for d in delta), np.float32)
    for i in range(6, 12):
        values[i] = np.asarray(1., np.float32)
    for i in range(40, 43):
        values[i] = np.empty((8,), np.int32)
    for g in range(9):
        count = 2 if g in (5, 7) else 0
        values[86 + 3 * g] = np.empty((count, 1, 3, 4), np.float32)
        values[87 + 3 * g] = np.empty((count, 12347), np.float32)
        values[88 + 3 * g] = np.zeros((count, 3), np.int32)
    return {"arguments": values, "attributes": {"metric_kind": 0, "boundary_code": 3072,
                                                 "coincident_source_group_mask": 160}}


def test_native_moving_source_class_checks_actual_coincidence_not_only_flag():
    captured = packet()
    assert moving_packet_admission(captured)["checked"]
    captured["arguments"][88 + 3 * 5][1, 0] = 1
    with pytest.raises(ValueError, match="uploaded starts"):
        moving_packet_admission(captured)


@pytest.mark.parametrize("change", ("pre-E", "missing-coincidence", "dense-H"))
def test_unsupported_moving_classes_fail_before_the_reference_solve(change):
    captured = packet()
    if change == "pre-E":
        for offset in range(3):
            captured["arguments"][86 + offset] = captured["arguments"][86 + 3 * 5 + offset]
    elif change == "missing-coincidence":
        captured["attributes"]["coincident_source_group_mask"] = 0
    else:
        captured["arguments"][6] = np.ones((2, 2, 2), np.float32)
    with pytest.raises(ValueError):
        moving_packet_admission(captured)
