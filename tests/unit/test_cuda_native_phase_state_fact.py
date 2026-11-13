"""Actual host byte-range predicate for reordered point-local psi updates."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.cuda_native_phase_state_fact import generate_guard


class Buffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("rank", ctypes.c_int32), ("element_type", ctypes.c_int32),
                ("dims", ctypes.c_int64 * 4), ("row_stride", ctypes.c_int32), ("plane_stride", ctypes.c_int32),
                ("compact_slots", ctypes.c_void_p), ("compact_ny", ctypes.c_int32), ("compact_nx", ctypes.c_int32)]


@pytest.fixture(scope="module")
def guard(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    out = tmp_path_factory.mktemp("phase-state-fact")
    (out / "guard.h").write_text(generate_guard()["source"])
    source = out / "guard.cc"
    source.write_text('#include "guard.h"\nextern "C" bool check(const BeamzBuffer*s,const BeamzBuffer*p){return BeamzNativeIndependentPhaseState(s,p);}\n')
    library = out / "guard.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I",
                    str(Path(__file__).resolve().parents[2] / "cuda/src"), str(source), "-o", str(library)],
                   check=True, capture_output=True)
    fn = ctypes.CDLL(str(library)).check
    fn.argtypes = [ctypes.POINTER(Buffer), ctypes.POINTER(Buffer)]
    fn.restype = ctypes.c_bool
    return fn


def descriptors():
    states, profiles = (Buffer * 12)(), (Buffer * 36)()
    for i, value in enumerate(states):
        value.data, value.rank, value.element_type = 0x100000 + 0x10000 * i, 3, 0
        value.dims[:] = (24, 4, 4, 1)
        value.row_stride, value.plane_stride = 4, 16
    for i, value in enumerate(profiles):
        value.data, value.rank, value.element_type = 0x800000 + 0x1000 * i, 3, 0
        value.dims[:] = (24, 1, 1, 1)
        value.row_stride, value.plane_stride = 1, 1
    return states, profiles


@pytest.mark.parametrize("case", ["disjoint", "adjacent", "same", "partial", "profile-overlap", "profile-adjacent",
                                  "pointer-overflow", "profile-pointer-overflow", "elements-overflow", "null", "dtype",
                                  "strides", "compact", "empty"])
def test_actual_range_guard_handles_aliasing_and_integer_boundaries(guard, case):
    states, profiles = descriptors()
    if case == "adjacent":
        states[1].data = states[0].data + 24 * 4 * 4 * 4
    elif case == "same":
        states[1].data = states[0].data
    elif case == "partial":
        states[1].data = states[0].data + 32
    elif case == "profile-overlap":
        profiles[0].data = states[11].data + 16
    elif case == "profile-adjacent":
        profiles[0].data = states[11].data + 24 * 4 * 4 * 4
    elif case == "pointer-overflow":
        states[0].data = ctypes.c_void_p(-16).value
    elif case == "profile-pointer-overflow":
        profiles[0].data = ctypes.c_void_p(-16).value
    elif case == "elements-overflow":
        states[0].dims[:] = (2**30, 4, 4, 1)
    elif case == "null":
        states[0].data = None
    elif case == "dtype":
        states[0].element_type = 2
    elif case == "strides":
        states[0].plane_stride = 17
    elif case == "compact":
        states[0].compact_slots = 1
    elif case == "empty":
        states[0].dims[0] = 0
    assert guard(states, profiles) == (case in {"disjoint", "adjacent", "profile-adjacent"})
