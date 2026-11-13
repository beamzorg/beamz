"""Host-compiled profile dispatch predicate against its typed Python class."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_native_profile_fact import generate_guard
from scripts.cuda_readonly_profile_bindings import (
    AXES,
    equal_axis_profile_class,
    profile_input_index,
)


class Buffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("rank", ctypes.c_int32), ("element_type", ctypes.c_int32),
                ("dims", ctypes.c_int64 * 4), ("row_stride", ctypes.c_int32), ("plane_stride", ctypes.c_int32),
                ("compact_slots", ctypes.c_void_p), ("compact_ny", ctypes.c_int32), ("compact_nx", ctypes.c_int32)]


@pytest.fixture(scope="module")
def guard(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    out = tmp_path_factory.mktemp("profile-fact")
    (out / "guard.h").write_text(generate_guard()["source"])
    source = out / "guard.cc"
    source.write_text('#include "guard.h"\nextern "C" bool check(const BeamzBuffer*d,size_t n,const void*const*h){return BeamzNativeEqualAxisProfileFact(d,n,h);}\n')
    library = out / "guard.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-I",
                    str(Path(__file__).resolve().parents[2] / "cuda/src"), str(source), "-o", str(library)],
                   check=True, capture_output=True)
    fn = ctypes.CDLL(str(library)).check
    fn.argtypes = [ctypes.POINTER(Buffer), ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    fn.restype = ctypes.c_bool
    return fn


def packet():
    result = [None] * 62
    for field, delta in enumerate(SUPPORT_DELTA):
        result[field] = np.zeros(tuple(32 + d for d in delta), np.float32)
    for phase in (0, 1):
        for term, axis in enumerate(AXES):
            for kind in range(3):
                shape = tuple(24 if a == axis else 1 for a in range(3))
                result[profile_input_index(phase, term, kind)] = (
                    np.arange(24, dtype=np.float32) + phase * 30 + axis * 3 + kind).reshape(shape).copy()
    return result


def descriptors(values):
    buffers, host = (Buffer * 62)(), (ctypes.c_void_p * 62)()
    for index, value in enumerate(values):
        if value is None:
            continue
        d = buffers[index]
        d.data = host[index] = value.ctypes.data
        d.rank = value.ndim
        d.element_type = 0 if value.dtype == np.dtype("float32") else 1
        for axis, extent in enumerate(value.shape):
            d.dims[axis] = extent
        if value.ndim == 3:
            d.row_stride, d.plane_stride = value.strides[1] // value.itemsize, value.strides[0] // value.itemsize
    return buffers, host


@pytest.mark.parametrize("case", ["equal", "different-phase", "bit", "signed-zero", "matching-negative-zero", "nonfinite", "dtype",
                                  "rank", "shape", "strided", "field-shape", "short-packet", "null-host"])
def test_actual_cpp_guard_matches_typed_fact(guard, case):
    values = packet()
    index = profile_input_index(0, 2, 0)
    if case == "different-phase":
        for term in range(6):
            for kind in range(3):
                values[profile_input_index(1, term, kind)] *= np.float32(2)
    elif case == "bit":
        values[index].view(np.uint32).flat[0] ^= 1
    elif case == "signed-zero":
        values[profile_input_index(0, 1, 0)].flat[0], values[index].flat[0] = 0., -0.
    elif case == "matching-negative-zero":
        values[profile_input_index(0, 1, 0)].flat[0], values[index].flat[0] = -0., -0.
    elif case == "nonfinite":
        values[index].flat[0] = np.inf
    elif case == "dtype":
        values[index] = values[index].astype(np.float64)
    elif case == "rank":
        values[index] = values[index].reshape(24)
    elif case == "shape":
        values[index] = values[index].reshape(1, 24, 1)
    elif case == "strided":
        values[index] = np.zeros((48, 1, 1), np.float32)[::2]
    elif case == "field-shape":
        values[4] = np.zeros((34, 32, 33), np.float32)
    native, host = descriptors(values)
    if case == "null-host":
        host[index] = None
    expected = equal_axis_profile_class(values) is not None and case not in {"short-packet", "null-host"}
    assert guard(native, 61 if case == "short-packet" else 62, host) == expected


@pytest.mark.parametrize("initially_equal", [False, True])
def test_private_snapshot_predicate_does_not_follow_caller_mutation(guard, initially_equal):
    caller = packet()
    if not initially_equal:
        caller[profile_input_index(0, 2, 0)].view(np.uint32).flat[0] ^= 1
    captured = [value.copy() if value is not None else None for value in caller]
    native, host = descriptors(captured)
    assert guard(native, 62, host) == initially_equal
    caller[profile_input_index(0, 2, 0)].view(np.uint32).flat[0] ^= 1
    assert (equal_axis_profile_class(caller) is not None) != initially_equal
    assert guard(native, 62, host) == initially_equal
