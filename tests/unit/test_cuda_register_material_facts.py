"""Execute the actual host certificate against a separate NumPy oracle."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from scripts.patch_cuda_register_material import FACTS
from scripts.patch_cuda_register_material_coverage import coverage_facts


@pytest.fixture(scope="module", params=(False, True), ids=("v2", "coverage75"))
def classifier(tmp_path_factory, request):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    out = tmp_path_factory.mktemp("uniform-d2-host")
    prelude = r"""
#include <vector>
#include <cstring>
#include <cstdint>
#include <climits>
#include <algorithm>
constexpr int kBeamzS32=1,kBeamzF32=2,cudaMemcpyHostToDevice=0;
struct Buffer{int rank=0,element_type=0;int64_t dims[3]{};void*data=nullptr;};
struct Host{Buffer descriptor;size_t bytes=0;const void*data=nullptr;};
inline int cudaMemcpyAsync(void*d,const void*s,size_t n,int,int){std::memcpy(d,s,n);return 0;}
inline void Check(int,const char*){}
struct Session{
 bool mstep_=true;int workspace_count_=12,monitor_offset_=113;
 struct{bool cpml_enabled=true;}attributes_;
 struct{int value=0;}stream_;
 std::vector<Host>host_=std::vector<Host>(134);
 std::vector<Buffer>base_inputs_=std::vector<Buffer>(134);
"""
    tail = r"""
};
extern "C" size_t run(int n,const int32_t*cx,const int32_t*cy,const int32_t*cz,
 const uint32_t*table,int nt,const int32_t*bounds,const int32_t*masks,int rows,int columns,int32_t*out){
 Session s;
 const int32_t*codes[3]={cx,cy,cz};const int dims[3][3]={{n+1,n+1,n},{n+1,n,n+1},{n,n+1,n+1}};
 for(int c=0;c<3;++c){
  s.host_[3+c].descriptor.rank=3;s.host_[3+c].descriptor.element_type=kBeamzF32;
  size_t count=1;for(int a=0;a<3;++a){s.host_[3+c].descriptor.dims[a]=dims[c][a];count*=dims[c][a];}
  auto&t=s.host_[37+c];t.descriptor.rank=1;t.descriptor.element_type=kBeamzF32;t.descriptor.dims[0]=nt;t.data=table;t.bytes=nt*4;
  auto&code=s.host_[40+c];code.descriptor.rank=1;code.descriptor.element_type=kBeamzS32;code.data=codes[c];code.bytes=(count+3)/4*4;
 }
 auto&b=s.host_[129];b.descriptor.rank=1;b.descriptor.element_type=kBeamzS32;b.data=bounds;b.bytes=8;
 auto&m=s.host_[130];m.descriptor.rank=2;m.descriptor.element_type=kBeamzS32;m.descriptor.dims[0]=rows;m.descriptor.dims[1]=columns;m.bytes=rows*columns*4;m.data=masks;
 std::memcpy(out,masks,m.bytes);s.base_inputs_[130].data=out;s.PrepareUniformD2();return s.uniform_d2_tiles_;
}
"""
    source = out / "facts.cc"
    source.write_text(
        prelude + (coverage_facts(FACTS) if request.param else FACTS) + tail
    )
    library = out / "facts.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    fn = ctypes.CDLL(str(library)).run
    pointer = ctypes.c_void_p
    fn.argtypes = [
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
        pointer,
    ]
    fn.restype = ctypes.c_size_t
    fn.coverage_guard = request.param
    return fn


@pytest.mark.parametrize(
    "mode", ["uniform", "aliases", "corner", "signed_zero", "invalid_zero"]
)
def test_actual_constructor_certificate(classifier, mode):
    n = 44
    shapes = ((n + 1, n + 1, n), (n + 1, n, n + 1), (n, n + 1, n + 1))
    table = np.array([1, 2, 2], np.float32)
    codes = [np.full(shape, 2, np.uint8) for shape in shapes]
    if mode == "aliases":
        for value in codes:
            value[:, ::2] = 1
    elif mode == "corner":
        codes[1][5, 14, 31] = 0
    elif mode == "signed_zero":
        table = np.array([0, -0.0], np.float32)
        codes = [np.full(shape, 1, np.uint8) for shape in shapes]
        codes[0][5, 14, 31] = 0
    elif mode == "invalid_zero":
        table = np.array([2, 0], np.float32)
        codes = [np.full(shape, 255, np.uint8) for shape in shapes]
    packed = []
    for value in codes:
        array = np.zeros((value.size + 3) // 4 * 4, np.uint8)
        array[: value.size] = value.ravel()
        packed.append(array.view(np.int32))
    bounds = np.array([4, n - 4], np.int32)
    masks = np.array([[15, 3], [12, 1], [0, 8]], np.int32)
    original = masks.copy()
    caller_arrays = [*packed, table, bounds, masks]
    caller_bits = [value.view(np.uint8).copy() for value in caller_arrays]
    output = np.empty_like(masks)
    count = classifier(
        n,
        *(value.ctypes.data for value in packed),
        table.ctypes.data,
        len(table),
        bounds.ctypes.data,
        masks.ctypes.data,
        *masks.shape,
        output.ctypes.data,
    )
    expected = masks.copy()
    lut = np.zeros(256, np.uint32)
    lut[: len(table)] = table.view(np.uint32)
    for by, bx in np.ndindex(masks.shape):
        if not masks[by, bx]:
            continue
        same = True
        for value in codes:
            shape = value.shape
            region = value[
                max(0, bounds[0] - 2) : min(shape[0], bounds[1] + 2),
                max(0, by * 16 - 2) : min(shape[1], by * 16 + 18),
                max(0, bx * 32 - 2) : min(shape[2], bx * 32 + 34),
            ]
            same &= bool(np.all(lut[region] == table.view(np.uint32)[-1]))
        if same:
            expected[by, bx] |= 1 << 20
    if classifier.coverage_guard:
        active = np.count_nonzero(original)
        admitted = np.count_nonzero(expected & (1 << 20))
        if not active or 4 * admitted < 3 * active:
            expected = original.copy()
    assert np.array_equal(output, expected)
    assert np.array_equal(masks, original)
    assert count == np.count_nonzero(expected & (1 << 20))
    for value, bits in zip(caller_arrays, caller_bits, strict=True):
        assert np.array_equal(value.view(np.uint8), bits)


@pytest.mark.parametrize("bad_tiles", (0, 1, 2, None))
def test_coverage_threshold_and_partial_masks_are_global(classifier, bad_tiles):
    n = 128
    shapes = ((n + 1, n + 1, n), (n + 1, n, n + 1), (n, n + 1, n + 1))
    table = np.array([1.0, 2.0], np.float32)
    codes = [np.ones(shape, np.uint8) for shape in shapes]
    masks = np.zeros((8, 4), np.int32)
    if bad_tiles is not None:
        masks[[0, 2, 4, 6], 1] = [15, 3, 8, 1]
    for by in (0, 2)[: bad_tiles or 0]:
        codes[1][12, by * 16 + 8, 48] = 0
    packed = []
    for value in codes:
        array = np.zeros((value.size + 3) // 4 * 4, np.uint8)
        array[: value.size] = value.ravel()
        packed.append(array.view(np.int32))
    bounds = np.array([10, 20], np.int32)
    original = masks.copy()
    caller_arrays = [*packed, table, bounds, masks]
    caller_bits = [value.view(np.uint8).copy() for value in caller_arrays]
    output = np.empty_like(masks)
    count = classifier(
        n,
        *(value.ctypes.data for value in packed),
        table.ctypes.data,
        len(table),
        bounds.ctypes.data,
        masks.ctypes.data,
        *masks.shape,
        output.ctypes.data,
    )
    expected = original.copy()
    # The independent oracle selects only the explicitly unmodified columns;
    # their checked halo boxes are disjoint from every corrupted point.
    eligible_rows = [] if bad_tiles is None else [0, 2, 4, 6][bad_tiles:]
    if not classifier.coverage_guard or len(eligible_rows) >= 3:
        expected[eligible_rows, 1] |= 1 << 20
    assert np.array_equal(output, expected)
    assert count == np.count_nonzero(expected & (1 << 20))
    for value, bits in zip(caller_arrays, caller_bits, strict=True):
        assert np.array_equal(value.view(np.uint8), bits)
