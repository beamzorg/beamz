"""Prove direct raw IDs against an independent dictionary, including padding."""

import numpy as np
import pytest

from scripts.cuda_sparse_observation import build_sparse_observation, lookup_sparse


def test_sparse_lookup_exhaustive_staggered_fields_preserves_original_raw_ids():
    shapes = np.array(
        [[5, 7, 10], [5, 8, 9], [6, 7, 9], [6, 8, 9], [6, 7, 10], [5, 8, 10]]
    )
    prefix = np.r_[0, np.cumsum(np.prod(shapes, axis=1))]
    rng = np.random.default_rng(231)
    keys = np.sort(rng.choice(int(prefix[-1]), 213, replace=False)).astype(np.int64)
    plan = build_sparse_observation(shapes, keys)
    expected = {}
    for raw, key in enumerate(keys):
        c = next(c for c in range(6) if key < prefix[c + 1])
        p = int(key - prefix[c])
        nz, ny, nx = shapes[c]
        expected[c, p // (ny * nx), (p // nx) % ny, p % nx] = raw
    for c in range(6):
        for z in range(7):
            for y in range(9):
                for x in range(11):
                    assert lookup_sparse(plan, c, z, y, x) == expected.get(
                        (c, z, y, x), -1
                    )
    assert plan["extra_gpu_bytes"] == plan["coarse"].nbytes + plan["leaf"].nbytes


def test_sparse_lookup_empty_and_rejects_duplicate_invalid_or_reordered_keys():
    shapes = [(4, 5, 6)] * 6
    plan = build_sparse_observation(shapes, np.array([], np.int64))
    assert plan["leaf"].size == 0
    assert lookup_sparse(plan, 0, 0, 0, 0) == -1
    for keys in [[0, 0], [1, 0], [-1], [720]]:
        with pytest.raises(ValueError):
            build_sparse_observation(shapes, np.asarray(keys, np.int64))


def test_sparse_lookup_rejects_huge_offsets_before_allocating():
    empty = np.array([], np.int64)
    with pytest.raises(ValueError, match="Field offsets"):
        build_sparse_observation([(2**31, 2**31, 2)] * 6, empty)
    shapes = [(2**30, 1, 1), (1, 2**30, 1), (1, 1, 2**30)] * 2
    with pytest.raises(ValueError, match="Coarse observation offsets"):
        build_sparse_observation(shapes, empty)


def test_compiled_native_sparse_map_matches_independent_numpy_builder(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path

    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "map.cc"
    source.write_text(r"""
#include "native_moving.h"
#include <fstream>
int main(int argc,char**argv){
 BeamzBuffer f[6]{};int shape[6][3]={{5,7,10},{5,8,9},{6,7,9},{6,8,9},{6,7,10},{5,8,10}};
 int64_t total=0;for(int c=0;c<6;++c){f[c].rank=3;int64_t count=1;for(int a=0;a<3;++a){f[c].dims[a]=shape[c][a];count*=shape[c][a];}total+=count;}
 std::vector<int64_t>keys;for(int64_t k=0;k<total;k+=17)keys.push_back(k);
 const auto p=BeamzBuildNativeMovingSparseObservation(f,keys);
 int32_t header[]={p.grid[0],p.grid[1],p.grid[2],int32_t(p.coarse.size()),int32_t(p.leaf.size())};
 std::ofstream out(argv[1],std::ios::binary);out.write((const char*)header,sizeof(header));out.write((const char*)p.coarse.data(),p.coarse.size()*4);out.write((const char*)p.leaf.data(),p.leaf.size()*4);
}
""")
    binary = tmp_path / "map"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(root / "cuda/src"),
            str(source),
            str(root / "cuda/src/native_moving.cc"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    output = tmp_path / "map.bin"
    subprocess.run([str(binary), str(output)], check=True)
    actual = np.fromfile(output, dtype=np.int32)
    shapes = np.array(
        [[5, 7, 10], [5, 8, 9], [6, 7, 9], [6, 8, 9], [6, 7, 10], [5, 8, 10]]
    )
    keys = np.arange(0, int(np.prod(shapes, axis=1).sum()), 17, dtype=np.int64)
    expected = build_sparse_observation(shapes, keys)
    assert tuple(actual[:3]) == expected["grid_zyx"]
    assert np.array_equal(actual[5 : 5 + actual[3]], expected["coarse"])
    assert np.array_equal(actual[5 + actual[3] :], expected["leaf"].ravel())
