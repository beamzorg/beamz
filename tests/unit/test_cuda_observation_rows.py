"""Check production row lookups against the sorted input keys."""

import shutil
import subprocess

import numpy as np
import pytest

from tests.cuda_source_layout import CUDA_INCLUDE_FLAGS


@pytest.fixture(scope="module")
def row_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    folder = tmp_path_factory.mktemp("row-observation")
    source = folder / "rows.cc"
    source.write_text(r"""
#include "native_moving.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include "native_observation_rows.inc"

int main(int argc, char** argv) {
  std::ifstream in(argv[1], std::ios::binary);
  BeamzBuffer fields[6]{};
  for (auto& field : fields) {
    field.rank = 3;
    for (int axis = 0; axis < 3; ++axis) {
      int64_t extent;
      in.read(reinterpret_cast<char*>(&extent), sizeof(extent));
      field.dims[axis] = extent;
    }
  }
  int64_t count;
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  std::vector<int64_t> keys(count);
  in.read(reinterpret_cast<char*>(keys.data()), keys.size() * sizeof(int64_t));
  if (!in) return 1;

  try {
    const auto estimate = BeamzBuildNativeMovingRowObservation<false>(fields, keys);
    const auto rows = BeamzBuildNativeMovingRowObservation(fields, keys);
    if (!estimate.coarse.empty() || !estimate.leaf.empty() ||
        estimate.origin != rows.origin || estimate.grid != rows.grid ||
        estimate.device_bytes != rows.device_bytes ||
        rows.device_bytes != rows.coarse.size() * sizeof(int32_t) +
                             rows.leaf.size() * sizeof(uint64_t)) return 3;

    int64_t key = 0;
    for (int field = 0; field < 6; ++field)
      for (int z = 0; z < fields[field].dims[0]; ++z)
        for (int y = 0; y < fields[field].dims[1]; ++y)
          for (int x = 0; x < fields[field].dims[2]; ++x, ++key) {
            const auto position = std::lower_bound(keys.begin(), keys.end(), key);
            const int raw = position != keys.end() && *position == key
                                ? int(position - keys.begin()) : -1;
            if (BeamzMovingRawOffset(rows, field, z, y, x) != raw) return 4;
          }

    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(rows.grid.data()), 3 * sizeof(int));
    out.write(reinterpret_cast<const char*>(rows.origin.data()), 3 * sizeof(int));
    return out ? 0 : 1;
  } catch (const std::invalid_argument&) {
    return 2;
  }
}
""")
    binary = folder / "rows"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", *CUDA_INCLUDE_FLAGS, source, "-o", binary],
        check=True,
    )
    return binary


def _probe(binary, tmp_path, dimensions, keys, *, check=True):
    packet = tmp_path / "keys.bin"
    output = tmp_path / "rows.bin"
    packet.write_bytes(
        np.asarray(dimensions, np.int64).tobytes()
        + np.asarray([len(keys)], np.int64).tobytes()
        + np.asarray(keys, np.int64).tobytes()
    )
    result = subprocess.run([binary, packet, output], check=check)
    return np.fromfile(output, np.int32) if check else result.returncode


@pytest.mark.parametrize("shape", [(3, 5, 17), (9, 11, 33), (1, 1, 1)])
@pytest.mark.parametrize("density", [0, 0.03, 0.5, 1])
def test_sorted_row_rank_matches_arbitrary_sparse_and_dense_keys(
    row_probe, tmp_path, shape, density
):
    dimensions = np.asarray(shape) + [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]
    rng = np.random.default_rng(714)
    keys = np.flatnonzero(rng.random(np.prod(dimensions, axis=1).sum()) < density)
    _probe(row_probe, tmp_path, dimensions, keys)


@pytest.mark.parametrize("keys", [[1, 1], [2, 1], [-1], [10**12]])
def test_malformed_key_order_or_support_is_rejected(row_probe, tmp_path, keys):
    assert _probe(row_probe, tmp_path, [(3, 3, 3)] * 6, keys, check=False) == 2


@pytest.mark.parametrize(
    "dimensions",
    [
        [(2**31, 2**31, 2)] * 6,
        [(2**30, 1, 1), (1, 2**30, 1), (1, 1, 2**30)] * 2,
    ],
)
def test_oversized_row_offsets_are_rejected(row_probe, tmp_path, dimensions):
    assert _probe(row_probe, tmp_path, dimensions, [], check=False) == 2


def test_offset_mode_monitor_only_allocates_its_observed_tile_box(row_probe, tmp_path):
    shape = (129, 65, 145)
    keys = sorted(
        field * int(np.prod(shape)) + np.ravel_multi_index((z, y, x), shape)
        for field in (0, 3, 5)
        for z in (67, 68)
        for y in (40, 47)
        for x in (113, 127, 128)
    )
    layout = _probe(row_probe, tmp_path, [shape] * 6, keys)
    np.testing.assert_array_equal(layout, (2, 1, 2, 67, 5, 7))
