// Analytical checks of the streamed production stencil, including poisoned
// storage padding. This is compiled as ordinary C++ on CPU-only machines.
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "boundary.h"

int main() {
  const int dimensions[3] = {3, 4, 5};
  for (int axis = 0; axis < 3; ++axis) {
    std::vector<float> storage(3 * 48,
                              std::numeric_limits<float>::quiet_NaN());
    BeamzBuffer value{};
    value.data = storage.data();
    value.rank = 3;
    value.row_stride = 8;
    value.plane_stride = 48;
    for (int d = 0; d < 3; ++d) value.dims[d] = dimensions[d];
    for (int z = 0; z < dimensions[0]; ++z)
      for (int y = 0; y < dimensions[1]; ++y)
        for (int x = 0; x < dimensions[2]; ++x) {
          const int coordinate[3] = {z, y, x};
          storage[BeamzOffset3D(value, z, y, x)] =
              0.75f + 3.0f * coordinate[axis];
        }
    for (int wall = 0; wall < 3; ++wall) {  // open, PEC, PMC
      for (bool high : {false, true}) {
        const int bit = 1 << (2 * axis + int(high));
        const int pec = wall == 1 ? bit : 0;
        const int pmc = wall == 2 ? bit : 0;
        if (!beamz::cuda::boundary::ValidFaces(pec, pmc)) return 1;
        for (int coordinate = 0; coordinate <= dimensions[axis]; ++coordinate) {
          int p[3] = {1, 1, 1};
          p[axis] = coordinate;
          // A varying inverse distance also checks metric scaling at both walls.
          const float scale = 0.2f + 0.125f * coordinate;
          float expected = 3.0f;
          if (coordinate == 0)
            expected = high ? 0.f : wall * 0.75f;
          if (coordinate == dimensions[axis])
            expected = high ? -wall * (0.75f + 3.f * (coordinate - 1)) : 0.f;
          const float actual = beamz::cuda::boundary::BackwardDifference(
              value, axis, p[0], p[1], p[2], pec, pmc, scale);
          if (!std::isfinite(actual) || std::abs(actual - expected * scale) > 1e-6f) {
            std::fprintf(stderr, "Boundary stencil mismatch: axis=%d wall=%d high=%d coordinate=%d\n",
                         axis, wall, int(high), coordinate);
            return 2;
          }
        }
      }
    }
  }
  if (beamz::cuda::boundary::ValidFaces(1, 1) ||
      beamz::cuda::boundary::ValidFaces(64, 0) ||
      beamz::cuda::boundary::ValidFaces(0, -1)) return 3;
  return 0;
}
