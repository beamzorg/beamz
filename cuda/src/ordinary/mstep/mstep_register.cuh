// Experimental packed D2 core: fixed physical-XY lanes, one shared plane per
// intermediate stage, and short-lived register histories. Included by mstep.cuh
// only when BEAMZ_CUDA_MSTEP_REGISTER_CORE is enabled. Numerical updates still
// use MstepCellAdvance and the canonical packed-material decoder.
template <int X, int Y> struct MstepRegisterTile {
  static constexpr int wx = X + 4, wy = Y + 4;
  static constexpr int threads = 256;
  static constexpr int slots = (wx * wy + threads - 1) / threads;
  static constexpr int h1_area = (X + 3) * (Y + 3);
  static constexpr int e1_area = (X + 2) * (Y + 2);
  static constexpr int h2_area = (X + 1) * (Y + 1);
  static constexpr int stage_floats = 3 * (h1_area + e1_area + h2_area);
  static constexpr int shared_bytes =
      (stage_floats + 6 * wx * wy) * sizeof(float);
};

template <int X, int Y, int K>
__device__ __forceinline__ float &MstepRegisterShared(float *shared, int c,
                                                      int qy, int qx) {
  using Tile = MstepRegisterTile<X, Y>;
  constexpr int low = (K + 1) / 2;
  constexpr int sx = X + 3 - K, sy = Y + 3 - K;
  constexpr int base = K == 0 ? 0 :
      (K == 1 ? 3 * Tile::h1_area : 3 * (Tile::h1_area + Tile::e1_area));
  return shared[base + (c * sy + qy - low) * sx + qx - low];
}

template <int X, int Y>
__device__ __forceinline__ float &MstepRegisterInput(float *shared, int c,
                                                     int z, int qy, int qx) {
  using Tile = MstepRegisterTile<X, Y>;
  return shared[Tile::stage_floats +
                ((c * 2 + (z & 1)) * Tile::wy + qy) * Tile::wx + qx];
}

template <int X, int Y, int Layout = 0>
__global__ __launch_bounds__(
    256, BEAMZ_CUDA_MSTEP_REGISTER_FOUR_CTAS && X == 16 && Y == 8 ? 4 : 1)
void MstepRegisterColumnCore(
    const MstepPackedCoreView v, const int *masks, const int *bounds, int ntx,
    int segment) {
  using Tile = MstepRegisterTile<X, Y>;
  constexpr int Slots = Tile::slots;
  const int ox = blockIdx.x * X, oy = blockIdx.y * Y;
  const int shift = ((oy % 16) / 8) * 2 + (ox % 32) / 16;
  const int mask = masks[(oy / 16) * ntx + ox / 32] >> shift;
  if (!mask) return;
  const int zb = bounds[0] + blockIdx.z * segment;
  const int zt = segment ? min(bounds[1], zb + segment) : bounds[1];
  if (zb >= zt) return;

  int qx[Slots], qy[Slots];
  bool valid[Slots], owned[Slots];
  int h_offset[3][Slots], e_offset[3][Slots];
  MstepHScalars hc;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    hc.decay[c] = *v.h_decay[c];
    hc.source[c] = *v.h_source[c];
  }
#pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    const int q = threadIdx.x + slot * Tile::threads;
    qx[slot] = q % Tile::wx;
    qy[slot] = q / Tile::wx;
    valid[slot] = q < Tile::wx * Tile::wy;
    const int x = min(max(ox - 2 + qx[slot], 0), v.min_x - 1);
    const int y = min(max(oy - 2 + qy[slot], 0), v.min_y - 1);
    const int ix = qx[slot] - 2, iy = qy[slot] - 2;
    const int quadrant = (iy >= 8 ? 2 : 0) + (ix >= 16 ? 1 : 0);
    owned[slot] = ix >= 0 && ix < X && iy >= 0 && iy < Y &&
                  (mask & (1 << quadrant));
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      h_offset[c][slot] = MstepCoreOffset(v.input[c], zb - 2, y, x);
      e_offset[c][slot] = MstepCoreOffset(v.input[3 + c], zb - 2, y, x);
    }
  }
  extern __shared__ float shared[];
#pragma unroll
  for (int slot = 0; slot < Slots; ++slot) {
    if (!valid[slot]) continue;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const auto &f = v.input[3 + c];
      MstepRegisterInput<X, Y>(shared, c, zb - 2, qy[slot], qx[slot]) =
          f.data[e_offset[c][slot]];
      MstepRegisterInput<X, Y>(shared, c, zb - 1, qy[slot], qx[slot]) =
          f.data[e_offset[c][slot] + f.plane];
    }
  }
  __syncthreads();

  const int waves = zt - zb + 3;
  for (int wave = 0; wave < waves; ++wave) {
    const int p = zb + wave - 2;
    // History is retired from shared memory afresh in this wave; none of
    // these values crosses a wave boundary. Keep their lifetimes local so
    // the allocator need not preserve predicated loop-carried array values.
    float current_h[3][Slots] = {}, current_e[3][Slots] = {};
    float old_h[3][Slots] = {}, old_e[3][Slots] = {};
    // H1 needs the previous H1 sample only at its own physical XY. No CTA may
    // still read H1 from the preceding wave: the intervening H2 publication
    // barrier completed all E1 reads. Thus each lane may retire and overwrite
    // its H1 slot directly, then synchronize once for the current E1 stencil.
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot) {
      const int x = qx[slot], y = qy[slot];
      if (x >= X + 3 || y >= Y + 3) continue;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        if (wave)
          old_h[c][slot] = MstepRegisterShared<X, Y, 0>(shared, c, y, x);
        const auto other = [&](int component, int dz, int dy, int dx) {
          return MstepRegisterInput<X, Y>(shared, component, p + dz, y + dy, x + dx);
        };
        current_h[c][slot] = MstepCellAdvance<0>(
            c, v.input[c].data[h_offset[c][slot]], hc.decay[c], hc.source[c],
            v.inv, other);
        MstepRegisterShared<X, Y, 0>(shared, c, y, x) = current_h[c][slot];
      }
    }
    __syncthreads();

    if (wave >= 1) {
#pragma unroll
      for (int slot = 0; slot < Slots; ++slot) {
        const int x = qx[slot], y = qy[slot];
        if (x < 1 || y < 1 || x >= X + 3 || y >= Y + 3) continue;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          const auto other = [&](int component, int dz, int dy, int dx) {
            if (dz) return old_h[component][slot];
            if (dy || dx)
              return MstepRegisterShared<X, Y, 0>(shared, component, y + dy, x + dx);
            return current_h[component][slot];
          };
          const float source = MstepMaterialSource<Layout>(v, c, e_offset[c][slot]);
          current_e[c][slot] = MstepCellAdvance<1>(
              c, MstepRegisterInput<X, Y>(shared, c, p, y, x), 1.f,
              source, v.inv, other);
        }
      }
    }

    // All transverse E0 reads finished at H1's barrier. E1 reads only its own
    // E0 cell, so a lane can prefetch p+2 into its retired p slot after E1.
    if (wave + 1 < waves) {
#pragma unroll
      for (int slot = 0; slot < Slots; ++slot) {
        if (!valid[slot]) continue;
#pragma unroll
        for (int c = 0; c < 3; ++c)
          __pipeline_memcpy_async(
              &MstepRegisterInput<X, Y>(shared, c, p + 2, qy[slot], qx[slot]),
              v.input[3 + c].data + e_offset[c][slot] +
                  2 * v.input[3 + c].plane,
              sizeof(float));
      }
      __pipeline_commit();
    }

    if (wave >= 2) {
#pragma unroll
      for (int slot = 0; slot < Slots; ++slot) {
        const int x = qx[slot], y = qy[slot];
        if (x < 1 || y < 1 || x >= X + 2 || y >= Y + 2) continue;
#pragma unroll
        for (int c = 0; c < 3; ++c)
          old_e[c][slot] = MstepRegisterShared<X, Y, 1>(shared, c, y, x);
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          const auto other = [&](int component, int dz, int dy, int dx) {
            if (dz) return current_e[component][slot];
            if (dy || dx)
              return MstepRegisterShared<X, Y, 1>(shared, component, y + dy, x + dx);
            return old_e[component][slot];
          };
          const float next = MstepCellAdvance<0>(
              c, old_h[c][slot], hc.decay[c], hc.source[c], v.inv, other);
          // E2 needs the prior H2 z sample. Save it before publishing H2.
          if (wave >= 3)
            old_h[c][slot] = MstepRegisterShared<X, Y, 2>(shared, c, y, x);
          current_h[c][slot] = next;
        }
      }
      // H2 has now consumed all prior-plane E1 transverse neighbours.
      __syncthreads();
    }
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot) {
      const int x = qx[slot], y = qy[slot];
      if (wave >= 2 && x >= 1 && y >= 1 && x < X + 2 && y < Y + 2) {
#pragma unroll
        for (int c = 0; c < 3; ++c)
          MstepRegisterShared<X, Y, 2>(shared, c, y, x) = current_h[c][slot];
      }
      if (wave >= 1 && x >= 1 && y >= 1 && x < X + 3 && y < Y + 3) {
#pragma unroll
        for (int c = 0; c < 3; ++c)
          MstepRegisterShared<X, Y, 1>(shared, c, y, x) = current_e[c][slot];
      }
    }
    // One barrier certifies both H2/E1 publication and next-wave E0 readiness.
    if (wave + 1 < waves) __pipeline_wait_prior(0);
    __syncthreads();

    if (wave >= 3) {
#pragma unroll
      for (int slot = 0; slot < Slots; ++slot) {
        const int x = qx[slot], y = qy[slot];
        if (!owned[slot]) continue;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
          const auto other = [&](int component, int dz, int dy, int dx) {
            if (dz) return old_h[component][slot];
            if (dy || dx)
              return MstepRegisterShared<X, Y, 2>(shared, component, y + dy, x + dx);
            return current_h[component][slot];
          };
          const int e_linear = e_offset[c][slot] - v.input[3 + c].plane;
          const float source = MstepMaterialSource<Layout>(v, c, e_linear);
          v.output[3 + c].data[e_linear] = MstepCellAdvance<1>(
              c, old_e[c][slot], 1.f, source, v.inv, other);
          v.output[c].data[h_offset[c][slot] - v.input[c].plane] = current_h[c][slot];
        }
      }
    }
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot) {
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        h_offset[c][slot] += v.input[c].plane;
        e_offset[c][slot] += v.input[3 + c].plane;
      }
    }
  }
}
