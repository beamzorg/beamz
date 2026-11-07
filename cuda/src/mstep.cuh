// Column tiles advancing S steps per pass (2S stages: H, E, H, E, ...).
// Eligibility and compact storage retain 8x16 quadrants. Stage K covers
// (X + 2S - 1 - K) x (Y + 2S - 1 - K) lanes starting -S + (K + 1) / 2 cells
// from the tile origin, on plane p - K / 2 of wave p; each keeps two planes.
template <int X, int Y, int S, int K> struct MstepStage {
  static constexpr int sx = X + 2 * S - 1 - K, sy = Y + 2 * S - 1 - K;
  static constexpr int origin = -S + (K + 1) / 2;
  static constexpr int offset = [] {
    int value = 0;
    for (int k = 0; k < K; ++k)
      value += 6 * (X + 2 * S - 1 - k) * (Y + 2 * S - 1 - k);
    return value;
  }();
  __device__ static __forceinline__ float &At(float *shared, int c, int z,
                                              int y, int x) {
    return shared[offset + ((c * 2 + (z & 1)) * sy + y) * sx + x];
  }
};
template <int X, int Y, int S = 2> struct MstepTile {
  static constexpr int wx = X + 2 * S, wy = Y + 2 * S;
  static constexpr int threads = wx * wy;
  // Stages 0..2S-2 live in shared memory; the last writes the output banks.
  static constexpr int input = MstepStage<X, Y, S, 2 * S - 1>::offset;
  static constexpr size_t shared = (input + 3 * threads) * sizeof(float);
  static constexpr size_t packed_shared = shared + 3 * threads * sizeof(float);
};
template <int X, int Y, int S>
__device__ __forceinline__ float &MstepInput(float *shared, int c, int y, int x) {
  return shared[MstepTile<X,Y,S>::input + c * MstepTile<X,Y,S>::threads + y * MstepTile<X,Y,S>::wx + x];
}
// Full field banks are canonical; compact addressing belongs only to edges.
__device__ __forceinline__ int MstepFieldOffset(const BeamzBuffer &f, int z,
                                                int y, int x) {
  return z * f.plane_stride + y * f.row_stride + x;
}
__device__ __forceinline__ float MstepFieldRead(const BeamzBuffer &f, int z,
                                                int y, int x) {
  return static_cast<const float *>(f.data)[MstepFieldOffset(f, z, y, x)];
}
struct MstepHScalars {
  float decay[3], source[3];
};
template <bool Packed, bool ScalarH, int X, int Y, int S, int K = 0>
__device__ __forceinline__ void
MstepStages(float *shared, const FusedYeePhase &h, const FusedYeePhase &e,
            const PairFieldOutputs &final, const MstepHScalars &hc, int ox,
            int oy, int oz, int wave, int mask) {
  using Here = MstepStage<X, Y, S, K>;
  constexpr int last = 2 * S - 1;
  constexpr int phase = K % 2, shift = phase == 1 ? 1 : 0;
  constexpr int sign = phase == 0 ? 1 : -1;
  const auto &launch = phase == 0 ? h : e;
  const int z = oz + wave - S - K / 2;
  if (wave >= K) {
    for (int i = threadIdx.x; i < Here::sx * Here::sy; i += blockDim.x) {
      const int ix = i % Here::sx, iy = i / Here::sx;
      const int x = ox + Here::origin + ix;
      const int y = oy + Here::origin + iy;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float next = 0.f;
        const auto &out = launch.outputs[c];
        if (BufferContains(out, z, y, x)) {
          auto other = [&](int component, int dz, int dy, int dx) {
            if constexpr (K == 0) {
              if (!dz)
                return MstepInput<X,Y,S>(shared, component, iy + dy, ix + dx);
              const auto &f = h.inputs[3 + component];
              return BufferContains(f, z + dz, y + dy, x + dx)
                         ? MstepFieldRead(f, z + dz, y + dy, x + dx)
                         : 0.f;
            } else {
              return MstepStage<X, Y, S, K - 1>::At(
                  shared, component, z + dz, iy + shift + dy, ix + shift + dx);
            }
          };
          auto difference = [&](int component, int axis) {
            const float a = other(component, 0, 0, 0);
            const float b = other(component, axis == 0 ? sign : 0,
                                  axis == 1 ? sign : 0, axis == 2 ? sign : 0);
            return beamz::cuda::yee::ScaleYeeDifference(
                phase == 0 ? b - a : a - b, launch.inv_resolution);
          };
          const auto a = beamz::cuda::yee::FirstCurlTerm(c);
          const auto b = beamz::cuda::yee::SecondCurlTerm(c);
          const float curl = difference(a.source_component, a.derivative_axis) -
                             difference(b.source_component, b.derivative_axis);
          float old;
          if constexpr (K == 1)
            old = MstepInput<X,Y,S>(shared, c, iy + 1, ix + 1);
          else if constexpr (K == 0)
            old = MstepFieldRead(launch.inputs[c], z, y, x);
          else
            old = MstepStage<X, Y, S, K - 2>::At(shared, c, z, iy + 1, ix + 1);
          const int logical = (z * static_cast<int>(out.dims[1]) + y) *
                                  static_cast<int>(out.dims[2]) +
                              x;
          float decay, source;
          if constexpr (phase == 0 && ScalarH) {
            decay = hc.decay[c];
            source = hc.source[c];
          } else if constexpr (phase == 1 && Packed) {
            decay = 1.f;
            source = beamz::cuda::yee::PackedMaterialSource(
                launch.inputs[6 + c], launch.inputs[9 + c], logical);
          } else {
            decay = Read(launch.inputs[6 + c], z, y, x);
            source = Read(launch.inputs[9 + c], z, y, x);
          }
          next = beamz::cuda::yee::AdvanceYeeField(phase, old, decay, source,
                                                   curl);
        }
        if constexpr (K == last) {
          const int quadrant = (iy >= 8 ? 2 : 0) + (ix >= 16 ? 1 : 0);
          if (mask & (1 << quadrant)) {
            const auto &ef = final.fields[3 + c];
            const auto &hf = final.fields[c];
            static_cast<float *>(ef.data)[MstepFieldOffset(ef, z, y, x)] = next;
            static_cast<float *>(hf.data)[MstepFieldOffset(hf, z, y, x)] =
                MstepStage<X, Y, S, last - 1>::At(shared, c, z, iy + 1, ix + 1);
          }
        } else
          Here::At(shared, c, z, iy, ix) = next;
      }
    }
  }
  __syncthreads();
  if constexpr (K < last)
    MstepStages<Packed, ScalarH, X, Y, S, K + 1>(shared, h, e, final, hc, ox,
                                                 oy, oz, wave, mask);
}

template <bool Packed, bool ScalarH, int X, int Y, int S>
__global__ __launch_bounds__((MstepTile<X,Y,S>::threads)) void MstepColumnCore(
    BEAMZ_GRID_ARG(FusedYeePhase) h,
    BEAMZ_GRID_ARG(FusedYeePhase) e,
    BEAMZ_GRID_ARG(PairFieldOutputs) final,
    BEAMZ_GRID_ARG(BeamzBuffer) masks,
    BEAMZ_GRID_ARG(BeamzBuffer) bounds, int segment) {
  const int ox = blockIdx.x * X, oy = blockIdx.y * Y;
  const int shift = ((oy % 16) / 8) * 2 + (ox % 32) / 16;
  const int mask = static_cast<const int *>(masks.data)[(oy / 16) * masks.dims[1] + ox / 32] >> shift;
  if (!mask)
    return;
  const int zb = static_cast<const int *>(bounds.data)[0] + blockIdx.z * segment;
  const int end = static_cast<const int *>(bounds.data)[1];
  const int zt = segment ? min(end, zb + segment) : end;
  if (zb >= zt) return;
  MstepHScalars hc{};
  if constexpr (ScalarH) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      hc.decay[c] = static_cast<const float *>(h.inputs[6 + c].data)[0];
      hc.source[c] = static_cast<const float *>(h.inputs[9 + c].data)[0];
    }
  }
  BEAMZ_DYNAMIC_SHARED(float, shared);
  for (int wave = 0; wave < zt - zb + 2 * S - 1; ++wave) {
    const int z = zb + wave - S;
    // One window lane per thread on GPUs (the block is MstepTile::threads).
    for (int i = threadIdx.x; i < MstepTile<X,Y,S>::threads; i += blockDim.x) {
      const int y = i / MstepTile<X,Y,S>::wx;
      const int x = i % MstepTile<X,Y,S>::wx;
      const int gy = oy - S + y;
      const int gx = ox - S + x;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        const auto &f = h.inputs[3 + c];
        MstepInput<X,Y,S>(shared, c, y, x) =
            BufferContains(f, z, gy, gx) ? MstepFieldRead(f, z, gy, gx) : 0.f;
      }
    }
    __syncthreads();
    MstepStages<Packed, ScalarH, X, Y, S>(shared, h, e, final, hc, ox, oy, zb,
                                          wave, mask);
  }
}

template <int X, int Y, int S>
__device__ __forceinline__ float &MstepPackedInput(float *shared, int c, int z, int y, int x) {
  return shared[MstepTile<X,Y,S>::input + ((z & 1) * 3 + c) * MstepTile<X,Y,S>::threads + y * MstepTile<X,Y,S>::wx + x];
}

// Lean core for packed E and scalar H coefficients. Buffer metadata is lowered
// once on the host; plane coordinates/clamping are invariant along the column.
// The planner proves every dependency of an owned quadrant lies inside all six
// fields. Clamping affects only discarded halo results of partially owned
// tiles.
struct MstepCoreField {
  float *data;
  int row, plane;
};
struct MstepPackedCoreView {
  MstepCoreField input[6], output[6];
  const uint32_t *codes[3];
  const float *tables[3], *h_decay[3], *h_source[3];
  int table_sizes[3], min_y, min_x;
  float inv;
};
struct MstepCoreLane {
  int ix, iy, x, y;
};
// Stage K's lane (iy, ix) of the tile at (ox, oy), clamped into the field.
template <int X, int Y, int S, int K>
__device__ __forceinline__ MstepCoreLane MstepMakeLane(int iy, int ix, int ox,
                                                       int oy, int ny, int nx) {
  constexpr int origin = MstepStage<X, Y, S, K>::origin;
  return {ix, iy, min(max(ox + origin + ix, 0), nx - 1),
          min(max(oy + origin + iy, 0), ny - 1)};
}
__device__ __forceinline__ int MstepCoreOffset(const MstepCoreField &f, int z,
                                               int y, int x) {
  return z * f.plane + y * f.row + x;
}

// Lanes are recomputed from the cell index and plane rather than carried per
// thread, so nothing per-thread crosses a barrier and the kernel is correct
// for any block size (see BEAMZ_BARRIER_BLOCK).
template <int X, int Y, int S, int K>
__device__ __forceinline__ void
MstepPackedStage(float *shared, const MstepPackedCoreView &v,
                 const MstepHScalars &hc, int ox, int oy, int z, int mask) {
  using Stage = MstepStage<X, Y, S, K>;
  constexpr int last = 2 * S - 1;
  constexpr int phase = K % 2, shift = phase == 1 ? 1 : 0;
  constexpr int sign = phase == 0 ? 1 : -1;
  // Loop-invariant material bases, so gathers from them can vectorise.
  const float *const tables[3] = {v.tables[0], v.tables[1], v.tables[2]};
  const uint32_t *const codes[3] = {v.codes[0], v.codes[1], v.codes[2]};
  const int table_sizes[3] = {v.table_sizes[0], v.table_sizes[1],
                              v.table_sizes[2]};
  BEAMZ_TILE_LOOP(iy, ix, Stage::sy, Stage::sx) {
    const auto lane =
        MstepMakeLane<X, Y, S, K>(iy, ix, ox, oy, v.min_y, v.min_x);
    BEAMZ_UNROLL
    for (int c = 0; c < 3; ++c) {
      auto other = [&](int component, int dz, int dy, int dx) {
        if constexpr (K == 0) {
          return MstepPackedInput<X,Y,S>(shared, component, z + dz,
                                         lane.iy + dy, lane.ix + dx);
        } else
          return MstepStage<X, Y, S, K - 1>::At(shared, component, z + dz,
                                                lane.iy + shift + dy,
                                                lane.ix + shift + dx);
      };
      auto difference = [&](int component, int axis) {
        float a = other(component, 0, 0, 0);
        float b = other(component, axis == 0 ? sign : 0, axis == 1 ? sign : 0,
                        axis == 2 ? sign : 0);
        return beamz::cuda::yee::ScaleYeeDifference(phase == 0 ? b - a : a - b,
                                                    v.inv);
      };
      // c is unrolled; the same component/axis convention as all CUDA phases.
      const auto a = beamz::cuda::yee::FirstCurlTerm(c);
      const auto b = beamz::cuda::yee::SecondCurlTerm(c);
      float curl = difference(a.source_component, a.derivative_axis) -
                   difference(b.source_component, b.derivative_axis);
      // Full input/output banks have identical canonical component pitches,
      // and E's packed codes follow the E input layout.
      const int logical = MstepCoreOffset(v.input[phase * 3 + c], z, lane.y,
                                          lane.x);
      float old;
      if constexpr (K == 0)
        old = v.input[c].data[logical];
      else if constexpr (K == 1)
        old = MstepPackedInput<X,Y,S>(shared, c, z, lane.iy + 1, lane.ix + 1);
      else
        old = MstepStage<X, Y, S, K - 2>::At(shared, c, z, lane.iy + 1,
                                             lane.ix + 1);
      float decay, source;
      if constexpr (phase == 0) {
        decay = hc.decay[c];
        source = hc.source[c];
      } else {
        decay = 1.f;
        source = beamz::cuda::yee::PackedMaterialSource(
            tables[c], table_sizes[c], codes[c], logical);
      }
      float next =
          beamz::cuda::yee::AdvanceYeeField(phase, old, decay, source, curl);
      if constexpr (K == last) {
        const int quadrant = (lane.iy >= 8 ? 2 : 0) + (lane.ix >= 16 ? 1 : 0);
        if (mask & (1 << quadrant)) {
          v.output[3 + c].data[logical] = next;
          v.output[c].data[MstepCoreOffset(v.output[c], z, lane.y, lane.x)] =
              MstepStage<X, Y, S, last - 1>::At(shared, c, z, lane.iy + 1,
                                                lane.ix + 1);
        }
      } else
        Stage::At(shared, c, z, lane.iy, lane.ix) = next;
    }
  }
  __syncthreads();
}

// Stages 2..2S-1 of wave p: stage K works on plane p - K / 2 once the wave
// has reached it.
template <int X, int Y, int S, int K>
__device__ __forceinline__ void
MstepPackedLaterStages(float *shared, const MstepPackedCoreView &v,
                       const MstepHScalars &hc, int ox, int oy, int p,
                       int wave, int mask) {
  if (wave >= K)
    MstepPackedStage<X, Y, S, K>(shared, v, hc, ox, oy, p - K / 2, mask);
  if constexpr (K < 2 * S - 1)
    MstepPackedLaterStages<X, Y, S, K + 1>(shared, v, hc, ox, oy, p, wave,
                                           mask);
}

// E planes of the input window, one lane per thread on GPUs.
template <int X, int Y, int S>
__device__ __forceinline__ void
MstepLoadWindow(float *shared, const MstepPackedCoreView &v, int ox, int oy,
                int z, bool asynchronous) {
  using Tile = MstepTile<X, Y, S>;
  for (int i = threadIdx.x; i < Tile::threads; i += blockDim.x) {
    const int wx = i % Tile::wx, wy = i / Tile::wx;
    const int gx = min(max(ox - S + wx, 0), v.min_x - 1);
    const int gy = min(max(oy - S + wy, 0), v.min_y - 1);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const auto &f = v.input[3 + c];
      float *destination = &MstepPackedInput<X,Y,S>(shared, c, z, wy, wx);
      const float *source = f.data + MstepCoreOffset(f, z, gy, gx);
      if (asynchronous)
        BEAMZ_ASYNC_COPY_F32(destination, source);
      else
        *destination = *source;
#if defined(BEAMZ_PLATFORM_CPU)
      // A wave reads short rows from many planes, more streams than the
      // hardware prefetchers follow: request the next plane's E and H lines.
      if ((wx & 15) == 0) {
        __builtin_prefetch(source + f.plane, 0, 1);
        const auto &h = v.input[c];
        __builtin_prefetch(h.data + MstepCoreOffset(h, z + 1, gy, gx), 0, 1);
      }
#endif
    }
  }
}

template <int X, int Y, int S>
__global__ BEAMZ_LAUNCH_BOUNDS((MstepTile<X,Y,S>::threads), 1) void MstepPackedColumnCore(
    const MstepPackedCoreView v, const int *masks, const int *bounds, int ntx, int segment) {
  const int ox = blockIdx.x * X, oy = blockIdx.y * Y;
  const int shift = ((oy % 16) / 8) * 2 + (ox % 32) / 16;
  const int mask = masks[(oy / 16) * ntx + ox / 32] >> shift;
  if (!mask)
    return;
  const int zb = bounds[0] + blockIdx.z * segment;
  const int zt = segment ? min(bounds[1], zb + segment) : bounds[1];
  if (zb >= zt) return;
  MstepHScalars hc;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    hc.decay[c] = *v.h_decay[c];
    hc.source[c] = *v.h_source[c];
  }
  BEAMZ_DYNAMIC_SHARED(float, shared);
  // Two E0 planes rotate through the input ring. Prefetch p+2 after E1's last
  // use of p, while later stages consume separate shared rings. The block
  // waits before the next wave may read a neighbour's asynchronous copy.
  MstepLoadWindow<X,Y,S>(shared, v, ox, oy, zb - S, false);
  MstepLoadWindow<X,Y,S>(shared, v, ox, oy, zb - S + 1, false);
  __syncthreads();
  const int waves = zt - zb + 2 * S - 1;
  for (int wave = 0; wave < waves; ++wave) {
    int p = zb + wave - S;
    MstepPackedStage<X,Y,S,0>(shared, v, hc, ox, oy, p, mask);
    if (wave >= 1)
      MstepPackedStage<X,Y,S,1>(shared, v, hc, ox, oy, p, mask);
    if (wave + 1 < waves) {
      MstepLoadWindow<X,Y,S>(shared, v, ox, oy, p + 2, true);
      BEAMZ_ASYNC_COMMIT();
    }
    MstepPackedLaterStages<X,Y,S,2>(shared, v, hc, ox, oy, p, wave, mask);
    if (wave + 1 < waves) {
      BEAMZ_ASYNC_WAIT_ALL();
      __syncthreads();
    }
  }
}

template <int Phase, bool Cpml, bool Packed, bool UniformCpml = false>
__global__ void MstepEdgePhase(BEAMZ_GRID_ARG(BeamzLaunch) launch,
                               BEAMZ_GRID_ARG(BeamzBuffer) cells,
                               int ny, int nx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells.dims[0])
    return;
  const int row = static_cast<const int *>(cells.data)[i];
  if (row < 0)  // Padding of a shorter edge level.
    return;
  // Two unsigned divisions; remainders by multiplication.
  const unsigned zy = static_cast<unsigned>(row) / static_cast<unsigned>(nx);
  const unsigned uz = zy / static_cast<unsigned>(ny);
  const int x = row - static_cast<int>(zy) * nx;
  const int y = static_cast<int>(zy - uz * static_cast<unsigned>(ny));
  const int z = static_cast<int>(uz);
  UpdateComponent<Phase, 0, Cpml, 0, true, UniformCpml, -1, Packed, true>(launch, z,
                                                                    y, x);
  UpdateComponent<Phase, 1, Cpml, 0, true, UniformCpml, -1, Packed, true>(launch, z,
                                                                    y, x);
  UpdateComponent<Phase, 2, Cpml, 0, true, UniformCpml, -1, Packed, true>(launch, z,
                                                                    y, x);
}
