// Two-step column tiles. Eligibility and compact storage retain 8x16 quadrants.
template <int X, int Y> struct MstepTile {
  static constexpr int wx = X + 4, wy = Y + 4;
  static constexpr int threads = wx * wy;
  static constexpr int input = PairStage<X, Y, 3>::offset;
  static constexpr size_t shared = (input + 3 * threads) * sizeof(float);
  static constexpr size_t packed_shared = shared + 3 * threads * sizeof(float);
};
template <int X, int Y>
__device__ __forceinline__ float &MstepInput(float *shared, int c, int y, int x) {
  return shared[MstepTile<X,Y>::input + c * MstepTile<X,Y>::threads + y * MstepTile<X,Y>::wx + x];
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
template <bool Packed, bool ScalarH, int X, int Y, int K = 0>
__device__ __forceinline__ void
MstepStages(float *shared, const FusedYeePhase &h, const FusedYeePhase &e,
            const PairFieldOutputs &final, const MstepHScalars &hc, int ox,
            int oy, int oz, int wave, int mask) {
  using Here = PairStage<X, Y, K>;
  constexpr int phase = K % 2, shift = phase == 1 ? 1 : 0;
  constexpr int sign = phase == 0 ? 1 : -1;
  const auto &launch = phase == 0 ? h : e;
  const int z = oz + wave - 2 - K / 2;
  if (wave >= K) {
    for (int i = threadIdx.x; i < Here::sx * Here::sy; i += blockDim.x) {
      const int ix = i % Here::sx, iy = i / Here::sx;
      const int x = ox - 2 + (K + 1) / 2 + ix;
      const int y = oy - 2 + (K + 1) / 2 + iy;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float next = 0.f;
        const auto &out = launch.outputs[c];
        if (BufferContains(out, z, y, x)) {
          auto other = [&](int component, int dz, int dy, int dx) {
            if constexpr (K == 0) {
              if (!dz)
                return MstepInput<X,Y>(shared, component, iy + dy, ix + dx);
              const auto &f = h.inputs[3 + component];
              return BufferContains(f, z + dz, y + dy, x + dx)
                         ? MstepFieldRead(f, z + dz, y + dy, x + dx)
                         : 0.f;
            } else {
              return PairStage<X, Y, K - 1>::At(
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
            old = MstepInput<X,Y>(shared, c, iy + 1, ix + 1);
          else if constexpr (K == 0)
            old = MstepFieldRead(launch.inputs[c], z, y, x);
          else
            old = PairStage<X, Y, K - 2>::At(shared, c, z, iy + 1, ix + 1);
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
        if constexpr (K == 3) {
          const int quadrant = (iy >= 8 ? 2 : 0) + (ix >= 16 ? 1 : 0);
          if (mask & (1 << quadrant)) {
            const auto &ef = final.fields[3 + c];
            const auto &hf = final.fields[c];
            static_cast<float *>(ef.data)[MstepFieldOffset(ef, z, y, x)] = next;
            static_cast<float *>(hf.data)[MstepFieldOffset(hf, z, y, x)] =
                PairStage<X, Y, 2>::At(shared, c, z, iy + 1, ix + 1);
          }
        } else
          Here::At(shared, c, z, iy, ix) = next;
      }
    }
  }
  __syncthreads();
  if constexpr (K < 3)
    MstepStages<Packed, ScalarH, X, Y, K + 1>(shared, h, e, final, hc, ox, oy, oz,
                                        wave, mask);
}

template <bool Packed, bool ScalarH, int X, int Y>
__global__ __launch_bounds__((MstepTile<X,Y>::threads)) void MstepColumnCore(
    const __grid_constant__ FusedYeePhase h,
    const __grid_constant__ FusedYeePhase e,
    const __grid_constant__ PairFieldOutputs final,
    const __grid_constant__ BeamzBuffer masks,
    const __grid_constant__ BeamzBuffer bounds, int segment) {
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
  extern __shared__ float shared[];
  for (int wave = 0; wave < zt - zb + 3; ++wave) {
    const int y = threadIdx.x / MstepTile<X,Y>::wx;
    const int x = threadIdx.x % MstepTile<X,Y>::wx;
    const int z = zb + wave - 2;
    const int gy = oy - 2 + y;
    const int gx = ox - 2 + x;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const auto &f = h.inputs[3 + c];
      MstepInput<X,Y>(shared, c, y, x) =
          BufferContains(f, z, gy, gx) ? MstepFieldRead(f, z, gy, gx) : 0.f;
    }
    __syncthreads();
    MstepStages<Packed, ScalarH, X, Y>(shared, h, e, final, hc, ox,
                                 oy, zb, wave, mask);
  }
}

template <int X, int Y>
__device__ __forceinline__ float &MstepPackedInput(float *shared, int c, int z, int y, int x) {
  return shared[MstepTile<X,Y>::input + ((z & 1) * 3 + c) * MstepTile<X,Y>::threads + y * MstepTile<X,Y>::wx + x];
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
template <int X, int Y, int K>
__device__ __forceinline__ MstepCoreLane MstepMakeLane(int ox, int oy, int ny,
                                                       int nx) {
  using Stage = PairStage<X, Y, K>;
  int ix = threadIdx.x % Stage::sx, iy = threadIdx.x / Stage::sx;
  return {ix, iy, min(max(ox - 2 + (K + 1) / 2 + ix, 0), nx - 1),
          min(max(oy - 2 + (K + 1) / 2 + iy, 0), ny - 1)};
}
__device__ __forceinline__ int MstepCoreOffset(const MstepCoreField &f, int z,
                                               int y, int x) {
  return z * f.plane + y * f.row + x;
}

template <int X, int Y, int K>
__device__ __forceinline__ void
MstepPackedStage(float *shared, const MstepPackedCoreView &v,
                 const MstepHScalars &hc, const MstepCoreLane &lane, int z,
                 bool owned, const int (&logical)[3],
                 const int (&h_output)[3]) {
  using Stage = PairStage<X, Y, K>;
  constexpr int phase = K % 2, shift = phase == 1 ? 1 : 0;
  constexpr int sign = phase == 0 ? 1 : -1;
  if (threadIdx.x < Stage::sx * Stage::sy) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      auto other = [&](int component, int dz, int dy, int dx) {
        if constexpr (K == 0) {
          return MstepPackedInput<X,Y>(shared, component, z + dz, lane.iy + dy,
                                  lane.ix + dx);
        } else
          return PairStage<X, Y, K - 1>::At(shared, component, z + dz,
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
      float old;
      if constexpr (K == 0) {
        const auto &f = v.input[c];
        old = f.data[logical[c]];
      } else if constexpr (K == 1)
        old = MstepPackedInput<X,Y>(shared, c, z, lane.iy + 1, lane.ix + 1);
      else
        old = PairStage<X, Y, K - 2>::At(shared, c, z, lane.iy + 1,
                                           lane.ix + 1);
      float decay, source;
      if constexpr (phase == 0) {
        decay = hc.decay[c];
        source = hc.source[c];
      } else {
        decay = 1.f;
        source = beamz::cuda::yee::PackedMaterialSource(
            v.tables[c], v.table_sizes[c], v.codes[c], logical[c]);
      }
      float next =
          beamz::cuda::yee::AdvanceYeeField(phase, old, decay, source, curl);
      if constexpr (K == 3) {
        if (owned) {
          const auto &e = v.output[3 + c];
          const auto &h = v.output[c];
          e.data[logical[c]] = next;
          h.data[h_output[c]] =
              PairStage<X, Y, 2>::At(shared, c, z, lane.iy + 1, lane.ix + 1);
        }
      } else
        Stage::At(shared, c, z, lane.iy, lane.ix) = next;
    }
  }
  __syncthreads();
}

template <int X, int Y>
__global__ BEAMZ_LAUNCH_BOUNDS((MstepTile<X,Y>::threads), 1) void MstepPackedColumnCore(
    const MstepPackedCoreView v, const int *masks, const int *bounds, int ntx, int segment) {
  const int ox = blockIdx.x * X, oy = blockIdx.y * Y;
  const int shift = ((oy % 16) / 8) * 2 + (ox % 32) / 16;
  const int mask = masks[(oy / 16) * ntx + ox / 32] >> shift;
  if (!mask)
    return;
  const int zb = bounds[0] + blockIdx.z * segment;
  const int zt = segment ? min(bounds[1], zb + segment) : bounds[1];
  if (zb >= zt) return;
  const auto l0 = MstepMakeLane<X,Y,0>(ox, oy, v.min_y, v.min_x);
  const auto l1 = MstepMakeLane<X,Y,1>(ox, oy, v.min_y, v.min_x);
  const auto l2 = MstepMakeLane<X,Y,2>(ox, oy, v.min_y, v.min_x);
  const auto l3 = MstepMakeLane<X,Y,3>(ox, oy, v.min_y, v.min_x);
  const int quadrant = (l3.iy >= 8 ? 2 : 0) + (l3.ix >= 16 ? 1 : 0);
  const bool owned = (mask & (1 << quadrant)) != 0;
  const int wx = threadIdx.x % MstepTile<X,Y>::wx;
  const int wy = threadIdx.x / MstepTile<X,Y>::wx;
  const int gx = min(max(ox - 2 + wx, 0), v.min_x - 1);
  const int gy = min(max(oy - 2 + wy, 0), v.min_y - 1);
  MstepHScalars hc;
  // Carry linear addresses along the column rather than recomputing the
  // invariant y/x products in every stage of every wave. Full input/output
  // banks have identical canonical component pitches.
  int old_h[3], first_e[3], final_e[3], final_h[3], next_input[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    hc.decay[c] = *v.h_decay[c];
    hc.source[c] = *v.h_source[c];
    old_h[c] = MstepCoreOffset(v.input[c], zb - 2, l0.y, l0.x);
    first_e[c] = MstepCoreOffset(v.input[3 + c], zb - 2, l1.y, l1.x);
    final_e[c] = MstepCoreOffset(v.input[3 + c], zb - 3, l3.y, l3.x);
    final_h[c] = MstepCoreOffset(v.output[c], zb - 3, l3.y, l3.x);
    next_input[c] = MstepCoreOffset(v.input[3 + c], zb, gy, gx);
  }
  extern __shared__ float shared[];
  // Two E0 planes rotate through the input ring. Prefetch p+2 after E1's last
  // use of p, while H2/E2 consume separate shared rings. The block waits before
  // the next wave may read a neighbour's asynchronous copy.
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const auto &f = v.input[3 + c];
    MstepPackedInput<X,Y>(shared, c, zb - 2, wy, wx) =
        f.data[MstepCoreOffset(f, zb - 2, gy, gx)];
    MstepPackedInput<X,Y>(shared, c, zb - 1, wy, wx) =
        f.data[MstepCoreOffset(f, zb - 1, gy, gx)];
  }
  __syncthreads();
  const int waves = zt - zb + 3;
  for (int wave = 0; wave < waves; ++wave) {
    int p = zb + wave - 2;
    MstepPackedStage<X,Y,0>(shared, v, hc, l0, p, owned, old_h, final_h);
    if (wave >= 1)
      MstepPackedStage<X,Y,1>(shared, v, hc, l1, p, owned, first_e, final_h);
    if (wave + 1 < waves) {
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        const auto &f = v.input[3 + c];
        BEAMZ_ASYNC_COPY_F32((&MstepPackedInput<X,Y>(shared, c, p + 2, wy, wx)),
                             (f.data + next_input[c]));
      }
      BEAMZ_ASYNC_COMMIT();
    }
    if (wave >= 2)
      MstepPackedStage<X,Y,2>(shared, v, hc, l2, p - 1, owned, old_h, final_h);
    if (wave >= 3)
      MstepPackedStage<X,Y,3>(shared, v, hc, l3, p - 1, owned, final_e, final_h);
    if (wave + 1 < waves) {
      BEAMZ_ASYNC_WAIT_ALL();
      __syncthreads();
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        old_h[c] += v.input[c].plane;
        first_e[c] += v.input[3 + c].plane;
        final_e[c] += v.input[3 + c].plane;
        final_h[c] += v.output[c].plane;
        next_input[c] += v.input[3 + c].plane;
      }
    }
  }
}

template <int Phase, bool Cpml, bool Packed, bool UniformCpml = false>
__global__ void MstepEdgePhase(const __grid_constant__ BeamzLaunch launch,
                               const __grid_constant__ BeamzBuffer cells,
                               int ny, int nx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells.dims[0])
    return;
  const int row = static_cast<const int *>(cells.data)[i];
  const int z = row / (ny * nx), y = (row / nx) % ny, x = row % nx;
  UpdateComponent<Phase, 0, Cpml, 0, true, UniformCpml, -1, Packed, true>(launch, z,
                                                                    y, x);
  UpdateComponent<Phase, 1, Cpml, 0, true, UniformCpml, -1, Packed, true>(launch, z,
                                                                    y, x);
  UpdateComponent<Phase, 2, Cpml, 0, true, UniformCpml, -1, Packed, true>(launch, z,
                                                                    y, x);
}
