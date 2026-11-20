// Two-step column tiles. Eligibility and compact storage retain 8x16 quadrants.
#include "mstep_index.h"
#include "mstep_regions.h"
// CMake selects the validated defaults and leaves research-only schedules off.
// Compile-time switches retain exact A/B provenance without changing the
// schedule ABI or a graph-cache key within one loaded component. Standalone
// inclusion falls back to the original schedule unless explicitly configured.
#ifndef BEAMZ_CUDA_MSTEP_SKIP_FINAL_BARRIER
#define BEAMZ_CUDA_MSTEP_SKIP_FINAL_BARRIER 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_EDGE_SPECIALIZE
#define BEAMZ_CUDA_MSTEP_EDGE_SPECIALIZE 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_EXACT_INDEX
#define BEAMZ_CUDA_MSTEP_EXACT_INDEX 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_REGISTER_CORE
#define BEAMZ_CUDA_MSTEP_REGISTER_CORE 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_REGISTER_FOUR_CTAS
#define BEAMZ_CUDA_MSTEP_REGISTER_FOUR_CTAS 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_EARLY_PREFETCH_WAIT
#define BEAMZ_CUDA_MSTEP_EARLY_PREFETCH_WAIT 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_LAYOUT_SPECIALIZE
#define BEAMZ_CUDA_MSTEP_LAYOUT_SPECIALIZE 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_CPML_AXIS_MASK
#define BEAMZ_CUDA_MSTEP_CPML_AXIS_MASK 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_INPLACE_EDGES
#define BEAMZ_CUDA_MSTEP_INPLACE_EDGES 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES
#define BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_FULL_EDGE_TYPED
#define BEAMZ_CUDA_MSTEP_FULL_EDGE_TYPED 0
#endif
#ifndef BEAMZ_CUDA_MSTEP_FULL_EDGE_COEFFICIENTS
#define BEAMZ_CUDA_MSTEP_FULL_EDGE_COEFFICIENTS 0
#endif

template <int Phase, typename Other>
__device__ __forceinline__ float MstepCellAdvance(
    int component, float old, float decay, float source, float inv_spacing,
    Other other) {
#if BEAMZ_CUDA_MSTEP_TRACED_CELL
  const auto read = [&](int field, int version, int dz, int dy, int dx) {
    // The generated one-cell contract reads only the old same component and
    // the opposite family's staggered neighbours. The enclosing stage proves
    // the relative version: E reads updated H, H reads initial E.
    return field / 3 == Phase ? old : other(field % 3, dz, dy, dx);
  };
  const auto coefficient = [&](int kind, int c) {
    return kind == 0 ? decay : source;
  };
  if (component == 0)
    return beamz::cuda::generated::AdvanceYeeCell<Phase, 0>(inv_spacing, read, coefficient);
  if (component == 1)
    return beamz::cuda::generated::AdvanceYeeCell<Phase, 1>(inv_spacing, read, coefficient);
  return beamz::cuda::generated::AdvanceYeeCell<Phase, 2>(inv_spacing, read, coefficient);
#else
  const auto difference = [&](int c, int axis) {
    constexpr int sign = Phase == 0 ? 1 : -1;
    const float a = other(c, 0, 0, 0);
    const float b = other(c, axis == 0 ? sign : 0, axis == 1 ? sign : 0,
                          axis == 2 ? sign : 0);
    return beamz::cuda::yee::ScaleYeeDifference(
        Phase == 0 ? b - a : a - b, inv_spacing);
  };
  const auto a = beamz::cuda::yee::FirstCurlTerm(component);
  const auto b = beamz::cuda::yee::SecondCurlTerm(component);
  const float curl = difference(a.source_component, a.derivative_axis) -
                     difference(b.source_component, b.derivative_axis);
  return beamz::cuda::yee::AdvanceYeeField(Phase, old, decay, source, curl);
#endif
}

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
__global__ __launch_bounds__(MstepTile<X,Y>::threads) void MstepColumnCore(
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
#if BEAMZ_CUDA_MSTEP_INPLACE_EDGES || BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES
struct MstepPublicationView {
  float *fields[6];
  const int *slots;
  const int *masks;
  const int *bounds;
  int qy, qx, mask_x, spare;
};
MstepPublicationView MakeMstepPublicationView(
    const BeamzBuffer *fields, const BeamzBuffer &slots,
    const BeamzBuffer &masks, const BeamzBuffer &bounds) {
  MstepPublicationView result{};
  for (int c = 0; c < 6; ++c)
    result.fields[c] = static_cast<float *>(fields[c].data);
  result.slots = static_cast<const int *>(slots.data);
  result.masks = static_cast<const int *>(masks.data);
  result.bounds = static_cast<const int *>(bounds.data);
  result.qy = static_cast<int>(slots.dims[1]);
  result.qx = static_cast<int>(slots.dims[2]);
  result.mask_x = static_cast<int>(masks.dims[1]);
  result.spare = static_cast<int>(fields[0].dims[0] / 128 - 1);
  return result;
}

__device__ __forceinline__ bool MstepOwnsCoreCell(
    const MstepPublicationView &publication, int z, int y, int x) {
  if (z < publication.bounds[0] || z >= publication.bounds[1]) return false;
  const int mask = publication.masks[(y / 16) * publication.mask_x + x / 32];
  const int quadrant = ((y & 15) >= 8 ? 2 : 0) + ((x & 31) >= 16 ? 1 : 0);
  return (mask & (1 << quadrant)) != 0;
}
#endif
struct MstepPackedCoreView {
  MstepCoreField input[6], output[6];
  const uint32_t *codes[3];
  const float *tables[3], *h_decay[3], *h_source[3];
  int table_sizes[3], code_ranks[3], min_y, min_x;
  float inv;
#if BEAMZ_CUDA_MSTEP_INPLACE_EDGES
  MstepPublicationView publication;
#endif
};
__device__ __forceinline__ float MstepBlockMaterialSource(
    const MstepPackedCoreView &view, int component, int logical) {
  // Rank-two packed coefficients store one signed header per 32 cells.
  // A negative header points to one uniform value, a nonnegative header to
  // 32 exact dense values. It is not a byte-codebook ID array.
  const int header = reinterpret_cast<const int32_t *>(view.codes[component])[logical >> 5];
  const int index = header < 0 ? ~header : header + (logical & 31);
  return view.tables[component][index];
}
template <int Layout = 0>
__device__ __forceinline__ float MstepMaterialSource(
    const MstepPackedCoreView &view, int component, int logical) {
  if constexpr (Layout == 2) {
    return MstepBlockMaterialSource(view, component, logical);
  } else {
    if constexpr (Layout == 0) {
      if (view.code_ranks[component] == 2)
        return MstepBlockMaterialSource(view, component, logical);
    }
    return beamz::cuda::yee::PackedMaterialSource(
        view.tables[component], view.table_sizes[component], view.codes[component], logical);
  }
}
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

template <int X, int Y, int K, int Layout = 0, bool Publish = false>
__device__ __forceinline__ void
MstepPackedStage(float *shared, const MstepPackedCoreView &v,
                 const MstepHScalars &hc, const MstepCoreLane &lane, int z,
                 bool owned, const int (&logical)[3],
                 const int (&h_output)[3], int core_mask, int publish_zb,
                 int publish_zt) {
  using Stage = PairStage<X, Y, K>;
  constexpr int phase = K % 2, shift = phase == 1 ? 1 : 0;
#if !BEAMZ_CUDA_MSTEP_TRACED_CELL
  constexpr int sign = phase == 0 ? 1 : -1;
#endif
  if (threadIdx.x < Stage::sx * Stage::sy) {
#if BEAMZ_CUDA_MSTEP_INPLACE_EDGES
    int publication_offset = -1;
    if constexpr (Publish && K < 2) {
      const int ox = blockIdx.x * X, oy = blockIdx.y * Y;
      const int x = ox - 2 + (K + 1) / 2 + lane.ix;
      const int y = oy - 2 + (K + 1) / 2 + lane.iy;
      // Stage K0/K1 lanes differ from final K3 lanes. Only this CTA's useful
      // XY rectangle and Z segment may publish, never a discarded halo.
      if (x >= ox && x < ox + X && y >= oy && y < oy + Y &&
          z >= publish_zb && z < publish_zt) {
        const int quadrant = ((y - oy) >= 8 ? 2 : 0) + ((x - ox) >= 16 ? 1 : 0);
        if (core_mask & (1 << quadrant)) {
          const auto &p = v.publication;
          const int slot = p.slots[(z * p.qy + y / 8) * p.qx + x / 16];
          // All missing quadrants alias one spare slot; publishing those
          // would create unrelated writers and is neither needed nor safe.
          if (slot >= 0 && slot < p.spare)
            publication_offset = slot * 128 + (y & 7) * 16 + (x & 15);
        }
      }
    }
#endif
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
#if !BEAMZ_CUDA_MSTEP_TRACED_CELL
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
#endif
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
        source = MstepMaterialSource<Layout>(v, c, logical[c]);
      }
#if BEAMZ_CUDA_MSTEP_TRACED_CELL
      float next = MstepCellAdvance<phase>(c, old, decay, source, v.inv, other);
#else
      float next = beamz::cuda::yee::AdvanceYeeField(phase, old, decay, source, curl);
#endif
#if BEAMZ_CUDA_MSTEP_INPLACE_EDGES
      if constexpr (Publish && K < 2) {
        if (publication_offset >= 0)
          v.publication.fields[3 * phase + c][publication_offset] = next;
      }
#endif
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
  // E2 publishes only global final fields and reads the H2 ring. The caller
  // waits for the asynchronous E0 input plane, then executes a block barrier
  // before another wave can overwrite shared state. Thus that caller barrier
  // also covers E2's final H2 reads. On the terminal wave no shared successor
  // exists, so no final-stage barrier is needed at all.
  // K2's publication barrier can also certify the next E0 plane, avoiding a
  // second loop-end barrier. The prologue waves that do not execute K2 retain
  // the old completion barrier in the caller.
  if constexpr (BEAMZ_CUDA_MSTEP_EARLY_PREFETCH_WAIT && K == 2)
    __pipeline_wait_prior(0);
  if constexpr ((!BEAMZ_CUDA_MSTEP_SKIP_FINAL_BARRIER &&
                 !BEAMZ_CUDA_MSTEP_EARLY_PREFETCH_WAIT) || K != 3)
    __syncthreads();
}

template <int X, int Y, int Layout = 0, bool Publish = false>
__global__ __launch_bounds__(MstepTile<X,Y>::threads, 1) void MstepPackedColumnCore(
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
    MstepPackedStage<X,Y,0,Layout,Publish>(shared, v, hc, l0, p, owned, old_h, final_h, mask, zb, zt);
    if (wave >= 1)
      MstepPackedStage<X,Y,1,Layout,Publish>(shared, v, hc, l1, p, owned, first_e, final_h, mask, zb, zt);
    if (wave + 1 < waves) {
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        const auto &f = v.input[3 + c];
        __pipeline_memcpy_async(&MstepPackedInput<X,Y>(shared, c, p + 2, wy, wx),
                                f.data + next_input[c], sizeof(float));
      }
      __pipeline_commit();
    }
    if (wave >= 2)
      MstepPackedStage<X,Y,2,Layout,Publish>(shared, v, hc, l2, p - 1, owned, old_h, final_h, mask, zb, zt);
    if (wave >= 3)
      MstepPackedStage<X,Y,3,Layout,Publish>(shared, v, hc, l3, p - 1, owned, final_e, final_h, mask, zb, zt);
    if (wave + 1 < waves) {
#if BEAMZ_CUDA_MSTEP_EARLY_PREFETCH_WAIT
      if (wave < 2) {
        __pipeline_wait_prior(0);
        __syncthreads();
      }
#else
      __pipeline_wait_prior(0);
      __syncthreads();
#endif
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

#if BEAMZ_CUDA_MSTEP_REGISTER_CORE
#include "mstep_register.cuh"
#endif

template <int Phase, bool Packed, bool HasMetallicEdges, int PsiType,
          int ActiveAxes>
__device__ __forceinline__ void MstepUniformEdgeComponents(
    const BeamzLaunch &launch, int z, int y, int x) {
  UpdateComponent<Phase, 0, true, 0, HasMetallicEdges, true, PsiType,
                  Packed, true, false, ActiveAxes>(launch, z, y, x);
  UpdateComponent<Phase, 1, true, 0, HasMetallicEdges, true, PsiType,
                  Packed, true, false, ActiveAxes>(launch, z, y, x);
  UpdateComponent<Phase, 2, true, 0, HasMetallicEdges, true, PsiType,
                  Packed, true, false, ActiveAxes>(launch, z, y, x);
}

template <int Phase, bool Cpml, bool Packed, bool UniformCpml = false,
          bool HasMetallicEdges = true, int PsiType = -1>
__global__ void MstepEdgePhase(const __grid_constant__ BeamzLaunch launch,
                               const __grid_constant__ BeamzBuffer cells,
                               int ny, int nx, const MstepIndexDecode decode) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells.dims[0])
    return;
  const int row = static_cast<const int *>(cells.data)[i];
#if BEAMZ_CUDA_MSTEP_EXACT_INDEX
  const int z = MstepQuotient(static_cast<uint32_t>(row), decode.plane);
  const int plane_row = row - z * ny * nx;
  const int y = MstepQuotient(static_cast<uint32_t>(plane_row), decode.x);
  const int x = plane_row - y * nx;
#else
  const int z = row / (ny * nx), y = (row / nx) % ny, x = row % nx;
#endif
#if BEAMZ_CUDA_MSTEP_CPML_AXIS_MASK
  if constexpr (Cpml && UniformCpml) {
    const int coordinate[] = {z, y, x};
    int axes = 0;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
      if (MstepCpmlAxisMayBeActive(coordinate[axis], launch.uniform_cpml_thickness,
                                   launch.outputs[0].dims[axis],
                                   launch.outputs[1].dims[axis],
                                   launch.outputs[2].dims[axis]))
        axes |= 1 << axis;
    }
    // Only proven-inactive axes bypass the recurrence. Included axes retain
    // CorrectCpml's own staggered-extent, precision and packed-index checks.
#define BEAMZ_MSTEP_AXIS_CASE(mask) \
    case mask: MstepUniformEdgeComponents<Phase, Packed, HasMetallicEdges, PsiType, mask>(launch, z, y, x); break
    switch (axes) {
      BEAMZ_MSTEP_AXIS_CASE(0);
      BEAMZ_MSTEP_AXIS_CASE(1);
      BEAMZ_MSTEP_AXIS_CASE(2);
      BEAMZ_MSTEP_AXIS_CASE(3);
      BEAMZ_MSTEP_AXIS_CASE(4);
      BEAMZ_MSTEP_AXIS_CASE(5);
      BEAMZ_MSTEP_AXIS_CASE(6);
      BEAMZ_MSTEP_AXIS_CASE(7);
    }
#undef BEAMZ_MSTEP_AXIS_CASE
    return;
  }
#endif
#if BEAMZ_CUDA_MSTEP_EDGE_SPECIALIZE && !BEAMZ_CUDA_MSTEP_CPML_AXIS_MASK
  if constexpr (Cpml && UniformCpml) {
    // The first-step compact queue includes monitor samples and dependencies
    // around core ownership boundaries, not only physical absorbing cells.
    // Prove all three components are outside every absorbing slab before
    // bypassing the six recurrence eligibility tests and wall predicates.
    const int low = launch.uniform_cpml_thickness;
    const int high_z = min(min(launch.outputs[0].dims[0], launch.outputs[1].dims[0]),
                           launch.outputs[2].dims[0]) - low;
    const int high_y = min(min(launch.outputs[0].dims[1], launch.outputs[1].dims[1]),
                           launch.outputs[2].dims[1]) - low;
    const int high_x = min(min(launch.outputs[0].dims[2], launch.outputs[1].dims[2]),
                           launch.outputs[2].dims[2]) - low;
    if (z >= low && y >= low && x >= low &&
        z < high_z && y < high_y && x < high_x) {
      UpdateComponent<Phase, 0, false, 0, false, false, -1, Packed, true>(launch, z, y, x);
      UpdateComponent<Phase, 1, false, 0, false, false, -1, Packed, true>(launch, z, y, x);
      UpdateComponent<Phase, 2, false, 0, false, false, -1, Packed, true>(launch, z, y, x);
      return;
    }
  }
#endif
  UpdateComponent<Phase, 0, Cpml, 0, HasMetallicEdges, UniformCpml, PsiType, Packed, true>(launch, z,
                                                                    y, x);
  UpdateComponent<Phase, 1, Cpml, 0, HasMetallicEdges, UniformCpml, PsiType, Packed, true>(launch, z,
                                                                    y, x);
  UpdateComponent<Phase, 2, Cpml, 0, HasMetallicEdges, UniformCpml, PsiType, Packed, true>(launch, z,
                                                                    y, x);
}

#if BEAMZ_CUDA_MSTEP_INPLACE_EDGES || BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES
template <int Phase, bool Cpml, bool Packed, bool UniformCpml, bool CopyCore,
          int PsiType = -1, bool HasMetallicEdges = true, int CoefficientLayout = 0>
__global__ void MstepFullEdgePhase(
    const __grid_constant__ BeamzLaunch launch,
    const __grid_constant__ BeamzBuffer cells, int ny, int nx,
    const MstepIndexDecode decode, const MstepPublicationView publication) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells.dims[0]) return;
  const int row = static_cast<const int *>(cells.data)[i];
#if BEAMZ_CUDA_MSTEP_EXACT_INDEX
  const int z = MstepQuotient(static_cast<uint32_t>(row), decode.plane);
  const int plane_row = row - z * ny * nx;
  const int y = MstepQuotient(static_cast<uint32_t>(plane_row), decode.x);
  const int x = plane_row - y * nx;
#else
  const int z = row / (ny * nx), y = (row / nx) % ny, x = row % nx;
#endif
  if constexpr (CopyCore) {
    if (MstepOwnsCoreCell(publication, z, y, x)) {
      const int slot = publication.slots[(z * publication.qy + y / 8) *
                                         publication.qx + x / 16];
      const int offset = slot * 128 + (y & 7) * 16 + (x & 15);
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        const auto &out = launch.outputs[c];
        static_cast<float *>(out.data)[MstepFieldOffset(out, z, y, x)] =
            publication.fields[Phase * 3 + c][offset];
      }
      return;
    }
  }
  // The core kernel has already finished all reads of the frozen input bank.
  // During each edge phase only this family is written; every neighbour read
  // is from the opposite family and therefore cannot race another row here.
  UpdateComponent<Phase, 0, Cpml, 0, HasMetallicEdges, UniformCpml, PsiType, Packed, false, false, -1, CoefficientLayout>(launch, z, y, x);
  UpdateComponent<Phase, 1, Cpml, 0, HasMetallicEdges, UniformCpml, PsiType, Packed, false, false, -1, CoefficientLayout>(launch, z, y, x);
  UpdateComponent<Phase, 2, Cpml, 0, HasMetallicEdges, UniformCpml, PsiType, Packed, false, false, -1, CoefficientLayout>(launch, z, y, x);
}

#if BEAMZ_CUDA_MSTEP_FULL_EDGE_TYPED
template <bool CopyCore, int PsiType, bool HasMetallicEdges>
void LaunchMstepTypedFullEdge(
    cudaStream_t stream, const BeamzLaunch &launch, const BeamzBuffer &cells,
    int ny, int nx, int blocks, const MstepIndexDecode &decode,
    const MstepPublicationView &publication) {
#if BEAMZ_CUDA_MSTEP_FULL_EDGE_COEFFICIENTS
  if (launch.phase == 0) {
    bool scalar = true;
    for (int material = 6; material < 12; ++material)
      scalar &= launch.inputs[material].rank == 0 &&
                launch.inputs[material].element_type == kBeamzF32;
    if (scalar) {
      MstepFullEdgePhase<0, true, false, true, CopyCore, PsiType, HasMetallicEdges, -1>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
      return;
    }
  } else if (launch.inputs[9].element_type == kBeamzS32) {
    const int layout = launch.inputs[9].rank;
    bool homogeneous = layout == 1 || layout == 2;
    for (int c = 0; c < 3; ++c)
      homogeneous &= launch.inputs[9 + c].rank == layout &&
          launch.inputs[9 + c].element_type == kBeamzS32 &&
          launch.inputs[6 + c].rank == 1 &&
          launch.inputs[6 + c].element_type == kBeamzF32 &&
          launch.inputs[6 + c].dims[0] > 0 &&
          launch.inputs[6 + c].dims[0] <= INT32_MAX;
    if (homogeneous && layout == 1) {
      MstepFullEdgePhase<1, true, true, true, CopyCore, PsiType, HasMetallicEdges, 1>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
      return;
    }
    if (homogeneous && layout == 2) {
      MstepFullEdgePhase<1, true, true, true, CopyCore, PsiType, HasMetallicEdges, 2>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
      return;
    }
  }
#endif
  if (launch.phase == 0)
    MstepFullEdgePhase<0, true, false, true, CopyCore, PsiType, HasMetallicEdges>
        <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
  else if (launch.inputs[9].element_type == kBeamzS32)
    MstepFullEdgePhase<1, true, true, true, CopyCore, PsiType, HasMetallicEdges>
        <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
  else
    MstepFullEdgePhase<1, true, false, true, CopyCore, PsiType, HasMetallicEdges>
        <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
}

template <bool CopyCore, bool HasMetallicEdges>
void LaunchMstepUniformFullEdge(
    cudaStream_t stream, const BeamzLaunch &launch, const BeamzBuffer &cells,
    int ny, int nx, int blocks, const MstepIndexDecode &decode,
    const MstepPublicationView &publication) {
  // Prove homogeneous auxiliary precision from every input AND output term.
  // Mixed foreign payloads retain the original dynamic decoder and PEC path.
  int psi_type = launch.outputs[3].element_type;
  for (int term = 0; term < 6; ++term) {
    if (launch.inputs[31 + term].element_type != psi_type ||
        launch.outputs[3 + term].element_type != psi_type) {
      psi_type = -1;
      break;
    }
  }
  if (psi_type == kBeamzF32)
    LaunchMstepTypedFullEdge<CopyCore, kBeamzF32, HasMetallicEdges>(
        stream, launch, cells, ny, nx, blocks, decode, publication);
  else if (psi_type == kBeamzBF16)
    LaunchMstepTypedFullEdge<CopyCore, kBeamzBF16, HasMetallicEdges>(
        stream, launch, cells, ny, nx, blocks, decode, publication);
  else
    LaunchMstepTypedFullEdge<CopyCore, -1, true>(
        stream, launch, cells, ny, nx, blocks, decode, publication);
}
#endif

template <bool CopyCore>
cudaError_t LaunchMstepFullEdge(
    cudaStream_t stream, const BeamzLaunch &launch, const BeamzBuffer &cells,
    int ny, int nx, const MstepPublicationView &publication) {
  if (!cells.dims[0]) return cudaSuccess;
  const int blocks = (cells.dims[0] + 255) / 256;
  const auto decode = MakeMstepIndexDecode(ny, nx);
  if (launch.uniform_cpml_thickness > 0) {
#if BEAMZ_CUDA_MSTEP_FULL_EDGE_TYPED
    if (launch.metallic_edges)
      LaunchMstepUniformFullEdge<CopyCore, true>(
          stream, launch, cells, ny, nx, blocks, decode, publication);
    else
      LaunchMstepUniformFullEdge<CopyCore, false>(
          stream, launch, cells, ny, nx, blocks, decode, publication);
#else
    if (launch.phase == 0)
      MstepFullEdgePhase<0, true, false, true, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
    else if (launch.inputs[9].element_type == kBeamzS32)
      MstepFullEdgePhase<1, true, true, true, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
    else
      MstepFullEdgePhase<1, true, false, true, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
#endif
  } else if (launch.phase == 0) {
    if (launch.nterms)
      MstepFullEdgePhase<0, true, false, false, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
    else
      MstepFullEdgePhase<0, false, false, false, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
  } else if (launch.inputs[9].element_type == kBeamzS32) {
    if (launch.nterms)
      MstepFullEdgePhase<1, true, true, false, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
    else
      MstepFullEdgePhase<1, false, true, false, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
  } else {
    if (launch.nterms)
      MstepFullEdgePhase<1, true, false, false, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
    else
      MstepFullEdgePhase<1, false, false, false, CopyCore>
          <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, decode, publication);
  }
  return cudaGetLastError();
}
#endif

#if BEAMZ_CUDA_MSTEP_EDGE_SPECIALIZE
template <int Phase, bool Packed, bool HasMetallicEdges>
void LaunchMstepUniformEdge(cudaStream_t stream, const BeamzLaunch &launch,
                           const BeamzBuffer &cells, int ny, int nx, int blocks) {
  // Foreign FFI callers can provide mixed auxiliary precision. Retain the
  // dynamic fallback unless every input/output term proves the same type.
  int psi_type = launch.outputs[3].element_type;
  for (int term = 0; term < 6; ++term) {
    if (launch.inputs[31 + term].element_type != psi_type ||
        launch.outputs[3 + term].element_type != psi_type) {
      psi_type = -1;
      break;
    }
  }
  if (psi_type == kBeamzF32)
    MstepEdgePhase<Phase, true, Packed, true, HasMetallicEdges, kBeamzF32>
        <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, MakeMstepIndexDecode(ny, nx));
  else if (psi_type == kBeamzBF16)
    MstepEdgePhase<Phase, true, Packed, true, HasMetallicEdges, kBeamzBF16>
        <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, MakeMstepIndexDecode(ny, nx));
  else
    MstepEdgePhase<Phase, true, Packed, true, HasMetallicEdges>
        <<<blocks, 256, 0, stream>>>(launch, cells, ny, nx, MakeMstepIndexDecode(ny, nx));
}

template <int Phase, bool Packed>
void LaunchMstepUniformEdge(cudaStream_t stream, const BeamzLaunch &launch,
                           const BeamzBuffer &cells, int ny, int nx, int blocks) {
  if (launch.metallic_edges)
    LaunchMstepUniformEdge<Phase, Packed, true>(stream, launch, cells, ny, nx, blocks);
  else
    LaunchMstepUniformEdge<Phase, Packed, false>(stream, launch, cells, ny, nx, blocks);
}
#endif
