#ifndef BEAMZ_CUDA_LAUNCH_H_
#define BEAMZ_CUDA_LAUNCH_H_

#include <cstdint>

enum BeamzElementType : int32_t {
  kBeamzF32 = 0,
  kBeamzS32 = 1,
  kBeamzBF16 = 2,
};

struct BeamzBuffer {
  void* data;
  int32_t rank;
  int32_t element_type;
  int64_t dims[4];
  // Logical dimensions remain unchanged when storage is padded. Rank-three
  // descriptors always have explicit pitches, including contiguous arrays.
  int32_t row_stride = 0;
  int32_t plane_stride = 0;
  // cuda_mstep intermediate fields share a quadrant slot map. All other
  // buffers retain ordinary pitches. Logical shapes never change.
  const int32_t *compact_slots = nullptr;
  int32_t compact_ny = 0;
  int32_t compact_nx = 0;
};

#ifdef __CUDACC__
#define BEAMZ_HD __host__ __device__ __forceinline__
#else
#define BEAMZ_HD inline
#endif
template <bool Compact = true>
BEAMZ_HD int BeamzOffset3D(const BeamzBuffer& value, int z, int y, int x) {
  if constexpr (Compact) {
    if (value.compact_slots) {
      const int q = (z * value.compact_ny + y / 8) * value.compact_nx + x / 16;
      return value.compact_slots[q] * 128 + (y % 8) * 16 + x % 16;
    }
  }
  return z * value.plane_stride + y * value.row_stride + x;
}
BEAMZ_HD int BeamzPhysicalIndex(const BeamzBuffer& value, int logical) {
  const int nx = static_cast<int>(value.dims[2]);
  const int ny = static_cast<int>(value.dims[1]);
  if (!value.compact_slots && value.row_stride == nx &&
      value.plane_stride == nx * ny)
    return logical;
  const int zy = logical / nx;
  return BeamzOffset3D(value, zy / ny, zy % ny, logical % nx);
}
#undef BEAMZ_HD

// Optional streamed electric correction. Scratch is an XLA-owned output, never
// hidden device allocation. Origin and coefficients are dynamic device inputs.
struct BeamzTensorLaunch {
  BeamzBuffer cross;
  BeamzBuffer origin;
  BeamzBuffer widths[3];  // z, y, x primal widths (scalar on uniform axes)
  BeamzBuffer curls[3];   // Ex, Ey, Ez supports within the halo-inclusive box
};

struct BeamzLaunch {
  int32_t abi_version;
  int32_t cuda_flags;
  int32_t phase;
  int32_t nterms;
  int32_t metric_kind;
  float dt;
  float resolution;
  float inv_resolution;
  float dt_over_eps;
  float dt_over_mu;
  int32_t metallic_edges;
  int32_t magnetic_edges;
  int32_t uniform_cpml_thickness;
  BeamzBuffer inputs[37];
  BeamzBuffer metrics[3];
  BeamzBuffer outputs[9];
  // Sharded phases only: [axis, origin, return_curl], three logical target
  // shapes, then three logical source shapes. Resident on the execution device.
  BeamzBuffer shard_geometry;
  // Separate lower/upper one-cell faces, interleaved by source component.
  // Owned source fields remain contiguous; no field-sized halo concatenation.
  BeamzBuffer shard_halos[6];
  // Host dispatch only; tensor kernels receive the pointed-to record by value.
  const BeamzTensorLaunch* tensor = nullptr;
};

// The phase outputs hold incoming cotangents. Metadata is nondifferentiable;
// all other inputs have independently owned, float32 gradient buffers.
struct BeamzAdjointLaunch {
  BeamzLaunch phase;
  uint64_t gradient_mask;
  BeamzBuffer gradients[37];
  BeamzBuffer metric_gradients[3];
  bool accumulate_sources = false;
  bool accumulate_material = false;
  BeamzBuffer source_cotangents[3]{};
  BeamzBuffer source_decays[3]{};
  BeamzBuffer reverse_iteration{};
  int32_t psi_bank_stride[6]{};
};

struct BeamzSourceGroupLaunch {
  BeamzBuffer coefficients;
  BeamzBuffer waveforms;
  BeamzBuffer starts;
  BeamzBuffer current_step;
  int32_t component;
  int32_t timing;
  int32_t coincident;
  // Proven by the Python compiler from static source origins and extents. The
  // source kernel can replace atomics with ordinary adds only for this case.
  int32_t disjoint;
};

struct BeamzDftGroupLaunch {
  BeamzBuffer indices;
  BeamzBuffer weights;
  BeamzBuffer frequencies;
  BeamzBuffer component_masks;
  BeamzBuffer counts;
  BeamzBuffer codes;
  BeamzBuffer windows;
  BeamzBuffer dft_re;
  BeamzBuffer dft_im;
  BeamzBuffer dft_weight;
  // XLA-owned scratch used by the monitor-heavy phase-cache specialization.
  BeamzBuffer phase_sin;
  BeamzBuffer phase_cos;
  BeamzBuffer phase_window;
  // Paired monitor samples: [monitor, component, point, substep].
  BeamzBuffer pair_samples{};
  // Immutable invocation origin and integer offset of this native chunk.
  BeamzBuffer time;
  BeamzBuffer current_step;
  BeamzBuffer elapsed_steps;
  int32_t monitor_count;
};

// A non-owning, complete native timestep program assembled by the FFI decoder.
// One field bank describes in-place graph execution; two banks describe the
// temporal ping-pong schedule. Source and monitor pointers remain valid for the
// duration of graph lookup/capture because their storage belongs to the handler.
struct BeamzProgramLaunch {
  BeamzLaunch h_ab;
  BeamzLaunch e_ab;
  BeamzLaunch h_ba;
  BeamzLaunch e_ba;
  const BeamzSourceGroupLaunch* source_groups;
  int32_t source_group_count;
  const BeamzDftGroupLaunch* monitors;
  int32_t field_bank_count;
  int32_t nsteps;
  int32_t graph_cache_capacity = 32;
  // One immutable compiler decision, cross-validated once before capture. It
  // prevents individual launchers from re-inferring incompatible fast paths.
  int32_t schedule_flags = 0;
  // Experimental two-timestep core uses a third bank: old fields remain frozen
  // while one kernel publishes intermediate monitor state and final fields.
  BeamzBuffer pair_fields[6]{};
  // Optional read-only 16x8x1 DFT publication map, in logical coordinates.
  BeamzBuffer pair_publication{};
};

// Returns zero after enqueueing all work, otherwise a CUDA runtime error code.
int BeamzLaunchStreamed(void* stream, const BeamzLaunch& launch);
int BeamzLaunchSharded(void* stream, const BeamzLaunch& launch);
int BeamzLaunchProgram(void* stream, const BeamzProgramLaunch& program);
int BeamzLaunchAdjoint(void* stream, const BeamzAdjointLaunch& launch);

struct BeamzMstepLaunch {
  BeamzProgramLaunch program;
  BeamzBuffer final_fields[6];
  BeamzBuffer scratch_fields[6];
  BeamzBuffer bounds;
  BeamzBuffer core_masks;
  BeamzBuffer slots;
  BeamzBuffer first_cells;
  BeamzBuffer edge_cells;
};
int BeamzLaunchMstep(void *stream, const BeamzMstepLaunch &launch);

#endif  // BEAMZ_CUDA_LAUNCH_H_
