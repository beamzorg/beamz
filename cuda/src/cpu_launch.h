#ifndef BEAMZ_CPU_LAUNCH_H_
#define BEAMZ_CPU_LAUNCH_H_

// CPU platform prelude: CUDA-style kernels compiled by the host compiler and
// run by an OpenMP block scheduler. Included by gpu_device.h only when
// BEAMZ_PLATFORM_CPU is defined.
//
// Execution model: blocks are distributed over OpenMP threads; inside a block
// the CUDA threads run one after another (x fastest, matching memory order).
// A warp is one lane wide. Kernels that need barriers or shared memory cannot
// be expressed this way; they all request dynamic shared memory, and the
// launcher rejects any launch that does with cudaErrorNotSupported.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__SSE__)
#include <xmmintrin.h>
#else
// XLA's CPU threads flush denormals; on x86 the launcher copies those modes
// (MXCSR) into its workers. Other ISAs need an FPCR equivalent, without which
// tiny DFT values can round differently depending on the worker.
#warning "beamz._cpu propagates flush-to-zero modes only on x86 (SSE)"
#endif

#include "gpu_runtime.h"

#if defined(BEAMZ_CPU_PROFILE)
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#endif

#define __global__ inline __attribute__((always_inline))
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...)
#define __grid_constant__
#define __restrict__ __restrict
// Only reached in kernels the launcher rejects (they all also use dynamic
// shared memory); one instance per OpenMP thread keeps them well formed.
#define __shared__ static thread_local
#define BEAMZ_LAUNCH_BOUNDS(threads, min_blocks)
#define BEAMZ_DYNAMIC_SHARED(type, name) type* name = nullptr
// Warps have one lane: width-dependent code takes its per-lane path.
#define BEAMZ_WAVE32 false
#define BEAMZ_ARG(type) const type&
#define BEAMZ_GRID_ARG(type) const type&
#define BEAMZ_BROADCAST_LEADER true

struct uint3 {
  unsigned int x, y, z;
};
struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
      : x(x_), y(y_), z(z_) {}
};

namespace beamz::cpu {
inline thread_local uint3 thread_index{0, 0, 0};
inline thread_local uint3 block_index{0, 0, 0};
inline thread_local dim3 block_dim{};
inline thread_local dim3 grid_dim{};
}  // namespace beamz::cpu

#define threadIdx (::beamz::cpu::thread_index)
#define blockIdx (::beamz::cpu::block_index)
#define blockDim (::beamz::cpu::block_dim)
#define gridDim (::beamz::cpu::grid_dim)

// Explicitly rounded arithmetic. Builds use -ffp-contract=off, so a plain
// multiply or add is never fused with a neighbouring operation.
inline float __fmaf_rn(float a, float b, float c) { return std::fma(a, b, c); }
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fadd_rn(float a, float b) { return a + b; }
using std::isfinite;
using std::isinf;
using std::isnan;

// CUDA's device min/max: integer overloads and fminf/fmaxf for floats.
template <typename A, typename B>
inline auto min(A a, B b) -> decltype(a < b ? a : b) {
  return b < a ? b : a;
}
template <typename A, typename B>
inline auto max(A a, B b) -> decltype(a < b ? a : b) {
  return a < b ? b : a;
}
inline float min(float a, float b) { return std::fmin(a, b); }
inline float max(float a, float b) { return std::fmax(a, b); }

// Every kernel with a barrier also requests dynamic shared memory, which the
// launcher rejects, so no barrier is reached on the CPU. A kernel that slips
// past (static __shared__ only, say) would race silently: stop instead.
[[noreturn]] inline void __syncthreads() {
  std::fputs("beamz._cpu: a kernel reached __syncthreads; barrier kernels "
             "cannot run on the CPU platform\n", stderr);
  std::abort();
}

// Only reached in shared-memory kernels, which the launcher rejects.
#define BEAMZ_ASYNC_COPY_F32(shared_destination, global_source) \
  (*(shared_destination) = *(global_source))
#define BEAMZ_ASYNC_COMMIT() ((void)0)
#define BEAMZ_ASYNC_WAIT_ALL() ((void)0)

// One-lane warps: every lane leads, shuffles return the lane's own value.
typedef unsigned beamz_lane_mask;
#define BEAMZ_FULL_LANE_MASK 1u
inline beamz_lane_mask __activemask() { return 1u; }
template <typename T>
inline T __shfl_sync(beamz_lane_mask, T value, int, int = 1) {
  return value;
}
template <typename T>
inline T __shfl_down_sync(beamz_lane_mask, T value, unsigned int, int = 1) {
  return value;
}
inline int beamz_first_lane(beamz_lane_mask mask) {
  return __builtin_ffs(static_cast<int>(mask));
}

// Blocks run concurrently on OpenMP threads.
inline float atomicAdd(float* address, float value) {
  float old;
#pragma omp atomic capture
  {
    old = *address;
    *address += value;
  }
  return old;
}
inline int atomicAdd(int* address, int value) {
  int old;
#pragma omp atomic capture
  {
    old = *address;
    *address += value;
  }
  return old;
}

// bfloat16 storage with CUDA's conversions: round to nearest even, NaN kept
// quiet.
struct beamz_bf16 {
  uint16_t bits;
};
inline float __bfloat162float(beamz_bf16 value) {
  const uint32_t word = static_cast<uint32_t>(value.bits) << 16;
  float result;
  std::memcpy(&result, &word, sizeof(result));
  return result;
}
inline beamz_bf16 beamz_float_to_bf16_rn(float value) {
  uint32_t word;
  std::memcpy(&word, &value, sizeof(word));
  if ((word & 0x7fffffffu) > 0x7f800000u)
    return {static_cast<uint16_t>((word >> 16) | 0x40u)};
  word += 0x7fffu + ((word >> 16) & 1u);
  return {static_cast<uint16_t>(word >> 16)};
}

namespace beamz::cpu {

#if defined(BEAMZ_CPU_PROFILE)
// Opt-in (-DBEAMZ_CPU_PROFILE): wall time per kernel, printed at exit.
struct KernelProfile {
  struct Entry {
    double seconds = 0;
    long launches = 0;
    double blocks = 0;
  };
  std::mutex mutex;
  std::map<std::string, Entry> entries;
  ~KernelProfile() {
    double total = 0;
    for (auto& [name, e] : entries) total += e.seconds;
    std::fprintf(stderr, "beamz cpu profile: %.3f s in kernels\n", total);
    for (auto& [name, e] : entries)
      std::fprintf(stderr, "%8.3f s %5.1f%% %8ld launches %10.0f blocks/launch  %s\n",
                   e.seconds, 100 * e.seconds / total, e.launches,
                   e.blocks / e.launches, name.c_str());
  }
  static KernelProfile& Get() {
    static KernelProfile profile;
    return profile;
  }
};
template <auto Kernel>
const char* KernelName() {
  return __PRETTY_FUNCTION__;
}
#endif

// CUDA threads below which a launch runs on the calling thread.
#if !defined(BEAMZ_CPU_PARALLEL_THRESHOLD)
#define BEAMZ_CPU_PARALLEL_THRESHOLD 32768
#endif
inline constexpr int64_t kParallelThreshold = BEAMZ_CPU_PARALLEL_THRESHOLD;

template <auto Kernel>
struct Launcher {
  dim3 grid;
  dim3 block;
  size_t shared_bytes;

  template <typename... Args>
  void operator()(const Args&... args) const {
    Run(std::index_sequence_for<Args...>{}, args...);
  }

  template <size_t... I, typename... Args>
  void Run(std::index_sequence<I...>, const Args&... args) const {
    if (shared_bytes != 0) {
      beamz_cpu_last_error = cudaErrorNotSupported;
      return;
    }
    const int64_t gx = grid.x, gy = grid.y;
    const int64_t blocks = gx * gy * static_cast<int64_t>(grid.z);
    if (blocks == 0) return;
    const dim3 g = grid, b = block;
#if defined(BEAMZ_CPU_PROFILE)
    const auto start = std::chrono::steady_clock::now();
    struct Record {
      std::chrono::steady_clock::time_point start;
      int64_t blocks;
      ~Record() {
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start).count();
        auto& p = KernelProfile::Get();
        std::lock_guard<std::mutex> lock(p.mutex);
        auto& e = p.entries[KernelName<Kernel>()];
        e.seconds += s;
        e.launches += 1;
        e.blocks += blocks;
      }
    } record{start, blocks};
#endif
#if defined(__SSE__)
    // Workers inherit the caller's flush-to-zero/denormals-are-zero modes.
    // XLA's CPU threads set both; fresh OpenMP threads do not, which made
    // results depend on which thread ran a block.
    const unsigned int csr = _mm_getcsr() & ~0x3fu;  // modes only, no flags
#endif
#if defined(_OPENMP)
    const int64_t chunk = blocks / (16 * static_cast<int64_t>(omp_get_max_threads())) + 1;
    // Waking the team costs far more than a few thousand CUDA threads of
    // work (source injection, DFT phases): run those on the caller.
    const int64_t work = blocks * static_cast<int64_t>(b.x) * b.y * b.z;
#pragma omp parallel if (work >= kParallelThreshold)
#endif
    {
#if defined(__SSE__)
    _mm_setcsr(csr);
#endif
    // A private copy per worker: kernels take their (multi-KB) descriptors by
    // value, and stores through the descriptors' pointers could alias a
    // shared original, forcing one full copy per CUDA thread. Nothing can
    // alias this local, so the per-thread copies fold away.
    const std::tuple<Args...> local(args...);
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, chunk)
#endif
    for (int64_t linear = 0; linear < blocks; ++linear) {
      grid_dim = g;
      block_dim = b;
      block_index = {static_cast<unsigned int>(linear % gx),
                     static_cast<unsigned int>((linear / gx) % gy),
                     static_cast<unsigned int>(linear / (gx * gy))};
      for (unsigned int tz = 0; tz < b.z; ++tz)
        for (unsigned int ty = 0; ty < b.y; ++ty)
          for (unsigned int tx = 0; tx < b.x; ++tx) {
            thread_index = {tx, ty, tz};
            Kernel(std::get<I>(local)...);
          }
    }
    }
  }
};

template <auto Kernel>
inline Launcher<Kernel> Launch(dim3 grid, dim3 block, size_t shared_bytes) {
  return {grid, block, shared_bytes};
}

}  // namespace beamz::cpu

#define BEAMZ_LAUNCH(grid, block, shared, stream, ...) \
  ((void)(stream), ::beamz::cpu::Launch<__VA_ARGS__>((grid), (block), (shared)))

#endif  // BEAMZ_CPU_LAUNCH_H_
