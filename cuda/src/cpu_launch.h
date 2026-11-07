#ifndef BEAMZ_CPU_LAUNCH_H_
#define BEAMZ_CPU_LAUNCH_H_

// CPU platform prelude: CUDA-style kernels compiled by the host compiler and
// run by an OpenMP block scheduler. Included by gpu_device.h only when
// BEAMZ_PLATFORM_CPU is defined.
//
// Execution model: blocks are distributed over OpenMP threads; inside a block
// the CUDA threads run one after another (x fastest, matching memory order).
// A warp is one lane wide. A barrier cannot be honoured between threads that
// run one after another, so kernels with barriers run only as one-thread
// blocks (BEAMZ_BARRIER_BLOCK): their block-stride loops then cover every
// stage before the barrier, which has nobody to wait for. Dynamic shared
// memory is a per-worker scratch buffer that stays in cache. The launcher
// rejects shared memory in blocks of more than one thread.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <type_traits>
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
// One instance per OpenMP worker, which runs one (one-thread) block at a time.
#define __shared__ static thread_local
#define BEAMZ_LAUNCH_BOUNDS(threads, min_blocks)
#define BEAMZ_DYNAMIC_SHARED(type, name) \
  type* name = static_cast<type*>(::beamz::cpu::shared_scratch)
#define BEAMZ_BARRIER_BLOCK(...) 1
#if defined(__clang__)
#define BEAMZ_UNROLL _Pragma("unroll")
#else
#define BEAMZ_UNROLL _Pragma("GCC unroll 16")
#endif
// Cells of one tile stage are independent: each writes only its own slot and
// reads the previous stage or frozen inputs. Saying so lets the compiler
// vectorise along x despite clamped addresses and possibly aliasing pointers.
#if defined(__clang__)
#define BEAMZ_INDEPENDENT_X _Pragma("clang loop vectorize(assume_safety)")
#else
#define BEAMZ_INDEPENDENT_X _Pragma("GCC ivdep")
#endif
#define BEAMZ_TILE_LOOP(iy, ix, ny, nx) \
  for (int iy = 0; iy < (ny); ++iy)     \
    BEAMZ_INDEPENDENT_X for (int ix = 0; ix < (nx); ++ix)
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
// The running block's dynamic shared memory (null when none was requested).
inline thread_local void* shared_scratch = nullptr;
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

// CUDA's device min/max: integer overloads and fminf/fmaxf for floats. The
// result is a value of the common type: decltype of a conditional over the
// named parameters would be a reference to a parameter that dies on return.
template <typename A, typename B>
inline std::common_type_t<A, B> min(A a, B b) {
  return b < a ? b : a;
}
template <typename A, typename B>
inline std::common_type_t<A, B> max(A a, B b) {
  return a < b ? b : a;
}
inline float min(float a, float b) { return std::fmin(a, b); }
inline float max(float a, float b) { return std::fmax(a, b); }

// A one-thread block has nobody to wait for. In a larger block the threads
// run one after another, so the first to arrive would read what later
// threads have not written yet: stop instead of racing.
inline void __syncthreads() {
  const dim3& b = ::beamz::cpu::block_dim;
  if (b.x * b.y * b.z == 1) return;
  std::fputs("beamz._cpu: a kernel reached __syncthreads in a block of more "
             "than one thread; launch it with BEAMZ_BARRIER_BLOCK\n", stderr);
  std::abort();
}

// Shared memory is ordinary cache-resident memory: copy synchronously.
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
// Busy is the summed time workers spent on blocks; busy / (seconds x team)
// is the launch's parallel efficiency (the rest is waiting for stragglers).
struct KernelProfile {
  struct Entry {
    double seconds = 0;
    long launches = 0;
    double blocks = 0;
    double busy = 0;
    double capacity = 0;
  };
  std::mutex mutex;
  std::map<std::string, Entry> entries;
  ~KernelProfile() {
    double total = 0;
    for (auto& [name, e] : entries) total += e.seconds;
    std::fprintf(stderr, "beamz cpu profile: %.3f s in kernels\n", total);
    for (auto& [name, e] : entries)
      std::fprintf(stderr, "%8.3f s %5.1f%% %8ld launches %10.0f blocks/launch %5.1f%% busy  %s\n",
                   e.seconds, 100 * e.seconds / total, e.launches,
                   e.blocks / e.launches,
                   e.capacity > 0 ? 100 * e.busy / e.capacity : 0.0,
                   name.c_str());
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

// Each worker keeps its largest shared-memory buffer for later launches.
inline void* SharedScratch(size_t bytes) {
  struct Buffer {
    void* data = nullptr;
    size_t size = 0;
    ~Buffer() { std::free(data); }
  };
  static thread_local Buffer buffer;
  if (bytes > buffer.size) {
    std::free(buffer.data);
    const size_t size = (bytes + 63) / 64 * 64;
    buffer.data = std::aligned_alloc(64, size);
    buffer.size = buffer.data ? size : 0;
  }
  return buffer.data;
}

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
    // Shared memory implies cooperating threads, which only a one-thread
    // block can run in order (see BEAMZ_BARRIER_BLOCK).
    if (shared_bytes != 0 &&
        static_cast<int64_t>(block.x) * block.y * block.z != 1) {
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
      double busy = 0;
      int team = 1;
      ~Record() {
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start).count();
        auto& p = KernelProfile::Get();
        std::lock_guard<std::mutex> lock(p.mutex);
        auto& e = p.entries[KernelName<Kernel>()];
        e.seconds += s;
        e.launches += 1;
        e.blocks += blocks;
        e.busy += busy;
        e.capacity += s * team;
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
    // A one-thread block with shared memory stands for a whole GPU block's
    // tile, so every such launch is worth the team.
    const int64_t work = blocks * static_cast<int64_t>(b.x) * b.y * b.z;
    const bool parallel = shared_bytes ? blocks > 1 : work >= kParallelThreshold;
#endif
    bool out_of_memory = false;
#if defined(_OPENMP)
#pragma omp parallel if (parallel)
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
    void* const scratch = shared_bytes ? SharedScratch(shared_bytes) : nullptr;
    const bool ready = shared_bytes == 0 || scratch != nullptr;
    if (!ready) {
#if defined(_OPENMP)
#pragma omp atomic write
#endif
      out_of_memory = true;
    }
    shared_scratch = scratch;
#if defined(BEAMZ_CPU_PROFILE)
    const auto worker_start = std::chrono::steady_clock::now();
#endif
#if defined(_OPENMP) && defined(BEAMZ_CPU_PROFILE)
#pragma omp for schedule(dynamic, chunk) nowait
#elif defined(_OPENMP)
#pragma omp for schedule(dynamic, chunk)
#endif
    for (int64_t linear = 0; linear < blocks; ++linear) {
      if (!ready) continue;
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
#if defined(BEAMZ_CPU_PROFILE)
    const double busy = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - worker_start)
                            .count();
#if defined(_OPENMP)
#pragma omp atomic
#endif
    record.busy += busy;
#if defined(_OPENMP)
#pragma omp single nowait
    record.team = omp_get_num_threads();
#endif
#endif
    }
    if (out_of_memory) beamz_cpu_last_error = cudaErrorMemoryAllocation;
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
