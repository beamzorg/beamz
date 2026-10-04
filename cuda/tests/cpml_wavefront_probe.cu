// Dependency-schedule probe, not the production Yee layout or a realistic
// application benchmark. Common field extents, packed 12-cell FP32 CPML,
// heterogeneous lossless coefficients, planar test excitation, three DFT bins.
// Identical arithmetic is used by global H/E sweeps and in-place wavefronts.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>

void Check(cudaError_t s) {
  if (s != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(s));
    std::exit(1);
  }
}
struct Grid {
  int z, y, x;
  int Count() const { return z * y * x; }
};
struct State {
  float *f[6], *psi[12], *monitor;
  const float *wave, *phasor;
};
struct Task {
  int z, y, x, pass;
};
constexpr int Pml = 12;
__device__ int Offset(Grid g, int z, int y, int x) {
  return (z * g.y + y) * g.x + x;
}
__device__ float Read(const float *f, Grid g, int z, int y, int x) {
  return z >= 0 && z < g.z && y >= 0 && y < g.y && x >= 0 && x < g.x
             ? f[Offset(g, z, y, x)]
             : 0.f;
}
template <int Axis, bool Shared = false>
__device__ float Correct(float d, float *psi, Grid g, int z, int y, int x,
                         Grid storage = {}, Grid origin = {}) {
  int c = Axis == 0   ? z
          : Axis == 1 ? y
                      : x,
      n = Axis == 0   ? g.z
          : Axis == 1 ? g.y
                      : g.x;
  int packed = c < Pml ? c : c >= n - Pml ? Pml + c - (n - Pml) : -1;
  if (packed < 0)
    return d;
  int index = Axis == 0   ? (packed * g.y + y) * g.x + x
              : Axis == 1 ? (z * 2 * Pml + packed) * g.x + x
                          : (z * g.y + y) * 2 * Pml + packed;
  if constexpr (Shared)
    index = Offset({storage.z - 2, storage.y - 2, storage.x - 2},
                   z - origin.z - 1, y - origin.y - 1, x - origin.x - 1);
  float strength =
      packed < Pml ? float(Pml - packed) / Pml : float(packed - Pml + 1) / Pml;
  float a = -.08f * strength, b = .99f - .09f * strength,
        k = 1.f / (1.f + 2.f * strength);
  float next = __fmaf_rn(b, psi[index], __fmul_rn(a, d));
  psi[index] = next;
  return __fmaf_rn(k, d, next);
}
template <int Phase, int C, bool Shared = false>
__device__ void Component(State s, Grid g, int z, int y, int x, int step,
                          Grid storage = {}, Grid origin = {}) {
  constexpr int sa[3] = {2, 0, 1}, sb[3] = {1, 2, 0}, aa[3] = {1, 0, 2},
                ab[3] = {0, 2, 1};
  constexpr int sign = Phase == 0 ? 1 : -1, other = (1 - Phase) * 3;
  const int i = Shared
                    ? Offset(storage, z - origin.z, y - origin.y, x - origin.x)
                    : Offset(g, z, y, x);
  auto diff = [&](int c, int axis) {
    float a =
        Shared ? Read(s.f[other + c], storage,
                      z - origin.z + (axis == 0 ? sign : 0),
                      y - origin.y + (axis == 1 ? sign : 0),
                      x - origin.x + (axis == 2 ? sign : 0))
               : Read(s.f[other + c], g, z + (axis == 0 ? sign : 0),
                      y + (axis == 1 ? sign : 0), x + (axis == 2 ? sign : 0));
    float b = s.f[other + c][i];
    return Phase == 0 ? a - b : b - a;
  };
  float da =
      Correct<aa[C], Shared>(diff(sa[C], aa[C]), s.psi[Phase * 6 + 2 * C], g, z,
                             y, x, storage, origin);
  float db =
      Correct<ab[C], Shared>(diff(sb[C], ab[C]), s.psi[Phase * 6 + 2 * C + 1],
                             g, z, y, x, storage, origin);
  float coefficient =
      Phase == 0 ? -.18f : ((x / 7 + y / 11 + z / 13) % 2 ? .045f : .18f);
  float next = __fmaf_rn(coefficient, da - db, s.f[Phase * 3 + C][i]);
  // Analytic planar test pattern, not an eigenmode-solver fixture.
  if (C == 1 && x == Pml + 2 && z > Pml && z < g.z - Pml && y > Pml &&
      y < g.y - Pml)
    next += s.wave[2 * step + Phase] * float((y % 7) + 1) / 7.f;
  s.f[Phase * 3 + C][i] = next;
}
template <int Phase, bool Shared = false>
__device__ void Cell(State s, Grid g, int z, int y, int x, int step,
                     Grid storage = {}, Grid origin = {}) {
  Component<Phase, 0, Shared>(s, g, z, y, x, step, storage, origin);
  Component<Phase, 1, Shared>(s, g, z, y, x, step, storage, origin);
  Component<Phase, 2, Shared>(s, g, z, y, x, step, storage, origin);
  if constexpr (Phase == 1) {
    if (z == g.z / 2 && y == g.y / 2 && x == g.x / 2) {
      int i = Shared ? Offset(storage, z - origin.z, y - origin.y, x - origin.x)
                     : Offset(g, z, y, x);
      float value = s.f[4][i] + s.f[1][i];
      for (int q = 0; q < 6; ++q)
        s.monitor[q] = __fmaf_rn(value, s.phasor[6 * step + q], s.monitor[q]);
    }
  }
}
template <int Phase> __global__ void Sweep(State s, Grid g, int step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < g.z * g.y * g.x)
    Cell<Phase>(s, g, i / (g.x * g.y), (i / g.x) % g.y, i % g.x, step);
}
template <int Z, int Y, int X, int Depth, bool Unit>
__global__ void Wavefront(State s, Grid g, const Task *tasks, int start) {
  Task task = tasks[blockIdx.x];
  for (int t = 0; t < Depth; ++t) {
    int oz = task.z * Z - t * (Unit ? 1 : Z / Depth),
        oy = task.y * Y - t * (Unit ? 1 : Y / Depth),
        ox = task.x * X - t * (Unit ? 1 : X / Depth);
    int step = start + task.pass * Depth + t;
    for (int j = threadIdx.x; j < X * Y * Z; j += blockDim.x) {
      int x = ox + j % X, y = oy + (j / X) % Y, z = oz + j / (X * Y);
      if (x >= 0 && x < g.x && y >= 0 && y < g.y && z >= 0 && z < g.z)
        Cell<0>(s, g, z, y, x, step);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < X * Y * Z; j += blockDim.x) {
      int x = ox + j % X, y = oy + (j / X) % Y, z = oz + j / (X * Y);
      if (x >= 0 && x < g.x && y >= 0 && y < g.y && z >= 0 && z < g.z)
        Cell<1>(s, g, z, y, x, step);
    }
    __syncthreads();
  }
}
template <int Z, int Y, int X, int D>
__host__ __device__ bool SharedOwned(int z, int y, int x) {
  int low = max(0, max(D - z, max(D - y, D - x)));
  int high = min(D - 1, min(D + Z - 1 - z, min(D + Y - 1 - y, D + X - 1 - x)));
  return low <= high;
}
// Host-side footprint classification; padded storage never changes PML
// thickness.
template <int Z, int Y, int X, int D> int SharedMask(Task task, Grid g) {
  int mask = 0, b[3] = {task.z * Z, task.y * Y, task.x * X},
      n[3] = {g.z, g.y, g.x}, size[3] = {Z, Y, X};
  for (int a = 0; a < 3; ++a)
    if (b[a] - (D - 1) < Pml || b[a] + size[a] > n[a] - Pml)
      mask |= 1 << a;
  return mask;
}
__host__ __device__ constexpr int MaskCount(int mask) {
  return (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1);
}
__device__ int GlobalPsiIndex(int axis, Grid g, int z, int y, int x) {
  int c = axis == 0   ? z
          : axis == 1 ? y
                      : x,
      n = axis == 0   ? g.z
          : axis == 1 ? g.y
                      : g.x;
  int packed = c < Pml ? c : c >= n - Pml ? Pml + c - (n - Pml) : -1;
  if (packed < 0)
    return -1;
  return axis == 0   ? (packed * g.y + y) * g.x + x
         : axis == 1 ? (z * 2 * Pml + packed) * g.x + x
                     : (z * g.y + y) * 2 * Pml + packed;
}
template <int Z, int Y, int X, int D, int Mask>
__global__ void SharedWavefront(State global, Grid g, const Task *tasks,
                                int start) {
  constexpr int SZ = Z + D + 1, SY = Y + D + 1, SX = X + D + 1,
                V = SZ * SY * SX;
  constexpr int PZ = SZ - 2, PY = SY - 2, PX = SX - 2, PV = PZ * PY * PX;
  Task task = tasks[blockIdx.x];
  Grid origin{task.z * Z - D, task.y * Y - D, task.x * X - D},
      storage{SZ, SY, SX};
  if (origin.z + SZ <= 1 || origin.y + SY <= 1 || origin.x + SX <= 1 ||
      origin.z + 1 >= g.z || origin.y + 1 >= g.y || origin.x + 1 >= g.x)
    return;
  extern __shared__ float data[];
  State local = global;
#pragma unroll
  for (int c = 0; c < 6; ++c)
    local.f[c] = data + c * V;
  constexpr int axes[6] = {1, 0, 0, 2, 2, 1};
  int slot = 0;
#pragma unroll
  for (int t = 0; t < 12; ++t) {
    local.psi[t] =
        (Mask & (1 << axes[t % 6])) ? data + 6 * V + (slot++) * PV : nullptr;
  }
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    int x = i % SX, y = (i / SX) % SY, z = i / (SX * SY);
    bool read = SharedOwned<Z, Y, X, D>(z, y, x) ||
                SharedOwned<Z, Y, X, D>(z - 1, y, x) ||
                SharedOwned<Z, Y, X, D>(z + 1, y, x) ||
                SharedOwned<Z, Y, X, D>(z, y - 1, x) ||
                SharedOwned<Z, Y, X, D>(z, y + 1, x) ||
                SharedOwned<Z, Y, X, D>(z, y, x - 1) ||
                SharedOwned<Z, Y, X, D>(z, y, x + 1);
#pragma unroll
    for (int c = 0; c < 6; ++c)
      local.f[c][i] =
          read ? Read(global.f[c], g, z + origin.z, y + origin.y, x + origin.x)
               : 0.f;
  }
  for (int i = threadIdx.x; i < PV; i += blockDim.x) {
    int x = i % PX + 1, y = (i / PX) % PY + 1, z = i / (PX * PY) + 1;
    bool own = SharedOwned<Z, Y, X, D>(z, y, x);
    int gx = x + origin.x, gy = y + origin.y, gz = z + origin.z;
    bool valid =
        gx >= 0 && gx < g.x && gy >= 0 && gy < g.y && gz >= 0 && gz < g.z;
#pragma unroll
    for (int t = 0; t < 12; ++t)
      if (Mask & (1 << axes[t % 6])) {
        int index =
            own && valid ? GlobalPsiIndex(axes[t % 6], g, gz, gy, gx) : -1;
        local.psi[t][i] = index >= 0 ? global.psi[t][index] : 0.f;
      }
  }
  __syncthreads();
  for (int t = 0; t < D; ++t) {
    int oz = task.z * Z - t, oy = task.y * Y - t, ox = task.x * X - t,
        step = start + task.pass * D + t;
    for (int j = threadIdx.x; j < X * Y * Z; j += blockDim.x) {
      int x = ox + j % X, y = oy + (j / X) % Y, z = oz + j / (X * Y);
      if (x >= 0 && x < g.x && y >= 0 && y < g.y && z >= 0 && z < g.z)
        Cell<0, true>(local, g, z, y, x, step, storage, origin);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < X * Y * Z; j += blockDim.x) {
      int x = ox + j % X, y = oy + (j / X) % Y, z = oz + j / (X * Y);
      if (x >= 0 && x < g.x && y >= 0 && y < g.y && z >= 0 && z < g.z)
        Cell<1, true>(local, g, z, y, x, step, storage, origin);
    }
    __syncthreads();
  }
  // Publish only owned values. Read-only halo cells may belong to concurrent
  // tasks.
  for (int i = threadIdx.x; i < PV; i += blockDim.x) {
    int x = i % PX + 1, y = (i / PX) % PY + 1, z = i / (PX * PY) + 1,
        gx = x + origin.x, gy = y + origin.y, gz = z + origin.z;
    if (!SharedOwned<Z, Y, X, D>(z, y, x) || gx < 0 || gx >= g.x || gy < 0 ||
        gy >= g.y || gz < 0 || gz >= g.z)
      continue;
    int fi = Offset(storage, z, y, x), gi = Offset(g, gz, gy, gx);
#pragma unroll
    for (int c = 0; c < 6; ++c)
      global.f[c][gi] = local.f[c][fi];
#pragma unroll
    for (int t = 0; t < 12; ++t)
      if (Mask & (1 << axes[t % 6])) {
        int index = GlobalPsiIndex(axes[t % 6], g, gz, gy, gx);
        if (index >= 0)
          global.psi[t][index] = local.psi[t][i];
      }
  }
}
template <int Z, int Y, int X, int D, int Mask> constexpr int SharedBytes() {
  return 4 * (6 * (Z + D + 1) * (Y + D + 1) * (X + D + 1) +
              4 * MaskCount(Mask) * (Z + D - 1) * (Y + D - 1) * (X + D - 1));
}
template <int Z, int Y, int X, int D> void PrepareShared() {
#define PREP(M)                                                                \
  Check(cudaFuncSetAttribute(SharedWavefront<Z, Y, X, D, M>,                   \
                             cudaFuncAttributeMaxDynamicSharedMemorySize,      \
                             SharedBytes<Z, Y, X, D, M>()));
  PREP(0);
  PREP(1);
  PREP(2);
  PREP(3);
  PREP(4);
  PREP(5);
  PREP(6);
  PREP(7);
#undef PREP
}
template <int Z, int Y, int X, int D>
void LaunchShared(cudaStream_t stream, State s, Grid g, const Task *tasks,
                  const std::vector<Task> &host, int start) {
  size_t begin = 0;
#define LAUNCH(M)                                                              \
  {                                                                            \
    size_t end = begin;                                                        \
    while (end < host.size() && SharedMask<Z, Y, X, D>(host[end], g) == M)     \
      ++end;                                                                   \
    if (end > begin)                                                           \
      SharedWavefront<Z, Y, X, D, M>                                           \
          <<<end - begin, 256, SharedBytes<Z, Y, X, D, M>(), stream>>>(        \
              s, g, tasks + begin, start);                                     \
    begin = end;                                                               \
  }
  LAUNCH(0);
  LAUNCH(1);
  LAUNCH(2);
  LAUNCH(3);
  LAUNCH(4);
  LAUNCH(5);
  LAUNCH(6);
  LAUNCH(7);
#undef LAUNCH
}
struct Allocation {
  Grid g;
  State s{};
  std::vector<float *> buffers;
  std::vector<size_t> sizes;
  Allocation(Grid grid, const std::vector<std::vector<float>> &initial,
             const float *wave, const float *phasor)
      : g(grid) {
    for (const auto &v : initial) {
      float *p;
      Check(cudaMalloc(&p, v.size() * sizeof(float)));
      Check(cudaMemcpy(p, v.data(), v.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
      buffers.push_back(p);
      sizes.push_back(v.size());
    }
    for (int c = 0; c < 6; ++c)
      s.f[c] = buffers.at(c);
    for (int t = 0; t < 12; ++t)
      s.psi[t] = buffers.at(6 + t);
    s.monitor = buffers.at(18);
    s.wave = wave;
    s.phasor = phasor;
  }
  ~Allocation() {
    for (auto p : buffers)
      Check(cudaFree(p));
  }
};
void Compare(const Allocation &a, const Allocation &b) {
  for (size_t c = 0; c < a.buffers.size(); ++c) {
    std::vector<float> x(a.sizes[c]), y(a.sizes[c]);
    Check(cudaMemcpy(x.data(), a.buffers[c], x.size() * 4,
                     cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(y.data(), b.buffers[c], y.size() * 4,
                     cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < x.size(); ++i)
      if (!std::isfinite(x[i]) || !std::isfinite(y[i]) ||
          std::memcmp(&x[i], &y[i], 4)) {
        std::fprintf(
            stderr,
            "Mismatch buffer=%zu index=%zu reference=%.9g actual=%.9g\n", c, i,
            x[i], y[i]);
        std::exit(2);
      }
  }
}
template <int Depth, bool Unit, bool Shared = false, int Z = 16, int Y = 8,
          int X = 16>
void Run(Grid g, int steps, int split, bool benchmark = false) {
  if constexpr (Shared)
    PrepareShared<Z, Y, X, Depth>();
  std::mt19937 rng(41);
  std::uniform_real_distribution<float> d(-.001f, .001f);
  std::vector<std::vector<float>> initial;
  for (int c = 0; c < 6; ++c)
    initial.emplace_back(g.Count());
  constexpr int axes[6] = {1, 0, 0, 2, 2, 1};
  for (int t = 0; t < 12; ++t) {
    int a = axes[t % 6];
    initial.emplace_back(size_t(2 * Pml) * (a == 0   ? g.y * g.x
                                            : a == 1 ? g.z * g.x
                                                     : g.z * g.y));
  }
  initial.emplace_back(6, 0.f);
  for (size_t c = 0; c + 1 < initial.size(); ++c)
    for (auto &v : initial[c])
      v = d(rng);
  std::vector<float> wave(2 * steps), phasor(6 * steps);
  for (int t = 0; t < steps; ++t) {
    wave[2 * t] = .001f * std::sin(.13f * t);
    wave[2 * t + 1] = .002f * std::sin(.13f * (t + .5f));
    for (int f = 0; f < 3; ++f) {
      phasor[6 * t + 2 * f] = std::cos((.05f + .01f * f) * t);
      phasor[6 * t + 2 * f + 1] = std::sin((.05f + .01f * f) * t);
    }
  }
  float *dw, *dp;
  Check(cudaMalloc(&dw, wave.size() * 4));
  Check(cudaMalloc(&dp, phasor.size() * 4));
  Check(cudaMemcpy(dw, wave.data(), wave.size() * 4, cudaMemcpyHostToDevice));
  Check(
      cudaMemcpy(dp, phasor.data(), phasor.size() * 4, cudaMemcpyHostToDevice));
  Allocation reference(g, initial, dw, dp), candidate(g, initial, dw, dp);
  cudaStream_t stream;
  Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto segment = [&](int start, int end) {
    if (start == end)
      return;
    int passes = (end - start) / Depth, nz = (g.z + Z - 1) / Z + 1,
        ny = (g.y + Y - 1) / Y + 1, nx = (g.x + X - 1) / X + 1;
    std::vector<std::vector<Task>> ranks(nz + ny + nx + 4 * passes);
    for (int p = 0; p < passes; ++p)
      for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
          for (int x = 0; x < nx; ++x)
            ranks[z + y + x + 4 * p].push_back({z, y, x, p});
    std::vector<Task> all;
    for (auto &r : ranks) {
      std::shuffle(r.begin(), r.end(), rng);
      if constexpr (Shared)
        std::stable_sort(r.begin(), r.end(), [&](Task a, Task b) {
          return SharedMask<Z, Y, X, Depth>(a, g) <
                 SharedMask<Z, Y, X, Depth>(b, g);
        });
      all.insert(all.end(), r.begin(), r.end());
    }
    Task *tasks = nullptr;
    if (!all.empty()) {
      Check(cudaMalloc(&tasks, all.size() * sizeof(Task)));
      Check(cudaMemcpy(tasks, all.data(), all.size() * sizeof(Task),
                       cudaMemcpyHostToDevice));
    }
    auto emit = [&](bool wavefront) {
      if (!wavefront) {
        for (int t = start; t < end; ++t) {
          Sweep<0>
              <<<(g.Count() + 255) / 256, 256, 0, stream>>>(reference.s, g, t);
          Sweep<1>
              <<<(g.Count() + 255) / 256, 256, 0, stream>>>(reference.s, g, t);
        }
        return;
      }
      size_t offset = 0;
      for (auto &r : ranks) {
        if (!r.empty()) {
          if constexpr (Shared)
            LaunchShared<Z, Y, X, Depth>(stream, candidate.s, g, tasks + offset,
                                         r, start);
          else
            Wavefront<Z, Y, X, Depth, Unit><<<r.size(), 256, 0, stream>>>(
                candidate.s, g, tasks + offset, start);
        }
        offset += r.size();
      }
      for (int t = start + passes * Depth; t < end; ++t) {
        Sweep<0>
            <<<(g.Count() + 255) / 256, 256, 0, stream>>>(candidate.s, g, t);
        Sweep<1>
            <<<(g.Count() + 255) / 256, 256, 0, stream>>>(candidate.s, g, t);
      }
    };
    emit(false);
    emit(true);
    Check(cudaStreamSynchronize(stream));
    Compare(reference, candidate);
    if (benchmark) {
      cudaGraph_t graphs[2];
      cudaGraphExec_t executable[2];
      cudaEvent_t begin, finish;
      Check(cudaEventCreate(&begin));
      Check(cudaEventCreate(&finish));
      for (int v = 0; v < 2; ++v) {
        Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        emit(v);
        Check(cudaStreamEndCapture(stream, &graphs[v]));
        Check(cudaGraphInstantiate(&executable[v], graphs[v], 0));
      }
      for (int warm = 0; warm < 2; ++warm)
        for (int v = 0; v < 2; ++v)
          Check(cudaGraphLaunch(executable[v], stream));
      Check(cudaStreamSynchronize(stream));
      std::vector<float> times[2];
      for (int round = 0; round < 6; ++round)
        for (int q = 0; q < 2; ++q) {
          int v = q ^ (round & 1);
          Check(cudaEventRecord(begin, stream));
          Check(cudaGraphLaunch(executable[v], stream));
          Check(cudaEventRecord(finish, stream));
          Check(cudaEventSynchronize(finish));
          float ms;
          Check(cudaEventElapsedTime(&ms, begin, finish));
          times[v].push_back(ms);
        }
      Compare(reference, candidate);
      for (int v = 0; v < 2; ++v) {
        auto sorted = times[v];
        std::sort(sorted.begin(), sorted.end());
        float ms = (sorted[2] + sorted[3]) * .5f;
        std::printf("BENCH shape=%dx%dx%d depth=%d unit_shift=%d variant=%s "
                    "tile=%dx%dx%d gcups=%.6f samples_ms=",
                    g.z, g.y, g.x, Depth, Unit,
                    v ? (Shared ? "shared_wavefront" : "wavefront") : "sweep",
                    Z, Y, X, double(g.Count()) * (end - start) / (ms * 1e6));
        for (float t : times[v])
          std::printf("%.4f,", t);
        std::puts("");
        Check(cudaGraphExecDestroy(executable[v]));
        Check(cudaGraphDestroy(graphs[v]));
      }
      Check(cudaEventDestroy(begin));
      Check(cudaEventDestroy(finish));
    }
    if (tasks)
      Check(cudaFree(tasks));
  };
  segment(0, split);
  segment(split, steps);
  std::printf("PASS shape=%dx%dx%d depth=%d unit_shift=%d steps=%d split=%d "
              "shared=%d tile=%dx%dx%d fields/psi/DFT bitwise\n",
              g.z, g.y, g.x, Depth, Unit, steps, split, Shared, Z, Y, X);
  Check(cudaStreamDestroy(stream));
  Check(cudaFree(dw));
  Check(cudaFree(dp));
}
int main(int argc, char **argv) {
  if (argc == 6 && (std::strcmp(argv[1], "--bench") == 0 ||
                    std::strcmp(argv[1], "--bench-shared") == 0)) {
    Grid g{std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
    int steps = std::atoi(argv[5]);
    if (g.z <= 24 || g.y <= 24 || g.x <= 24 || steps <= 0 || steps > 4096 ||
        double(g.z) * g.y * g.x > 32000000)
      return 3;
    if (std::strcmp(argv[1], "--bench-shared") == 0) {
      Run<2, true, true, 8, 4, 8>(g, steps, steps, true);
      Run<4, true, true, 8, 4, 8>(g, steps, steps, true);
      return 0;
    }
    Run<8, false>(g, steps, steps, true);
    Run<8, true>(g, steps, steps, true);
    return 0;
  }
  bool small = argc > 1;
  std::vector<Grid> shapes =
      small ? std::vector<Grid>{{29, 31, 37}}
            : std::vector<Grid>{
                  {29, 31, 37}, {37, 29, 31}, {31, 37, 29}, {65, 49, 97}};
  for (auto g : shapes) {
    Run<2, false>(g, 35, 17);
    Run<4, false>(g, 67, 33);
    Run<8, false>(g, 131, 65);
    Run<8, true>(g, 131, 65);
    Run<2, true, true, 8, 4, 8>(g, 35, 17);
    Run<4, true, true, 8, 4, 8>(g, 67, 33);
  }
  return 0;
}
