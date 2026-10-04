// Isolated periodic, homogeneous Yee stencil. No CPML, sources, monitors,
// intermediate-state publication, or physical boundary kernels. This is a
// diagnostic ceiling experiment, not a production backend or application GCUPS.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

void Check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}
struct Shape {
  int z, y, x;
  int Count() const { return z * y * x; }
};
struct Fields {
  float *f[6];
};
Fields View(float *data, int count) {
  Fields value{};
  for (int c = 0; c < 6; ++c)
    value.f[c] = data + size_t(c) * count;
  return value;
}
__host__ __device__ int Wrap(int x, int n) {
  if (x >= 0 && x < n)
    return x;
  return ((x % n) + n) % n;
}
__host__ __device__ int Index(Shape s, int z, int y, int x) {
  return (Wrap(z, s.z) * s.y + Wrap(y, s.y)) * s.x + Wrap(x, s.x);
}
__device__ __forceinline__ float Global(Fields f, Shape s, int c, int z, int y,
                                        int x) {
  return f.f[c][Index(s, z, y, x)];
}
constexpr int SourceA[3] = {2, 0, 1}, SourceB[3] = {1, 2, 0};
constexpr int AxisA[3] = {1, 0, 2}, AxisB[3] = {0, 2, 1};

template <int Phase> __global__ void PhaseKernel(Fields f, Shape s) {
  constexpr int SourceA[3] = {2, 0, 1}, SourceB[3] = {1, 2, 0};
  constexpr int AxisA[3] = {1, 0, 2}, AxisB[3] = {0, 2, 1};
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= s.z * s.y * s.x)
    return;
  int x = i % s.x, y = (i / s.x) % s.y, z = i / (s.x * s.y);
  constexpr int own = Phase * 3, other = 3 - own;
  constexpr int sign = Phase == 0 ? 1 : -1;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    auto diff = [&](int component, int axis) {
      float center = Global(f, s, other + component, z, y, x);
      float adjacent =
          Global(f, s, other + component, z + (axis == 0 ? sign : 0),
                 y + (axis == 1 ? sign : 0), x + (axis == 2 ? sign : 0));
      return sign * (adjacent - center);
    };
    float curl = diff(SourceA[c], AxisA[c]) - diff(SourceB[c], AxisB[c]);
    f.f[own + c][i] += (Phase == 0 ? -0.5f : 0.5f) * curl;
  }
}

// Each stage holds two z planes. Successive H/E stages shrink their transverse
// halo by one cell; each H/E pair advances one z plane behind the preceding
// pair.
template <int N, int X, int Y, int K> struct Stage {
  static constexpr int halo = 2 * N - 1 - K;
  static constexpr int sx = X + halo, sy = Y + halo;
  static constexpr int offset = [] {
    int value = 0;
    for (int k = 0; k < K; ++k)
      value += 6 * (X + 2 * N - 1 - k) * (Y + 2 * N - 1 - k);
    return value;
  }();
  __device__ static __forceinline__ float &At(float *shared, int c, int z,
                                              int y, int x) {
    return shared[offset + ((c * 2 + (z & 1)) * sy + y) * sx + x];
  }
};

template <int N, int X, int Y, int K = 0>
__device__ __forceinline__ void AdvanceStages(float *shared, Fields in,
                                              Fields out, Shape s, int ox,
                                              int oy, int oz, int wave) {
  constexpr int SourceA[3] = {2, 0, 1}, SourceB[3] = {1, 2, 0};
  constexpr int AxisA[3] = {1, 0, 2}, AxisB[3] = {0, 2, 1};
  using Here = Stage<N, X, Y, K>;
  constexpr int phase = K % 2, own = phase * 3;
  constexpr int shift = phase == 1 ? 1 : 0;
  constexpr int sign = phase == 0 ? 1 : -1;
  const int z = oz + wave - N - K / 2;
  if (wave >= K) {
    for (int i = threadIdx.x; i < Here::sx * Here::sy; i += blockDim.x) {
      int ix = i % Here::sx, iy = i / Here::sx;
      int x = ox - N + (K + 1) / 2 + ix;
      int y = oy - N + (K + 1) / 2 + iy;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float old;
        if constexpr (K < 2)
          old = Global(in, s, own + c, z, y, x);
        else
          old = Stage<N, X, Y, K - 2>::At(shared, c, z, iy + 1, ix + 1);
        auto read_other = [&](int component, int dz, int dy, int dx) {
          if constexpr (K == 0)
            return Global(in, s, 3 + component, z + dz, y + dy, x + dx);
          else
            return Stage<N, X, Y, K - 1>::At(shared, component, z + dz,
                                             iy + shift + dy, ix + shift + dx);
        };
        auto diff = [&](int component, int axis) {
          float center = read_other(component, 0, 0, 0);
          float adjacent =
              read_other(component, axis == 0 ? sign : 0, axis == 1 ? sign : 0,
                         axis == 2 ? sign : 0);
          return sign * (adjacent - center);
        };
        float curl = diff(SourceA[c], AxisA[c]) - diff(SourceB[c], AxisB[c]);
        float next = old + (phase == 0 ? -0.5f : 0.5f) * curl;
        if constexpr (K == 2 * N - 1) {
          if (z < s.z && y < s.y && x < s.x) {
            int global = (z * s.y + y) * s.x + x;
            out.f[3 + c][global] = next;
            out.f[c][global] =
                Stage<N, X, Y, K - 1>::At(shared, c, z, iy + 1, ix + 1);
          }
        } else
          Here::At(shared, c, z, iy, ix) = next;
      }
    }
  }
  __syncthreads();
  if constexpr (K + 1 < 2 * N)
    AdvanceStages<N, X, Y, K + 1>(shared, in, out, s, ox, oy, oz, wave);
}

template <int N, int X, int Y, int Z>
__global__ void RingKernel(Fields in, Fields out, Shape s) {
  extern __shared__ float shared[];
  for (int wave = 0; wave < Z + 2 * N - 1; ++wave)
    AdvanceStages<N, X, Y>(shared, in, out, s, blockIdx.x * X, blockIdx.y * Y,
                           blockIdx.z * Z, wave);
}

using Launch = void (*)(Fields, Fields, Shape, cudaStream_t);
struct Variant {
  const char *name;
  int depth;
  bool in_place;
  Launch launch;
  int shared, registers, local, blocks;
};
template <int N, int X, int Y, int Z> Variant Ring(const char *name) {
  constexpr int bytes = Stage<N, X, Y, 2 * N - 1>::offset * sizeof(float);
  Check(cudaFuncSetAttribute(RingKernel<N, X, Y, Z>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             bytes));
  cudaFuncAttributes attributes{};
  Check(cudaFuncGetAttributes(&attributes, RingKernel<N, X, Y, Z>));
  int active;
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, RingKernel<N, X, Y, Z>, 256, bytes));
  return {
      name,
      N,
      false,
      [](Fields a, Fields b, Shape s, cudaStream_t stream) {
        RingKernel<N, X, Y, Z>
            <<<dim3((s.x + X - 1) / X, (s.y + Y - 1) / Y, (s.z + Z - 1) / Z),
               256, bytes, stream>>>(a, b, s);
      },
      bytes,
      attributes.numRegs,
      static_cast<int>(attributes.localSizeBytes),
      active};
}
void Separate(Fields a, Fields, Shape s, cudaStream_t stream) {
  PhaseKernel<0><<<(s.Count() + 255) / 256, 256, 0, stream>>>(a, s);
  PhaseKernel<1><<<(s.Count() + 255) / 256, 256, 0, stream>>>(a, s);
}
std::vector<Variant> Variants() {
  return {{"separate", 1, true, Separate, 0, 0, 0, 0},
          Ring<1, 32, 8, 8>("ring1_32x8x8"),
          Ring<1, 32, 8, 16>("ring1_32x8x16"),
          Ring<2, 32, 8, 8>("ring2_32x8x8"),
          Ring<2, 32, 8, 16>("ring2_32x8x16"),
          Ring<2, 32, 8, 32>("ring2_32x8x32"),
          Ring<2, 64, 4, 16>("ring2_64x4x16"),
          Ring<2, 16, 8, 16>("ring2_16x8x16"),
          Ring<4, 32, 4, 16>("ring4_32x4x16"),
          Ring<4, 32, 4, 32>("ring4_32x4x32"),
          Ring<4, 16, 8, 16>("ring4_16x8x16"),
          Ring<8, 8, 4, 32>("ring8_8x4x32")};
}

struct Graph {
  cudaGraphExec_t executable;
  bool final_a;
};
Graph Capture(const Variant &v, Fields a, Fields b, Shape s, int steps,
              cudaStream_t stream) {
  cudaGraph_t graph;
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  bool current_a = true;
  int remaining = steps;
  while (remaining >= v.depth) {
    v.launch(current_a ? a : b, current_a ? b : a, s, stream);
    if (!v.in_place)
      current_a = !current_a;
    remaining -= v.depth;
  }
  while (remaining-- > 0)
    Separate(current_a ? a : b, {}, s, stream);
  Check(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t executable;
  Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  Check(cudaGraphDestroy(graph));
  return {executable, current_a};
}
std::vector<float> Initial(Shape s) {
  std::mt19937 rng(20260918);
  std::uniform_real_distribution<float> dist(-.01f, .01f);
  std::vector<float> value(size_t(s.Count()) * 6);
  for (auto &x : value)
    x = dist(rng);
  return value;
}
void CpuAdvance(std::vector<float> &value, Shape s, int steps) {
  int n = s.Count();
  for (int t = 0; t < steps; ++t)
    for (int phase = 0; phase < 2; ++phase) {
      int own = phase * 3, other = 3 - own, sign = phase == 0 ? 1 : -1;
      for (int z = 0; z < s.z; ++z)
        for (int y = 0; y < s.y; ++y)
          for (int x = 0; x < s.x; ++x) {
            int i = (z * s.y + y) * s.x + x;
            for (int c = 0; c < 3; ++c) {
              auto diff = [&](int component, int axis) {
                int adjacent = Index(s, z + (axis == 0 ? sign : 0),
                                     y + (axis == 1 ? sign : 0),
                                     x + (axis == 2 ? sign : 0));
                return sign * (value[(other + component) * n + adjacent] -
                               value[(other + component) * n + i]);
              };
              value[(own + c) * n + i] +=
                  (phase == 0 ? -.5f : .5f) *
                  (diff(SourceA[c], AxisA[c]) - diff(SourceB[c], AxisB[c]));
            }
          }
    }
}
float Compare(const std::vector<float> &expected,
              const std::vector<float> &actual) {
  float worst = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    float error = std::abs(expected[i] - actual[i]);
    if (!std::isfinite(actual[i]) ||
        error > 2e-6f + 3e-5f * std::abs(expected[i])) {
      std::fprintf(stderr, "Mismatch %zu expected %.9g got %.9g\n", i,
                   expected[i], actual[i]);
      std::exit(2);
    }
    worst = std::max(worst, error);
  }
  return worst;
}

void Run(Shape s, int steps, int rounds, bool cpu_oracle) {
  auto variants = Variants();
  auto initial = Initial(s), expected = initial;
  size_t bytes = initial.size() * sizeof(float);
  float *a, *b, *seed;
  Check(cudaMalloc(&a, bytes));
  Check(cudaMalloc(&b, bytes));
  Check(cudaMalloc(&seed, bytes));
  Check(cudaMemcpy(seed, initial.data(), bytes, cudaMemcpyHostToDevice));
  cudaStream_t stream;
  Check(cudaStreamCreate(&stream));
  std::vector<Graph> graphs;
  for (auto &v : variants)
    graphs.push_back(
        Capture(v, View(a, s.Count()), View(b, s.Count()), s, steps, stream));
  if (cpu_oracle)
    CpuAdvance(expected, s, steps);
  std::vector<float> errors;
  for (size_t i = 0; i < variants.size(); ++i) {
    Check(cudaMemcpyAsync(a, seed, bytes, cudaMemcpyDeviceToDevice, stream));
    Check(cudaGraphLaunch(graphs[i].executable, stream));
    Check(cudaStreamSynchronize(stream));
    std::vector<float> actual(initial.size());
    Check(cudaMemcpy(actual.data(), graphs[i].final_a ? a : b, bytes,
                     cudaMemcpyDeviceToHost));
    if (i == 0 && !cpu_oracle)
      expected = actual;
    errors.push_back(Compare(expected, actual));
  }
  std::vector<std::vector<float>> timings(variants.size());
  cudaEvent_t start, stop;
  Check(cudaEventCreate(&start));
  Check(cudaEventCreate(&stop));
  std::mt19937 rng(20260918);
  std::vector<int> order(variants.size());
  for (size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  for (int round = -3; round < rounds; ++round) {
    std::shuffle(order.begin(), order.end(), rng);
    for (int i : order) {
      Check(cudaMemcpyAsync(a, seed, bytes, cudaMemcpyDeviceToDevice, stream));
      Check(cudaEventRecord(start, stream));
      Check(cudaGraphLaunch(graphs[i].executable, stream));
      Check(cudaEventRecord(stop, stream));
      Check(cudaEventSynchronize(stop));
      float ms;
      Check(cudaEventElapsedTime(&ms, start, stop));
      if (round >= 0)
        timings[i].push_back(ms);
    }
  }
  std::printf(
      "{\"shape\":[%d,%d,%d],\"steps\":%d,\"cpu_oracle\":%s,\"variants\":[",
      s.z, s.y, s.x, steps, cpu_oracle ? "true" : "false");
  for (size_t i = 0; i < variants.size(); ++i) {
    auto v = variants[i];
    auto sorted = timings[i];
    std::sort(sorted.begin(), sorted.end());
    float median =
        (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) * .5f;
    std::printf("%s{\"name\":\"%s\",\"depth\":%d,\"gcups\":%.9g,\"shared_"
                "bytes\":%d,\"registers\":%d,\"local_bytes\":%d,\"active_"
                "blocks_per_sm\":%d,\"max_abs_error\":%.9g,\"samples_ms\":[",
                i ? "," : "", v.name, v.depth,
                double(s.Count()) * steps / median / 1e6, v.shared, v.registers,
                v.local, v.blocks, errors[i]);
    for (size_t j = 0; j < timings[i].size(); ++j)
      std::printf("%s%.9g", j ? "," : "", timings[i][j]);
    std::printf("]}");
  }
  std::puts("]}");
  std::fflush(stdout);
  for (auto g : graphs)
    Check(cudaGraphExecDestroy(g.executable));
  Check(cudaEventDestroy(start));
  Check(cudaEventDestroy(stop));
  Check(cudaStreamDestroy(stream));
  Check(cudaFree(seed));
  Check(cudaFree(b));
  Check(cudaFree(a));
}
int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--check") {
    Run({17, 23, 35}, 9, 1, true);
    Run({33, 39, 67}, 16, 1, true);
    return 0;
  }
  if (argc != 6) {
    std::fprintf(stderr, "usage: bulk Z Y X STEPS ROUNDS | --check\n");
    return 1;
  }
  Shape s{std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};
  int steps = std::atoi(argv[4]), rounds = std::atoi(argv[5]);
  if (std::min({s.z, s.y, s.x}) < 9 || steps < 1 || rounds < 1 ||
      double(s.z) * s.y * s.x > 100000000)
    return 1;
  Run(s, steps, rounds, false);
}
