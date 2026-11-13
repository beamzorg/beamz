// Research field-only moving boxes with phase-correct incoming-face caches.
// Included by cuda_moving_box_probe.cu after its descriptor/arithmetic helpers.
#pragma once

#ifndef MOVING_FACE_CACHE_BAD_H_PHASE
#define MOVING_FACE_CACHE_BAD_H_PHASE 0
#endif

namespace moving_face_cache {
constexpr int PZ = BZ + 1, PY = BY + 1, PX = BX + 1;
constexpr int PAD = PZ * PY * PX;
constexpr int LOW = PAD - V;
constexpr int PURE_FACES = BY * BX + BZ * BX + BZ * BY;
constexpr int OWNED_LANES = (V + 255) / 256;

__host__ __device__ __forceinline__ int Pad(int z, int y, int x) {
  return (z * PY + y) * PX + x;
}

// Disjoint low-face union, including intersections exactly once.
__device__ __forceinline__ void LowCoordinate(int i, int& z, int& y, int& x) {
  if (i < PY * PX) {
    z = 0; y = i / PX; x = i % PX;
  } else if ((i -= PY * PX) < BZ * PX) {
    z = 1 + i / PX; y = 0; x = i % PX;
  } else {
    i -= BZ * PX;
    z = 1 + i / BY; y = 1 + i % BY; x = 0;
  }
}

// Current-H curls need only the low faces with both other axes owned.
__device__ __forceinline__ void PureCoordinate(int i, int& z, int& y, int& x) {
  if (i < BY * BX) {
    z = 0; y = 1 + i / BX; x = 1 + i % BX;
  } else if ((i -= BY * BX) < BZ * BX) {
    z = 1 + i / BX; y = 0; x = 1 + i % BX;
  } else {
    i -= BZ * BX;
    z = 1 + i / BY; y = 1 + i % BY; x = 0;
  }
}

template<int Steps>
__device__ __forceinline__ float Incoming(
    const float* initial, const float* trail, const size_t* bases,
    Geometry g, int version, int component, int oz, int oy, int ox,
    int z, int y, int x, int* bad) {
  const int uz = oz + z - 1, uy = oy + y - 1, ux = ox + x - 1;
  if (!Valid(component, uz - version, uy - version, ux - version, g.n)) return 0.f;
  if (!version) return GlobalAt(initial, component, uz, uy, ux, g.n);
  const int neighbor = (z == 0) | ((y == 0) << 1) | ((x == 0) << 2);
  const int surface = Surface(z ? z - 1 : BZ - 1,
                              y ? y - 1 : BY - 1, x ? x - 1 : BX - 1);
  const size_t base = bases[neighbor];
  if (!neighbor || surface < 0 || base == static_cast<size_t>(-1)) {
    atomicAdd(bad, 1);
    return __int_as_float(0x7fffffff);
  }
  return trail[base + ((version - 1) * 6 + component) * S + surface];
}
}  // namespace moving_face_cache

template<int Steps>
__global__ __launch_bounds__(256, MOVING_MIN_CTAS) void FaceCachedMoving(
    const int* tiles, Geometry g, const float* initial, float* output,
    float* trail, int* bad) {
  using namespace moving_face_cache;
  static_assert(THREADS == 256 && V > 0 && V <= 1024,
                "Face-cache prototype requires 256 threads / at most 1024 cells");
  static_assert(6 * PAD * sizeof(float) + 8 * sizeof(size_t) <= 48 * 1024,
                "Prototype uses static shared storage within 48 KiB");
  __shared__ float state[6 * PAD];
  __shared__ size_t incoming[8];
  const int tile = tiles[blockIdx.x];
  const int bz = tile / (g.nx * g.ny), by = (tile / g.nx) % g.ny, bx = tile % g.nx;
  const int oz = bz * BZ, oy = by * BY, ox = bx * BX;
#if MOVING_TRAIL_RING
  if (!threadIdx.x) g.owner_tags[TrailOwner(g, tile, bz + by + bx)] = tile;
  __syncthreads();
#endif
  if (threadIdx.x < 8) {
    const int bits = threadIdx.x;
    const int dz = bits & 1, dy = (bits >> 1) & 1, dx = (bits >> 2) & 1;
    size_t base = static_cast<size_t>(-1);
    if (bz >= dz && by >= dy && bx >= dx) {
      const int producer = tile - dz * g.nx * g.ny - dy * g.nx - dx;
      const int slot = TrailOwner(g, producer, bz + by + bx - dz - dy - dx);
#if MOVING_TRAIL_RING
      if (bits && g.owner_tags[slot] != producer) atomicAdd(bad, 1);
      else
#endif
        base = static_cast<size_t>(slot) * Steps * 6 * S;
    }
    incoming[bits] = base;
  }
  // Version zero is initialized once. Later steps retain all owned fields.
  for (int i = threadIdx.x; i < 6 * V; i += 256) {
    const int c = i / V, q = i % V;
    const int z = q / (BY * BX), y = (q / BX) % BY, x = q % BX;
    state[c * PAD + Pad(z + 1, y + 1, x + 1)] =
        GlobalAt(initial, c, oz + z, oy + y, ox + x, g.n);
  }
  __syncthreads();

  float old_h[3][OWNED_LANES], old_e[3][OWNED_LANES];
#pragma unroll 1
  for (int n = 1; n <= Steps; ++n) {
    // Cache old boundary values once, instead of once per derivative operand.
    // These low slots are disjoint from the preceding outgoing publications.
    for (int i = threadIdx.x; i < 6 * LOW; i += 256) {
      const int c = i / LOW;
      int z, y, x;
      LowCoordinate(i % LOW, z, y, x);
      state[c * PAD + Pad(z, y, x)] = Incoming<Steps>(
          initial, trail, incoming, g, n - 1, c, oz, oy, ox, z, y, x, bad);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < OWNED_LANES; ++j) {
      const int q = threadIdx.x + j * 256;
      if (q >= V) continue;
      const int z = q / (BY * BX), y = (q / BX) % BY, x = q % BX;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        old_h[c][j] = state[c * PAD + Pad(z, y, x)];
        old_e[c][j] = state[(c + 3) * PAD + Pad(z, y, x)];
      }
    }
    // A current H write may replace another thread's old own-field sample.
    __syncthreads();
#pragma unroll
    for (int j = 0; j < OWNED_LANES; ++j) {
      const int q = threadIdx.x + j * 256;
      if (q >= V) continue;
      const int z = q / (BY * BX), y = (q / BX) % BY, x = q % BX;
      auto read = [&] __device__ (int c, int, int dz, int dy, int dx) {
        return c < 3 ? old_h[c][j] : state[c * PAD + Pad(z + dz, y + dy, x + dx)];
      };
#pragma unroll
      for (int c = 0; c < 3; ++c)
        state[c * PAD + Pad(z + 1, y + 1, x + 1)] =
            Valid(c, oz + z - n, oy + y - n, ox + x - n, g.n)
                ? Generated<0>(c, read) : 0.f;
    }
    // Old H has been retired. Its halo can now hold H at the CURRENT version.
    for (int i = threadIdx.x; i < 3 * PURE_FACES; i += 256) {
      const int c = i / PURE_FACES;
      int z, y, x;
      PureCoordinate(i % PURE_FACES, z, y, x);
      state[c * PAD + Pad(z, y, x)] = Incoming<Steps>(
          initial, trail, incoming, g, n - MOVING_FACE_CACHE_BAD_H_PHASE,
          c, oz, oy, ox, z, y, x, bad);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < OWNED_LANES; ++j) {
      const int q = threadIdx.x + j * 256;
      if (q >= V) continue;
      const int z = q / (BY * BX), y = (q / BX) % BY, x = q % BX;
      auto read = [&] __device__ (int c, int, int dz, int dy, int dx) {
        return c >= 3 ? old_e[c - 3][j]
                      : state[c * PAD + Pad(z + 1 + dz, y + 1 + dy, x + 1 + dx)];
      };
#pragma unroll
      for (int c = 0; c < 3; ++c)
        state[(c + 3) * PAD + Pad(z + 1, y + 1, x + 1)] =
            Valid(c + 3, oz + z - n, oy + y - n, ox + x - n, g.n)
                ? Generated<1>(c, read) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < OWNED_LANES; ++j) {
      const int q = threadIdx.x + j * 256;
      if (q >= V) continue;
      const int z = q / (BY * BX), y = (q / BX) % BY, x = q % BX;
      const int surface = Surface(z, y, x);
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        const float value = state[c * PAD + Pad(z + 1, y + 1, x + 1)];
        if (surface >= 0) trail[incoming[0] + ((n - 1) * 6 + c) * S + surface] = value;
        if (n == Steps && Valid(c, oz + z - n, oy + y - n, ox + x - n, g.n))
          output[Offset(c, oz + z - n, oy + y - n, ox + x - n, g.n)] = value;
      }
    }
  }
}
