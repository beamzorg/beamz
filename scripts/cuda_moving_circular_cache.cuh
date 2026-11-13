// Figure-7-style physical-coordinate cache. Field-only research prototype.
// Same arithmetic, ownership, outgoing trails and rank schedule as the controls.
#pragma once
#include "cuda_moving_face_cache.cuh"

#ifndef MOVING_CIRCULAR_BAD_IMPORT_PHASE
#define MOVING_CIRCULAR_BAD_IMPORT_PHASE 0
#endif
#ifndef MOVING_CIRCULAR_ROLLING_HEADS
#define MOVING_CIRCULAR_ROLLING_HEADS 0
#endif

namespace moving_circular_cache {
using namespace moving_face_cache;

struct Heads { int z, y, x; };
struct Position { int own, dz, dy, dx; };
__device__ __forceinline__ int AddHead(int local, int head, int size) {
  const int value = local + head;
  return value >= size ? value - size : value;
}
__device__ __forceinline__ Heads Next(Heads h) {
  return {h.z ? h.z - 1 : PZ - 1,
          h.y ? h.y - 1 : PY - 1, h.x ? h.x - 1 : PX - 1};
}
__device__ __forceinline__ int FastSlot(int z, int y, int x, Heads h) {
  return Pad(AddHead(z, h.z, PZ), AddHead(y, h.y, PY), AddHead(x, h.x, PX));
}
template<bool Forward>
__device__ __forceinline__ Position At(int z, int y, int x, Heads h) {
  const int cz = AddHead(z, h.z, PZ), cy = AddHead(y, h.y, PY), cx = AddHead(x, h.x, PX);
  return {Pad(cz, cy, cx),
      Forward ? (cz + 1 < PZ ? PY * PX : -(PZ - 1) * PY * PX)
              : (cz ? -PY * PX : (PZ - 1) * PY * PX),
      Forward ? (cy + 1 < PY ? PX : -(PY - 1) * PX)
              : (cy ? -PX : (PY - 1) * PX),
      Forward ? (cx + 1 < PX ? 1 : -(PX - 1)) : (cx ? -1 : PX - 1)};
}

template<int Steps, int Size>
__device__ __forceinline__ int Wrap(int coordinate) {
  // Every used coordinate is >= -Steps-1. Keep C++ remainder nonnegative.
  constexpr int offset = Size * ((Steps + 1 + Size - 1) / Size);
  if constexpr ((Size & (Size - 1)) == 0)
    return static_cast<unsigned>(coordinate) & (Size - 1);
  else
    return (coordinate + offset) % Size;
}

template<int Steps>
__device__ __forceinline__ int PhysicalSlot(int z, int y, int x) {
  return Pad(Wrap<Steps, PZ>(z), Wrap<Steps, PY>(y), Wrap<Steps, PX>(x));
}
}  // namespace moving_circular_cache

template<int Steps>
__global__ __launch_bounds__(256, MOVING_MIN_CTAS) void CircularCachedMoving(
    const int* tiles, Geometry g, const float* initial, float* output,
    float* trail, int* bad) {
  using namespace moving_face_cache;
  using namespace moving_circular_cache;
  MOVING_PROFILE_DECLARE;MOVING_PROFILE_BEGIN();
  static_assert(THREADS == 256 && V > 0 && V <= 1024,
                "Circular-cache prototype requires 256 threads / at most 1024 cells");
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
  // P(0)=[a-1,a+B-1]. This includes Q(1) and its positive curl neighbors.
  for (int i = threadIdx.x; i < 6 * PAD; i += 256) {
    const int c = i / PAD, q = i % PAD;
    const int z = q / (PY * PX) - 1, y = (q / PX) % PY - 1, x = q % PX - 1;
    state[c * PAD + PhysicalSlot<Steps>(z, y, x)] =
        GlobalAt(initial, c, oz + z, oy + y, ox + x, g.n);
  }
  __syncthreads();
  MOVING_PROFILE_MARK(Init);

#if MOVING_CIRCULAR_ROLLING_HEADS
  Heads head{PZ - 1, PY - 1, PX - 1};
#endif
#pragma unroll 1
  for (int n = 1; n <= Steps; ++n) {
    MOVING_PROFILE_MARK(Loop);MOVING_PROFILE_MARK(HSetup);
    // The same physical point keeps the same cache slot. H reads only its own
    // old H and old E curls; distinct H writers never need one another's old H.
#pragma unroll
    for (int j = 0; j < OWNED_LANES; ++j) {
      const int q = threadIdx.x + j * 256;
      if (q >= V) continue;
      const int z = q / (BY * BX) - n, y = (q / BX) % BY - n, x = q % BX - n;
#if MOVING_CIRCULAR_ROLLING_HEADS
      const Position position = At<true>(q / (BY * BX), (q / BX) % BY, q % BX, head);
      auto read = [&] __device__ (int c, int, int dz, int dy, int dx) {
        return state[c * PAD + position.own + dz * position.dz + dy * position.dy + dx * position.dx];
      };
#else
      auto read = [&] __device__ (int c, int, int dz, int dy, int dx) {
        return state[c * PAD + PhysicalSlot<Steps>(z + dz, y + dy, x + dx)];
      };
#endif
#pragma unroll
      for (int c = 0; c < 3; ++c)
#if MOVING_CIRCULAR_ROLLING_HEADS
        state[c * PAD + position.own] =
#else
        state[c * PAD + PhysicalSlot<Steps>(z, y, x)] =
#endif
            Valid(c, oz + z, oy + y, ox + x, g.n) ? Generated<0>(c, read) : 0.f;
    }
    MOVING_PROFILE_MARK(HCompute);
    // Finishes H reads of old high E slots, and all previous publications,
    // before those slots are recycled for the new lower physical boundary.
    __syncthreads();
    MOVING_PROFILE_MARK(HWait);
#if MOVING_CIRCULAR_ROLLING_HEADS
    const Heads next_head = Next(head);
#endif
    for (int i = threadIdx.x; i < 6 * LOW; i += 256) {
      const int c = i / LOW;
      int z, y, x;
      LowCoordinate(i % LOW, z, y, x);
      const float value = Incoming<Steps>(
          initial, trail, incoming, g, n - MOVING_CIRCULAR_BAD_IMPORT_PHASE,
          c, oz, oy, ox, z, y, x, bad);
#if MOVING_CIRCULAR_ROLLING_HEADS
      state[c * PAD + FastSlot(z, y, x, next_head)] = value;
#else
      state[c * PAD + PhysicalSlot<Steps>(z - n - 1, y - n - 1, x - n - 1)] = value;
#endif
    }
    MOVING_PROFILE_MARK(Import);
    __syncthreads();
    MOVING_PROFILE_MARK(ImportWait);MOVING_PROFILE_MARK(ESetup);
    // Current H now covers Q(n) and its negative curl neighbors. The imported
    // current E boundary occupies evicted high slots, disjoint from old E own.
#pragma unroll
    for (int j = 0; j < OWNED_LANES; ++j) {
      const int q = threadIdx.x + j * 256;
      if (q >= V) continue;
      const int z = q / (BY * BX) - n, y = (q / BX) % BY - n, x = q % BX - n;
#if MOVING_CIRCULAR_ROLLING_HEADS
      const Position position = At<false>(q / (BY * BX), (q / BX) % BY, q % BX, head);
      auto read = [&] __device__ (int c, int, int dz, int dy, int dx) {
        return state[c * PAD + position.own - dz * position.dz - dy * position.dy - dx * position.dx];
      };
#else
      auto read = [&] __device__ (int c, int, int dz, int dy, int dx) {
        return state[c * PAD + PhysicalSlot<Steps>(z + dz, y + dy, x + dx)];
      };
#endif
#pragma unroll
      for (int c = 0; c < 3; ++c)
#if MOVING_CIRCULAR_ROLLING_HEADS
        state[(c + 3) * PAD + position.own] =
#else
        state[(c + 3) * PAD + PhysicalSlot<Steps>(z, y, x)] =
#endif
            Valid(c + 3, oz + z, oy + y, ox + x, g.n) ? Generated<1>(c, read) : 0.f;
    }
    MOVING_PROFILE_MARK(ECompute);
    __syncthreads();
    MOVING_PROFILE_MARK(EWait);
#pragma unroll
    for (int j = 0; j < OWNED_LANES; ++j) {
      const int q = threadIdx.x + j * 256;
      if (q >= V) continue;
      const int lz = q / (BY * BX), ly = (q / BX) % BY, lx = q % BX;
      const int z = lz - n, y = ly - n, x = lx - n;
      const int surface = Surface(lz, ly, lx);
#pragma unroll
      for (int c = 0; c < 6; ++c) {
        const bool final_output = n == Steps && Valid(c, oz + z, oy + y, ox + x, g.n);
        // Only the outgoing high-face union may be read while the next H
        // phase starts. Interior reads would overlap its in-place H writes.
        if (surface < 0 && !final_output) continue;
#if MOVING_CIRCULAR_ROLLING_HEADS
        const float value = state[c * PAD + FastSlot(lz, ly, lx, head)];
#else
        const float value = state[c * PAD + PhysicalSlot<Steps>(z, y, x)];
#endif
        if (surface >= 0) trail[incoming[0] + ((n - 1) * 6 + c) * S + surface] = value;
        if (final_output)
          output[Offset(c, oz + z, oy + y, ox + x, g.n)] = value;
      }
    }
    MOVING_PROFILE_MARK(Publish);
    // Q(n+1) excludes Q(n)'s high-face publication union. Its H writes cannot
    // clobber pending publication reads; the next H barrier protects recycling.
#if MOVING_CIRCULAR_ROLLING_HEADS
    head = next_head;
#endif
  }
  MOVING_PROFILE_MARK(Loop);MOVING_PROFILE_END(g,tile,bz+by+bx,Steps);
}
