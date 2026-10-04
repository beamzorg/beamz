// Included inside update.cu's implementation namespace. Two complete Yee steps
// share three rolling, two-plane stages: H1, E1, H2. E2 is written directly.
// The original field bank stays frozen until all blocks have completed.
struct PairFieldOutputs {
  BeamzBuffer fields[6];
};

template <int X, int Y, int K> struct PairStage {
  static constexpr int sx = X + 3 - K, sy = Y + 3 - K;
  static constexpr int offset = [] {
    int value = 0;
    for (int k = 0; k < K; ++k)
      value += 6 * (X + 3 - k) * (Y + 3 - k);
    return value;
  }();
  __device__ static __forceinline__ float &At(float *shared, int c, int z,
                                              int y, int x) {
    return shared[offset + ((c * 2 + (z & 1)) * sy + y) * sx + x];
  }
};

template <bool Packed, int X, int Y, int Z, int K = 0, bool PublishCoupling = true>
__device__ __forceinline__ void
PairStages(float *shared, const FusedYeePhase &h, const FusedYeePhase &e,
           const PairFieldOutputs &final, const FusedHSources &hs,
           const FusedHSources &es, const BeamzBuffer &publication,
           const bool *active_sources, int ox, int oy, int oz, int wave,
           int low, int hz, int hy, int hx) {
  using Here = PairStage<X, Y, K>;
  constexpr int phase = K % 2, shift = phase == 1 ? 1 : 0;
  constexpr int sign = phase == 0 ? 1 : -1;
  constexpr int a_source[] = {2, 0, 1}, b_source[] = {1, 2, 0};
  constexpr int a_axis[] = {1, 0, 2}, b_axis[] = {0, 2, 1};
  const auto &launch = phase == 0 ? h : e;
  const auto &sources = phase == 0 ? hs : es;
  const int z = oz + wave - 2 - K / 2;
  if (wave >= K) {
    for (int i = threadIdx.x; i < Here::sx * Here::sy; i += blockDim.x) {
      const int ix = i % Here::sx, iy = i / Here::sx;
      const int x = ox - 2 + (K + 1) / 2 + ix;
      const int y = oy - 2 + (K + 1) / 2 + iy;
      const bool valid =
          x >= low && y >= low && z >= low && x < hx && y < hy && z < hz;
      const bool owned = x >= ox && y >= oy && z >= oz && x < ox + X &&
                         y < oy + Y && z < oz + Z && x < hx - 2 && y < hy - 2 &&
                         z < hz - 2;
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        float next = 0.f;
        if (valid) {
          float old;
          if constexpr (K < 2)
            old = Read3D(launch.inputs[c], z, y, x);
          else
            old = PairStage<X, Y, K - 2>::At(shared, c, z, iy + 1, ix + 1);
          auto other = [&](int component, int dz, int dy, int dx) {
            if constexpr (K == 0)
              return Read3D(h.inputs[3 + component], z + dz, y + dy, x + dx);
            else
              return PairStage<X, Y, K - 1>::At(
                  shared, component, z + dz, iy + shift + dy, ix + shift + dx);
          };
          auto diff = [&](int component, int axis) {
            const float center = other(component, 0, 0, 0);
            const float adjacent =
                other(component, axis == 0 ? sign : 0, axis == 1 ? sign : 0,
                      axis == 2 ? sign : 0);
            return beamz::cuda::yee::ScaleYeeDifference(
                phase == 0 ? adjacent - center : center - adjacent,
                launch.inv_resolution);
          };
          const float curl =
              diff(a_source[c], a_axis[c]) - diff(b_source[c], b_axis[c]);
          const auto &out = launch.outputs[c];
          const int logical = (z * static_cast<int>(out.dims[1]) + y) *
                                  static_cast<int>(out.dims[2]) +
                              x;
          const float decay =
              phase == 1 && Packed ? 1.f : Read(launch.inputs[6 + c], z, y, x);
          const float coefficient =
              phase == 1 && Packed
                  ? beamz::cuda::yee::PackedMaterialSource(
                        launch.inputs[6 + c], launch.inputs[9 + c], logical)
                  : Read(launch.inputs[9 + c], z, y, x);
          next = beamz::cuda::yee::AdvanceYeeField(phase, old, decay,
                                                   coefficient, curl);
          if (active_sources[phase * 3 + c]) {
            next = c == 0   ? AddFusedHSource<0>(next, sources, z, y, x, K / 2)
                   : c == 1 ? AddFusedHSource<1>(next, sources, z, y, x, K / 2)
                            : AddFusedHSource<2>(next, sources, z, y, x, K / 2);
          }
        }
        if constexpr (K == 3) {
          if (owned) {
            const auto &eh = final.fields[3 + c];
            const auto &hh = final.fields[c];
            static_cast<float *>(eh.data)[BeamzOffset3D(eh, z, y, x)] = next;
            static_cast<float *>(hh.data)[BeamzOffset3D(hh, z, y, x)] =
                PairStage<X, Y, 2>::At(shared, c, z, iy + 1, ix + 1);
          }
        } else {
          Here::At(shared, c, z, iy, ix) = next;
          if constexpr (K < 2) {
            // The second boundary substep consumes two layers of the deep
            // core. Other intermediate fields are needed only by monitors.
            const bool coupling = PublishCoupling &&
                (x < low + 4 || y < low + 4 || z < low + 4 ||
                 x >= hx - 4 || y >= hy - 4 || z >= hz - 4);
            const bool observed =
                owned && publication.data &&
                static_cast<const int *>(publication.data)[BeamzOffset3D(
                    publication, z, y / 8, x / 16)] != 0;
            if (owned && (coupling || observed)) {
              const auto &out = launch.outputs[c];
              static_cast<float *>(out.data)[BeamzOffset3D(out, z, y, x)] =
                  next;
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if constexpr (K < 3)
    PairStages<Packed, X, Y, Z, K + 1, PublishCoupling>(shared, h, e, final, hs, es, publication,
                                       active_sources, ox, oy, oz, wave, low,
                                       hz, hy, hx);
}

template <bool Packed, int X, int Y, int Z>
__global__ __launch_bounds__(256) void TemporalPairCore(
    const __grid_constant__ FusedYeePhase h,
    const __grid_constant__ FusedYeePhase e,
    const __grid_constant__ PairFieldOutputs final,
    const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,
    const __grid_constant__ BeamzBuffer publication, int low, int hz, int hy,
    int hx) {
  extern __shared__ float shared[];
  const int ox = low + 2 + blockIdx.x * X;
  const int oy = low + 2 + blockIdx.y * Y;
  const int oz = low + 2 + blockIdx.z * Z;
  const int depth = min(Z, hz - 2 - oz);
  // Most blocks do not intersect any source. Determine that once per block,
  // rather than rereading slab metadata in every stage and halo cell.
  __shared__ bool active_sources[6];
  if (threadIdx.x < 6) {
    const int c = threadIdx.x;
    const auto &group = c < 3 ? hs.groups[c] : es.groups[c - 3];
    const auto *starts = static_cast<const int *>(group.starts.data);
    bool active = false;
    for (int source = 0; source < group.coefficients.dims[0]; ++source) {
      active |= starts[3 * source] < oz + depth + 2 &&
                starts[3 * source] + group.coefficients.dims[1] > oz - 2 &&
                starts[3 * source + 1] < oy + Y + 2 &&
                starts[3 * source + 1] + group.coefficients.dims[2] > oy - 2 &&
                starts[3 * source + 2] < ox + X + 2 &&
                starts[3 * source + 2] + group.coefficients.dims[3] > ox - 2;
    }
    active_sources[c] = active;
  }
  __syncthreads();
  for (int wave = 0; wave < depth + 3; ++wave)
    PairStages<Packed, X, Y, Z>(shared, h, e, final, hs, es, publication,
                                active_sources, ox, oy, oz, wave, low, hz, hy,
                                hx);
}

// A compact queue visits exactly the two-cell coupling band, avoiding full
// interior tiles whose outputs would mostly be discarded. H neighbors are
// recomputed from frozen inputs so blocks never depend on one another.
template <bool Packed>
__global__ void TemporalPairBand(FusedYeePhase h, FusedYeePhase e,
                                 FusedHSources sources, int low, int hz, int hy,
                                 int hx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int nx = hx - low, ny = hy - low, nz = hz - low;
  const int z_faces = 4 * ny * nx;
  const int y_faces = (nz - 4) * 4 * nx;
  const int x_faces = (nz - 4) * (ny - 4) * 4;
  if (i >= z_faces + y_faces + x_faces)
    return;
  int x, y, z;
  if (i < z_faces) {
    x = i % nx;
    y = (i / nx) % ny;
    z = i / (nx * ny);
    if (z >= 2)
      z += nz - 4;
  } else if ((i -= z_faces) < y_faces) {
    x = i % nx;
    y = (i / nx) % 4;
    z = i / (nx * 4) + 2;
    if (y >= 2)
      y += ny - 4;
  } else {
    i -= y_faces;
    x = i % 4;
    y = (i / 4) % (ny - 4) + 2;
    z = i / (4 * (ny - 4)) + 2;
    if (x >= 2)
      x += nx - 4;
  }
  x += low;
  y += low;
  z += low;
  const float hv[] = {CoreHValue<0>(h, sources, z, y, x, low, hz, hy, hx),
                      CoreHValue<1>(h, sources, z, y, x, low, hz, hy, hx),
                      CoreHValue<2>(h, sources, z, y, x, low, hz, hy, hx)};
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    constexpr int sa[] = {2, 0, 1}, sb[] = {1, 2, 0};
    constexpr int aa[] = {1, 0, 2}, ab[] = {0, 2, 1};
    auto diff = [&](int component, int axis) {
      const int nz = z - (axis == 0), ny = y - (axis == 1),
                nx = x - (axis == 2);
      const float adjacent =
          component == 0
              ? CoreHValue<0>(h, sources, nz, ny, nx, low, hz, hy, hx)
          : component == 1
              ? CoreHValue<1>(h, sources, nz, ny, nx, low, hz, hy, hx)
              : CoreHValue<2>(h, sources, nz, ny, nx, low, hz, hy, hx);
      return beamz::cuda::yee::ScaleYeeDifference(
          hv[component] - adjacent, e.inv_resolution);
    };
    const float curl = diff(sa[c], aa[c]) - diff(sb[c], ab[c]);
    const auto &hout = h.outputs[c];
    const auto &eout = e.outputs[c];
    const int logical = (z * static_cast<int>(eout.dims[1]) + y) *
                            static_cast<int>(eout.dims[2]) +
                        x;
    const float decay = Packed ? 1.f : Read(e.inputs[6 + c], z, y, x);
    const float coefficient =
        Packed ? beamz::cuda::yee::PackedMaterialSource(
                     e.inputs[6 + c], e.inputs[9 + c], logical)
               : Read(e.inputs[9 + c], z, y, x);
    static_cast<float *>(hout.data)[BeamzOffset3D(hout, z, y, x)] = hv[c];
    static_cast<float *>(eout.data)[BeamzOffset3D(eout, z, y, x)] =
        beamz::cuda::yee::AdvanceYeeField(1, Read3D(e.inputs[c], z, y, x),
                                          decay, coefficient, curl);
  }
}
