// Native corrections for tensor permeability and tensor-pole (hyperbolic) media.
//
// After each vacuum half-step the JAX path in beamz/simulation/anisotropic.py
// advances the pole ADE on a cropped Yee box and maps the polarization increment
// through the inverse background tensor. These kernels perform the same
// arithmetic with two launches per region and half-step:
//
//   op 0 (pole):    drive = W(E_old) [+ K(r)]; p_new = c1 p + c2 p_prev + drive;
//                   delta = sum_poles(p_new - p)
//   op 1 (inverse): field = (base or field) + sign * Inv(src)
//   op 2 (edot):    p += K(change) per pole, the implicit half of the centred
//                   E' term of CCPR poles (beamz/simulation/ccpr.py)
//
// For CCPR poles K = dt / (2 d) B and r = E^n - E^{n-1}, the previous field
// change, so op 0 applies the explicit half of K (E^{n+1} - E^{n-1}).
//
// A packed tensor acts as diagonal * own site plus cross terms on the shared
// support (primal nodes for E, cell centers for H): symmetric terms, plus
// optional antisymmetric ones (entry (r, c) = cross + skew for r < c and
// cross - skew for r > c) for non-symmetric static tensors. They are reached
// and left with
// the reciprocal averaging of beamz.constitutive.collocate_yee_component:
// center->node is width-weighted with copied ends, node->center a plain mean.
// Collocated values are recomputed on the fly, so no shared scratch is needed.
//
// Storage is canonical float32 (z, y, x). Component r (x, y, z) lies along
// storage axis 2 - r. Periodic seams, complex fields, and gyrotropic regions
// stay on the JAX path.

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include "abi_layout.h"
#include "ffi_handler.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kFamilyE = 0;
constexpr int kOpPole = 0;
constexpr int kOpInverse = 1;
constexpr int kOpEdot = 2;
constexpr int kThreads = 256;

struct Box {
  int cells[3];         // cells per storage axis (z, y, x)
  int family;           // 0 = E (shared nodes), 1 = H (shared cells)
  const float* widths[3];  // per storage axis, or nullptr when uniform

  __host__ __device__ int SiteExtent(int component, int axis) const {
    const bool along = axis == 2 - component;
    return cells[axis] + (family == kFamilyE ? !along : along);
  }
  __host__ __device__ int SharedExtent(int axis) const {
    return cells[axis] + (family == kFamilyE);
  }
  __host__ __device__ int64_t SiteSize(int component) const {
    return int64_t(SiteExtent(component, 0)) * SiteExtent(component, 1) *
           SiteExtent(component, 2);
  }
  __host__ __device__ int64_t SharedSize() const {
    return int64_t(SharedExtent(0)) * SharedExtent(1) * SharedExtent(2);
  }
  // Launch grid of (cells + 1) per axis, which covers every component's sites.
  __host__ __device__ int64_t GridSize() const {
    return int64_t(cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1);
  }
};

struct Vector3 {
  const float* data[3];
};

__device__ __forceinline__ int64_t Offset(int z, int y, int x, int ny, int nx) {
  return (int64_t(z) * ny + y) * nx + x;
}

// This thread's (z, y, x) on the launch grid; false past its end.
__device__ __forceinline__ bool GridPoint(const Box& box, int& z, int& y, int& x) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= box.GridSize()) return false;
  const int ny = box.cells[1] + 1, nx = box.cells[2] + 1;
  x = int(index % nx);
  y = int((index / nx) % ny);
  z = int(index / (int64_t(nx) * ny));
  return true;
}

__device__ __forceinline__ float SiteValue(const Box& box, const float* values,
                                           int component, int z, int y, int x) {
  return values[Offset(z, y, x, box.SiteExtent(component, 1),
                       box.SiteExtent(component, 2))];
}

// Width weight of the lower center when averaging centers m-1 and m to node m.
__device__ __forceinline__ float NodeWeight(const Box& box, int axis, int m) {
  const float* h = box.widths[axis];
  if (h == nullptr) return 0.5f;
  return h[m - 1] / (h[m - 1] + h[m]);
}

// Component ``column`` collocated from its own sites to shared position (z,y,x).
__device__ float ToShared(const Box& box, const float* values, int column, int z,
                          int y, int x) {
  const int axis = 2 - column;
  int at[3] = {z, y, x};
  const int m = at[axis];
  if (box.family == kFamilyE) {  // center -> node along the component axis
    const int count = box.cells[axis];
    if (m == 0) return SiteValue(box, values, column, at[0], at[1], at[2]);
    if (m == count) {
      at[axis] = count - 1;
      return SiteValue(box, values, column, at[0], at[1], at[2]);
    }
    const float weight = NodeWeight(box, axis, m);
    at[axis] = m - 1;
    const float low = SiteValue(box, values, column, at[0], at[1], at[2]);
    at[axis] = m;
    const float high = SiteValue(box, values, column, at[0], at[1], at[2]);
    return weight * low + (1.0f - weight) * high;
  }
  // H: node -> center along the component axis.
  const float low = SiteValue(box, values, column, at[0], at[1], at[2]);
  at[axis] = m + 1;
  const float high = SiteValue(box, values, column, at[0], at[1], at[2]);
  return 0.5f * (low + high);
}

__device__ __forceinline__ int PairIndex(int row, int column) {
  return row + column - 1;  // (x,y) -> 0, (x,z) -> 1, (y,z) -> 2
}

// sum_{col != row} entry(row, col)[node] * collocated(values_col)[node], with
// entry = cross +- skew; either array may be nullptr (zero).
__device__ float CrossAtShared(const Box& box, const float* cross,
                               const Vector3& values, int row, int z, int y,
                               int x, const float* skew = nullptr) {
  const int64_t node =
      Offset(z, y, x, box.SharedExtent(1), box.SharedExtent(2)) * 3;
  float sum = 0.0f;
  for (int column = 0; column < 3; ++column) {
    if (column == row) continue;
    const int pair = PairIndex(row, column);
    float entry = cross != nullptr ? cross[node + pair] : 0.0f;
    if (skew != nullptr)
      entry += row < column ? skew[node + pair] : -skew[node + pair];
    sum += entry * ToShared(box, values.data[column], column, z, y, x);
  }
  return sum;
}

// Shared-support cross terms averaged back to the site (z,y,x) of ``row``.
__device__ float CrossAtSite(const Box& box, const float* cross,
                             const Vector3& values, int row, int z, int y,
                             int x, const float* skew = nullptr) {
  const int axis = 2 - row;
  int at[3] = {z, y, x};
  const int m = at[axis];
  if (box.family == kFamilyE) {  // node -> center
    const float low =
        CrossAtShared(box, cross, values, row, at[0], at[1], at[2], skew);
    at[axis] = m + 1;
    const float high =
        CrossAtShared(box, cross, values, row, at[0], at[1], at[2], skew);
    return 0.5f * (low + high);
  }
  // H: center -> node with copied ends.
  const int count = box.cells[axis];
  if (m == 0)
    return CrossAtShared(box, cross, values, row, at[0], at[1], at[2], skew);
  if (m == count) {
    at[axis] = count - 1;
    return CrossAtShared(box, cross, values, row, at[0], at[1], at[2], skew);
  }
  const float weight = NodeWeight(box, axis, m);
  at[axis] = m - 1;
  const float low =
      CrossAtShared(box, cross, values, row, at[0], at[1], at[2], skew);
  at[axis] = m;
  const float high =
      CrossAtShared(box, cross, values, row, at[0], at[1], at[2], skew);
  return weight * low + (1.0f - weight) * high;
}

struct PoleLaunch {
  Box box;
  int poles;
  Vector3 old_fields;
  const float* c1;
  const float* c2;
  const float* diagonal[3];  // (poles, *site)
  const float* cross;        // (poles, *shared, 3) or nullptr
  float* current[3];         // (poles, *site), updated in place
  float* previous[3];
  float* delta[3];           // (*site)
  Vector3 edot_change;              // (*site) previous change r, CCPR only
  const float* edot_diagonal[3];    // (poles, *site) or nullptr
  const float* edot_cross;          // (poles, *shared, 3) or nullptr
};

__global__ void PoleKernel(PoleLaunch launch) {
  const Box& box = launch.box;
  int z, y, x;
  if (!GridPoint(box, z, y, x)) return;
  const int64_t shared_size = box.SharedSize();
  for (int row = 0; row < 3; ++row) {
    const int ez = box.SiteExtent(row, 0), ey = box.SiteExtent(row, 1),
              ex = box.SiteExtent(row, 2);
    if (z >= ez || y >= ey || x >= ex) continue;
    const int64_t site = Offset(z, y, x, ey, ex);
    const int64_t site_size = box.SiteSize(row);
    const float field = launch.old_fields.data[row][site];
    float delta = 0.0f;
    for (int p = 0; p < launch.poles; ++p) {
      const int64_t state = p * site_size + site;
      float drive = launch.diagonal[row][state] * field;
      if (launch.cross != nullptr)
        drive += CrossAtSite(box, launch.cross + p * shared_size * 3,
                             launch.old_fields, row, z, y, x);
      if (launch.edot_diagonal[row] != nullptr) {
        float explicit_drive =
            launch.edot_diagonal[row][state] * launch.edot_change.data[row][site];
        if (launch.edot_cross != nullptr)
          explicit_drive += CrossAtSite(box, launch.edot_cross + p * shared_size * 3,
                                        launch.edot_change, row, z, y, x);
        drive += explicit_drive;
      }
      const float current = launch.current[row][state];
      const float next = launch.c1[p] * current +
                         launch.c2[p] * launch.previous[row][state] + drive;
      launch.previous[row][state] = current;
      launch.current[row][state] = next;
      delta += next - current;
    }
    launch.delta[row][site] = delta;
  }
}

struct EdotLaunch {
  Box box;
  int poles;
  Vector3 change;            // (*site)
  const float* diagonal[3];  // (poles, *site)
  const float* cross;        // (poles, *shared, 3) or nullptr
  float* current[3];         // (poles, *site), updated in place
};

__global__ void EdotKernel(EdotLaunch launch) {
  const Box& box = launch.box;
  int z, y, x;
  if (!GridPoint(box, z, y, x)) return;
  const int64_t shared_size = box.SharedSize();
  for (int row = 0; row < 3; ++row) {
    const int ez = box.SiteExtent(row, 0), ey = box.SiteExtent(row, 1),
              ex = box.SiteExtent(row, 2);
    if (z >= ez || y >= ey || x >= ex) continue;
    const int64_t site = Offset(z, y, x, ey, ex);
    const int64_t site_size = box.SiteSize(row);
    const float field = launch.change.data[row][site];
    for (int p = 0; p < launch.poles; ++p) {
      const int64_t state = p * site_size + site;
      float drive = launch.diagonal[row][state] * field;
      if (launch.cross != nullptr)
        drive += CrossAtSite(box, launch.cross + p * shared_size * 3,
                             launch.change, row, z, y, x);
      launch.current[row][state] += drive;
    }
  }
}

struct InverseLaunch {
  Box box;
  int origin[3];
  int field_dims[3][3];  // full field extents per component
  float* fields[3];      // full arrays, updated in place on the box
  Vector3 source;        // (*site)
  const float* base[3];  // (*site) or nullptr to start from the field
  const float* diagonal[3];
  const float* cross;    // (*shared, 3) or nullptr
  const float* skew;     // (*shared, 3) antisymmetric cross terms or nullptr
  float sign;
};

__global__ void InverseKernel(InverseLaunch launch) {
  const Box& box = launch.box;
  int z, y, x;
  if (!GridPoint(box, z, y, x)) return;
  for (int row = 0; row < 3; ++row) {
    const int ez = box.SiteExtent(row, 0), ey = box.SiteExtent(row, 1),
              ex = box.SiteExtent(row, 2);
    if (z >= ez || y >= ey || x >= ex) continue;
    const int64_t site = Offset(z, y, x, ey, ex);
    float correction = launch.diagonal[row][site] * launch.source.data[row][site];
    if (launch.cross != nullptr || launch.skew != nullptr)
      correction += CrossAtSite(box, launch.cross, launch.source, row, z, y, x,
                                launch.skew);
    const int64_t target =
        Offset(launch.origin[0] + z, launch.origin[1] + y, launch.origin[2] + x,
               launch.field_dims[row][1], launch.field_dims[row][2]);
    const float start = launch.base[row] != nullptr ? launch.base[row][site]
                                                    : launch.fields[row][target];
    launch.fields[row][target] = start + launch.sign * correction;
  }
}

// --------------------------------------------------------------------------
// FFI decoding

ffi::Error F32(const ffi::AnyBuffer& buffer, int64_t expected, const float** out,
               const char* name, bool optional = false) {
  if (buffer.element_type() != ffi::DataType::F32)
    return ffi::Error::InvalidArgument(std::string(name) + " must be float32");
  const int64_t count = static_cast<int64_t>(buffer.element_count());
  if (optional && count == 0) {
    *out = nullptr;
    return ffi::Error::Success();
  }
  if (count != expected)
    return ffi::Error::InvalidArgument(std::string(name) + " has " +
                                       std::to_string(count) + " elements, expected " +
                                       std::to_string(expected));
  *out = static_cast<const float*>(buffer.untyped_data());
  return ffi::Error::Success();
}

#define BEAMZ_TRY(expression)                       \
  do {                                              \
    if (auto error = (expression); error.failure()) \
      return error;                                 \
  } while (0)

ffi::Error Arg(const ffi::RemainingArgs& args, size_t index, int64_t expected,
               const float** out, const char* name, bool optional = false) {
  auto buffer = args.get<ffi::AnyBuffer>(index);
  if (!buffer) return buffer.error();
  return F32(*buffer, expected, out, name, optional);
}

ffi::Error Ret(const ffi::RemainingRets& rets, size_t index, int64_t expected,
               float** out, const char* name) {
  auto buffer = rets.get<ffi::AnyBuffer>(index);
  if (!buffer) return buffer.error();
  const float* data = nullptr;
  BEAMZ_TRY(F32(**buffer, expected, &data, name));
  *out = const_cast<float*>(data);
  return ffi::Error::Success();
}

ffi::Error DecodeWidths(const ffi::RemainingArgs& args, size_t first, Box* box) {
  for (int axis = 0; axis < 3; ++axis) {
    BEAMZ_TRY(Arg(args, first + axis, box->cells[axis], &box->widths[axis],
                  "cell widths", /*optional=*/true));
  }
  return ffi::Error::Success();
}

int64_t Blocks(const Box& box) { return (box.GridSize() + kThreads - 1) / kThreads; }

ffi::Error Launched() {
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(std::string("BeamZ tensor-media launch failed: ") +
                                    cudaGetErrorString(error));
}

// args: old[3], widths[3], c1, c2, diagonal[3], cross, current[3], previous[3],
//       edot change[3], edot diagonal[3], edot cross (the last seven empty
//       unless CCPR; edot cross also empty when uncoupled)
// rets: current[3], previous[3], delta[3]
ffi::Error PoleOp(cudaStream_t stream, const ffi::RemainingArgs& args,
                  const ffi::RemainingRets& rets, const Box& shape, int poles,
                  bool coupled) {
  if (args.size() != 25 || rets.size() != 9)
    return ffi::Error::InvalidArgument("tensor-media pole op expects 25 args, 9 rets");
  PoleLaunch launch{};
  launch.box = shape;
  launch.poles = poles;
  BEAMZ_TRY(DecodeWidths(args, 3, &launch.box));
  const Box& box = launch.box;
  for (int r = 0; r < 3; ++r) {
    const int64_t site = box.SiteSize(r);
    BEAMZ_TRY(Arg(args, r, site, &launch.old_fields.data[r], "old field"));
    BEAMZ_TRY(Arg(args, 8 + r, poles * site, &launch.diagonal[r], "pole diagonal"));
    BEAMZ_TRY(Ret(rets, r, poles * site, &launch.current[r], "current"));
    BEAMZ_TRY(Ret(rets, 3 + r, poles * site, &launch.previous[r], "previous"));
    BEAMZ_TRY(Ret(rets, 6 + r, site, &launch.delta[r], "delta"));
    const float* unused = nullptr;
    BEAMZ_TRY(Arg(args, 12 + r, poles * site, &unused, "current"));
    BEAMZ_TRY(Arg(args, 15 + r, poles * site, &unused, "previous"));
    BEAMZ_TRY(Arg(args, 18 + r, site, &launch.edot_change.data[r],
                  "edot change", /*optional=*/true));
    BEAMZ_TRY(Arg(args, 21 + r, poles * site, &launch.edot_diagonal[r],
                  "edot diagonal", /*optional=*/true));
    if ((launch.edot_change.data[r] == nullptr) !=
        (launch.edot_diagonal[r] == nullptr))
      return ffi::Error::InvalidArgument("edot change and diagonal must pair");
  }
  BEAMZ_TRY(Arg(args, 24, poles * box.SharedSize() * 3, &launch.edot_cross,
                "edot cross terms", /*optional=*/true));
  BEAMZ_TRY(Arg(args, 6, poles, &launch.c1, "c1"));
  BEAMZ_TRY(Arg(args, 7, poles, &launch.c2, "c2"));
  BEAMZ_TRY(Arg(args, 11, poles * box.SharedSize() * 3, &launch.cross,
                "pole cross terms", /*optional=*/!coupled));
  if (coupled && launch.cross == nullptr)
    return ffi::Error::InvalidArgument("coupled pole region lacks cross terms");
  PoleKernel<<<Blocks(box), kThreads, 0, stream>>>(launch);
  return Launched();
}

// args: change[3], widths[3], diagonal[3], cross, current[3]
// rets: current[3]
ffi::Error EdotOp(cudaStream_t stream, const ffi::RemainingArgs& args,
                  const ffi::RemainingRets& rets, const Box& shape, int poles,
                  bool coupled) {
  if (args.size() != 13 || rets.size() != 3)
    return ffi::Error::InvalidArgument("tensor-media edot op expects 13 args, 3 rets");
  EdotLaunch launch{};
  launch.box = shape;
  launch.poles = poles;
  BEAMZ_TRY(DecodeWidths(args, 3, &launch.box));
  const Box& box = launch.box;
  for (int r = 0; r < 3; ++r) {
    const int64_t site = box.SiteSize(r);
    BEAMZ_TRY(Arg(args, r, site, &launch.change.data[r], "field change"));
    BEAMZ_TRY(Arg(args, 6 + r, poles * site, &launch.diagonal[r], "edot diagonal"));
    BEAMZ_TRY(Ret(rets, r, poles * site, &launch.current[r], "current"));
    const float* unused = nullptr;
    BEAMZ_TRY(Arg(args, 10 + r, poles * site, &unused, "current"));
  }
  BEAMZ_TRY(Arg(args, 9, poles * box.SharedSize() * 3, &launch.cross,
                "edot cross terms", /*optional=*/!coupled));
  if (coupled && launch.cross == nullptr)
    return ffi::Error::InvalidArgument("coupled edot region lacks cross terms");
  EdotKernel<<<Blocks(box), kThreads, 0, stream>>>(launch);
  return Launched();
}

// args: field[3], source[3], base[3], widths[3], diagonal[3], cross, skew
// rets: field[3]
ffi::Error InverseOp(cudaStream_t stream, const ffi::RemainingArgs& args,
                     const ffi::RemainingRets& rets, const Box& shape,
                     const int origin[3], bool coupled, float sign) {
  if (args.size() != 17 || rets.size() != 3)
    return ffi::Error::InvalidArgument("tensor-media inverse op expects 17 args, 3 rets");
  InverseLaunch launch{};
  launch.box = shape;
  launch.sign = sign;
  for (int a = 0; a < 3; ++a) launch.origin[a] = origin[a];
  BEAMZ_TRY(DecodeWidths(args, 9, &launch.box));
  const Box& box = launch.box;
  for (int r = 0; r < 3; ++r) {
    auto field = rets.get<ffi::AnyBuffer>(r);
    if (!field) return field.error();
    const auto dims = (*field)->dimensions();
    if (dims.size() != 3 || (*field)->element_type() != ffi::DataType::F32)
      return ffi::Error::InvalidArgument("tensor-media fields must be rank-3 float32");
    for (int a = 0; a < 3; ++a) {
      launch.field_dims[r][a] = static_cast<int>(dims[a]);
      if (origin[a] < 0 || origin[a] + box.SiteExtent(r, a) > dims[a])
        return ffi::Error::InvalidArgument("tensor-media box exceeds the field");
    }
    launch.fields[r] = static_cast<float*>((*field)->untyped_data());
    const int64_t site = box.SiteSize(r);
    BEAMZ_TRY(Arg(args, 3 + r, site, &launch.source.data[r], "source"));
    BEAMZ_TRY(Arg(args, 6 + r, site, &launch.base[r], "base", /*optional=*/true));
    BEAMZ_TRY(Arg(args, 12 + r, site, &launch.diagonal[r], "inverse diagonal"));
  }
  BEAMZ_TRY(Arg(args, 15, box.SharedSize() * 3, &launch.cross,
                "inverse cross terms", /*optional=*/!coupled));
  if (coupled && launch.cross == nullptr)
    return ffi::Error::InvalidArgument("coupled inverse lacks cross terms");
  BEAMZ_TRY(Arg(args, 16, box.SharedSize() * 3, &launch.skew,
                "inverse antisymmetric cross terms", /*optional=*/true));
  InverseKernel<<<Blocks(box), kThreads, 0, stream>>>(launch);
  return Launched();
}

ffi::Error TensorMediaHandler(cudaStream_t stream, ffi::RemainingArgs args,
                              ffi::RemainingRets rets, int32_t abi_version,
                              int32_t op, int32_t family, int32_t coupled,
                              int32_t poles, int32_t cells_z, int32_t cells_y,
                              int32_t cells_x, int32_t origin_z, int32_t origin_y,
                              int32_t origin_x, float sign) {
  if (abi_version != beamz::cuda::abi::kAbiVersion)
    return ffi::Error::InvalidArgument("tensor-media ABI mismatch");
  if ((family != 0 && family != 1) || cells_z < 1 || cells_y < 1 || cells_x < 1)
    return ffi::Error::InvalidArgument("invalid tensor-media box");
  Box box{};
  box.cells[0] = cells_z;
  box.cells[1] = cells_y;
  box.cells[2] = cells_x;
  box.family = family;
  if (op == kOpPole) {
    if (poles < 1) return ffi::Error::InvalidArgument("pole count must be positive");
    return PoleOp(stream, args, rets, box, poles, coupled != 0);
  }
  if (op == kOpInverse) {
    const int origin[3] = {origin_z, origin_y, origin_x};
    return InverseOp(stream, args, rets, box, origin, coupled != 0, sign);
  }
  if (op == kOpEdot) {
    if (poles < 1) return ffi::Error::InvalidArgument("pole count must be positive");
    return EdotOp(stream, args, rets, box, poles, coupled != 0);
  }
  return ffi::Error::InvalidArgument("unknown tensor-media op");
}

}  // namespace

// Plain kernel launches on XLA's stream; capture-compatible like the phases.
XLA_FFI_DEFINE_HANDLER_SYMBOL(beamz_cuda_tensor_media, TensorMediaHandler,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attr<int32_t>("abi_version")
                                  .Attr<int32_t>("op")
                                  .Attr<int32_t>("family")
                                  .Attr<int32_t>("coupled")
                                  .Attr<int32_t>("poles")
                                  .Attr<int32_t>("cells_z")
                                  .Attr<int32_t>("cells_y")
                                  .Attr<int32_t>("cells_x")
                                  .Attr<int32_t>("origin_z")
                                  .Attr<int32_t>("origin_y")
                                  .Attr<int32_t>("origin_x")
                                  .Attr<float>("sign"),
                              {ffi::Traits::kCmdBufferCompatible});
