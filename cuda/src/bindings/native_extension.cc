#include <nanobind/stl/array.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../runtime/sharded/halo/packets.h"
#include "native_launch.h"
#include "native_deep.h"
#include "native_runtime.h"
#include "native_moving.h"
#include "native_macro_slab.h"
#include "temporal/rolling.cuh"
#include "host/observation/dft_work.h"
#include "native_zero_reset.inc"
#include "native_observation_rows.inc"
#include "native_observation_keys.inc"
#include "native_moving_chunks.inc"
#include "kernels.h"
#ifndef BEAMZ_CUDA_NATIVE_MOVING
#define BEAMZ_CUDA_NATIVE_MOVING 0
#endif
#include "graph.h"
cudaError_t BeamzEnqueueTemporalDft(cudaStream_t,const BeamzLaunch&,const BeamzLaunch&,const BeamzDftGroupLaunch&,int,int,int,float*,const uint64_t*,const int32_t*,const void*,int,uint64_t);
int BeamzLaunchTracedProgram(void*,const BeamzProgramLaunch&);
cudaError_t BeamzEnqueueShardFace(cudaStream_t, BeamzBuffer, BeamzBuffer, int, int);
bool BeamzValidateNativeShard(const BeamzLaunch&);
int BeamzEnqueueNativeShard(void*, const BeamzLaunch&, const int*);




cudaError_t BeamzEnqueueElectricCoefficients(cudaStream_t,const float*,float*,int,float,float);
namespace nb = nanobind;
using namespace nb::literals;
using namespace beamz::cuda::abi;
using beamz::cuda::moving::kSlabMacros;

namespace {

void Check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

int CurrentDevice() {
  int device = 0;
  Check(cudaGetDevice(&device), "read CUDA device");
  return device;
}

struct DeviceScope {
  const int previous = CurrentDevice();
  ~DeviceScope() { cudaSetDevice(previous); }
};

bool CanAccessPeer(int from, int to) {
  int supported = 0;
  Check(cudaDeviceCanAccessPeer(&supported, from, to), "query CUDA peer route");
  return supported != 0;
}

bool HasDirectExchange(int a, int b) {
  return a == b || (CanAccessPeer(a, b) && CanAccessPeer(b, a));
}

void EnablePeerPair(int a, int b) {
  // Non-peer GPU pairs still support the CUDA-staged copy route.
  if (a == b) return;
  for (const auto& [from, to] : {std::pair{a, b}, std::pair{b, a}}) {
    if (!CanAccessPeer(from, to)) continue;
    Check(cudaSetDevice(from), "select shard peer device");
    const auto status = cudaDeviceEnablePeerAccess(to, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled)
      cudaGetLastError();
    else
      Check(status, "enable shard peer access");
  }
}

struct DeviceMemory {
  void* data = nullptr;
  size_t bytes = 0;
  explicit DeviceMemory(size_t size) : bytes(size) {
    // Zero-length ABI placeholders must still have a non-null device pointer.
    Check(cudaMalloc(&data, std::max(size, size_t{1})), "cudaMalloc");
  }
  ~DeviceMemory() { if (data) cudaFree(data); }
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
};

// DLPack views retain this detached allocation, independently of the session.
struct CheckpointStorage {
  void* data = nullptr;
  int device = 0;
  size_t bytes = 0;
  std::shared_ptr<std::atomic<size_t>> live_bytes;
  ~CheckpointStorage() {
    if (!data) return;
    int previous = device;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaFree(data);
    cudaSetDevice(previous);
    live_bytes->fetch_sub(bytes, std::memory_order_relaxed);
  }
};

struct Stream {
  cudaStream_t value = nullptr;
  Stream() { Check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
  ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};

struct Event {
  cudaEvent_t value = nullptr;
  Event() { Check(cudaEventCreate(&value), "create event"); }
  ~Event() { if (value) cudaEventDestroy(value); }
};

#include "../runtime/sharded/halo/transport.inc"

struct ShardResources {
  int axis = -1, origin = 0;
  std::array<int, 2> neighbors{-2, -2};  // -1 physical wall, -2 unspecified by a raw caller
  size_t bytes = 0;
  struct Phase {
    BeamzBuffer geometry{}, faces[6]{};
    std::array<int, 27> host_geometry{};
    std::unique_ptr<DeviceMemory> geometry_storage, receive[3], send[3];
    cudaEvent_t packed = nullptr, copied = nullptr;
    ~Phase() {
      if (packed) cudaEventDestroy(packed);
      if (copied) cudaEventDestroy(copied);
    }
  } phases[2];
};

#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
struct MovingDependencyEvent {
  cudaEvent_t value=nullptr;
  MovingDependencyEvent(){Check(cudaEventCreateWithFlags(&value,cudaEventDisableTiming),"create moving dependency event");}
  ~MovingDependencyEvent(){if(value)cudaEventDestroy(value);}
};
#endif

struct CompiledZeroBuffer {
 const std::vector<int64_t>shape;const std::string dtype;size_t elements=1;
 CompiledZeroBuffer(std::vector<int64_t>dims,std::string type):shape(std::move(dims)),dtype(std::move(type)){
  if(shape.empty()||shape.size()>4||(dtype!="float32"&&dtype!="int32"&&dtype!="bfloat16"))throw std::invalid_argument("zero descriptor requires rank1..4 and FP32, int32 or BF16");
  for(int64_t n:shape){if(n<0||n>INT32_MAX||(n&&elements>size_t(INT32_MAX)/n))throw std::invalid_argument("zero descriptor offsets exceed int32");elements*=n;}
  if(shape.size()==3&&shape[1]*shape[2]>INT32_MAX)throw std::invalid_argument("zero descriptor plane stride exceeds int32");
 }
 size_t nbytes()const{return elements*(dtype=="bfloat16"?2:4);}
};

struct HostBuffer {
  nb::object array;
  nb::object dtype;
  nb::tuple shape;
  BeamzBuffer descriptor{};
  const void* data = nullptr;
  size_t bytes = 0;
  bool literal_zero=false;
};

struct NativeRecording {
  BeamzNativeRecorder plan;
  nb::object owner;
  float* history;
  int64_t capacity;
  HostBuffer host_indices;
};

struct RecordingDestination {
  nb::object owner;
  float* data;
  int64_t frames, points;
};

RecordingDestination DescribeRecordingDestination(nb::handle value,const nb::object& numpy) {
  auto owner=nb::borrow<nb::object>(value);
  if(!nb::isinstance(owner,numpy.attr("ndarray"))||
     !nb::cast<bool>(owner.attr("flags").attr("c_contiguous"))||
     !nb::cast<bool>(owner.attr("flags").attr("writeable"))||
     nb::cast<std::string>(owner.attr("dtype").attr("name"))!="float32"||
     !nb::cast<bool>(owner.attr("dtype").attr("isnative")))
    throw std::invalid_argument("native recorder history requires writable contiguous FP32");
  const auto shape=nb::cast<nb::tuple>(owner.attr("shape"));
  if(shape.size()!=2)throw std::invalid_argument("native recorder requires [frames, points] history");
  const auto interface=nb::cast<nb::dict>(owner.attr("__array_interface__"));
  const auto location=nb::cast<nb::tuple>(interface["data"]);
  return {std::move(owner),reinterpret_cast<float*>(nb::cast<uintptr_t>(location[0])),
          nb::cast<int64_t>(shape[0]),nb::cast<int64_t>(shape[1])};
}

int AttributeInt(const nb::dict& attributes, const char* key, int fallback) {
  if (!attributes.contains(key)) return fallback;
  nb::object integer = nb::steal<nb::object>(PyNumber_Index(attributes[key].ptr()));
  if (!integer.is_valid()) throw nb::python_error();
  const auto value = nb::cast<int64_t>(integer);
  if (value < INT32_MIN || value > INT32_MAX)
    throw std::invalid_argument(std::string(key) + " exceeds int32");
  return static_cast<int>(value);
}

float AttributeFloat(const nb::dict& attributes, const char* key) {
  if (!attributes.contains(key))
    throw std::invalid_argument(std::string("missing attribute ") + key);
  const double value = PyFloat_AsDouble(attributes[key].ptr());
  if (PyErr_Occurred()) throw nb::python_error();
  if (!std::isfinite(value) || value <= 0 || value > std::numeric_limits<float>::max())
    throw std::invalid_argument(std::string(key) + " must be finite and positive");
  return static_cast<float>(value);
}

BeamzNativeAttributes Attributes(const nb::dict& attributes) {
  BeamzNativeAttributes a;
  a.abi_version = AttributeInt(attributes, "abi_version", kAbiVersion);
  a.cuda_flags = AttributeInt(attributes, "cuda_flags", kBeamzGraphCache);
  a.nsteps = AttributeInt(attributes, "nsteps", 1);
  a.boundary_code = AttributeInt(attributes, "boundary_code", 0);
  a.metric_kind = AttributeInt(attributes, "metric_kind", 0);
  const int cpml = AttributeInt(attributes, "cpml_enabled", 0);
  if (cpml != 0 && cpml != 1)
    throw std::invalid_argument("cpml_enabled must be zero or one");
  a.cpml_enabled = cpml != 0;
  a.monitor_count = AttributeInt(attributes, "monitor_count", 0);
  a.coincident_source_group_mask = AttributeInt(attributes, "coincident_source_group_mask", 0);
  a.disjoint_source_group_mask = AttributeInt(attributes, "disjoint_source_group_mask", 0);
  a.graph_cache_capacity = AttributeInt(attributes, "graph_cache_capacity", 32);
  a.schedule_flags = AttributeInt(attributes, "schedule_flags", 0);
  a.dt = AttributeFloat(attributes, "dt");
  a.resolution = AttributeFloat(attributes, "resolution");
  if (a.abi_version != kAbiVersion || a.nsteps < 1 || a.metric_kind < 0 ||
      a.metric_kind > 2 || a.boundary_code < 0 || a.monitor_count < 0 ||
      a.graph_cache_capacity < 0 || a.graph_cache_capacity > 4096 ||
      a.coincident_source_group_mask < 0 || a.coincident_source_group_mask >= 512 ||
      a.disjoint_source_group_mask < 0 || a.disjoint_source_group_mask >= 512 ||
      (a.coincident_source_group_mask & a.disjoint_source_group_mask))
    throw std::invalid_argument("invalid native program attributes");
  return a;
}

HostBuffer Describe(nb::handle value, const nb::module_& numpy, bool copy) {
  if(nb::isinstance<CompiledZeroBuffer>(value)){
    const auto&zero=nb::cast<const CompiledZeroBuffer&>(value);HostBuffer result;
    result.array=nb::borrow<nb::object>(value);result.dtype=numpy.attr("dtype")(zero.dtype);
    result.shape=nb::steal<nb::tuple>(PyTuple_New(zero.shape.size()));
    result.descriptor.rank=int(zero.shape.size());
    result.descriptor.element_type=zero.dtype=="float32"?kBeamzF32:zero.dtype=="bfloat16"?kBeamzBF16:kBeamzS32;
    for(int a=0;a<4;++a)result.descriptor.dims[a]=a<int(zero.shape.size())?zero.shape[a]:1;
    for(size_t a=0;a<zero.shape.size();++a)PyTuple_SET_ITEM(result.shape.ptr(),a,PyLong_FromLongLong(zero.shape[a]));
    if(zero.shape.size()==3){result.descriptor.row_stride=int(zero.shape[2]);result.descriptor.plane_stride=int(zero.shape[1]*zero.shape[2]);}
    result.bytes=zero.nbytes();result.literal_zero=true;return result;
  }

  if (!nb::isinstance(value, numpy.attr("ndarray")))
    throw std::invalid_argument("native inputs must be NumPy ndarrays");
  HostBuffer result;
  result.array = copy ? numpy.attr("array")(value, "copy"_a=true, "order"_a="C")
                      : nb::borrow<nb::object>(value);
  if (!nb::cast<bool>(result.array.attr("flags").attr("c_contiguous")))
    throw std::invalid_argument("native inputs must be C-contiguous");
  result.dtype = result.array.attr("dtype");
  const std::string name = nb::cast<std::string>(result.dtype.attr("name"));
  if (name == "float32") result.descriptor.element_type = kBeamzF32;
  else if (name == "int32") result.descriptor.element_type = kBeamzS32;
  else if (name == "bfloat16") result.descriptor.element_type = kBeamzBF16;
  else throw std::invalid_argument("native inputs support float32, int32 and bfloat16");
  if (!nb::cast<bool>(result.dtype.attr("isnative")))
    throw std::invalid_argument("native inputs require native byte order");
  result.shape = nb::cast<nb::tuple>(result.array.attr("shape"));
  if (result.shape.size() > 4) throw std::invalid_argument("native buffer rank exceeds four");
  result.descriptor.rank = static_cast<int>(result.shape.size());
  for (int axis = 0; axis < 4; ++axis) result.descriptor.dims[axis] = 1;
  int64_t elements = 1;
  for (size_t axis = 0; axis < result.shape.size(); ++axis) {
    const auto dimension = nb::cast<int64_t>(result.shape[axis]);
    if (dimension < 0 || dimension > INT32_MAX ||
        (dimension && elements > INT32_MAX / dimension))
      throw std::invalid_argument("native buffer offsets exceed int32");
    result.descriptor.dims[axis] = dimension;
    elements *= dimension;
  }
  if (result.shape.size() == 3) {
    result.descriptor.row_stride = static_cast<int>(result.descriptor.dims[2]);
    result.descriptor.plane_stride = static_cast<int>(
        result.descriptor.dims[1] * result.descriptor.dims[2]);
  }
  const nb::dict interface = nb::cast<nb::dict>(result.array.attr("__array_interface__"));
  const nb::tuple data = nb::cast<nb::tuple>(interface["data"]);
  result.data = reinterpret_cast<const void*>(nb::cast<uintptr_t>(data[0]));
  result.bytes = nb::cast<size_t>(result.array.attr("nbytes"));
  return result;
}

nb::object CopyHostArray(const nb::module_& numpy, nb::tuple shape,
                          const char* dtype, const void* source, size_t bytes) {
  nb::object result = numpy.attr("empty")(shape, "dtype"_a=dtype);
  const nb::dict interface = nb::cast<nb::dict>(result.attr("__array_interface__"));
  const nb::tuple data = nb::cast<nb::tuple>(interface["data"]);
  if (bytes)
    std::memcpy(reinterpret_cast<void*>(nb::cast<uintptr_t>(data[0])), source, bytes);
  return result;
}

HostBuffer DiagnosticArray(nb::handle value, const nb::module_& numpy,
                           const char* dtype) {
  if (!nb::isinstance(value, numpy.attr("ndarray")))
    throw std::invalid_argument("diagnostic inputs must be NumPy arrays");
  HostBuffer host;
  host.array = nb::borrow<nb::object>(value);
  host.dtype = host.array.attr("dtype");
  host.shape = nb::cast<nb::tuple>(host.array.attr("shape"));
  if (host.shape.size() != 1 ||
      nb::cast<std::string>(host.dtype.attr("name")) != dtype ||
      !nb::cast<bool>(host.dtype.attr("isnative")) ||
      !nb::cast<bool>(host.array.attr("flags").attr("c_contiguous")))
    throw std::invalid_argument(std::string("diagnostic arrays require contiguous native ") + dtype + " vectors");
  host.bytes = nb::cast<size_t>(host.array.attr("nbytes"));
  const auto interface = nb::cast<nb::dict>(host.array.attr("__array_interface__"));
  host.data = reinterpret_cast<const void*>(nb::cast<uintptr_t>(
      nb::cast<nb::tuple>(interface["data"])[0]));
  return host;
}

nb::dict DeepObservationPlan(nb::sequence field_shapes, nb::handle packed_indices) {
  if (nb::len(field_shapes) != 6)
    throw std::invalid_argument("deep observation plan requires six field shapes");
  const auto numpy = nb::module_::import_("numpy");
  const auto indices = Describe(packed_indices, numpy, false);
  BeamzBuffer fields[6]{};
  for (int c = 0; c < 6; ++c) {
    const auto shape = nb::borrow<nb::sequence>(field_shapes[c]);
    if (nb::len(shape) != 3) throw std::invalid_argument("field shapes must have three axes");
    fields[c].rank = 3;
    for (int axis = 0; axis < 3; ++axis)
      fields[c].dims[axis] = nb::cast<int64_t>(shape[axis]);
  }
  const auto plan = BeamzBuildNativeDeepPlan(fields, indices.descriptor,
                                              static_cast<const int32_t*>(indices.data));
  nb::dict result;
  result["keys"] = CopyHostArray(numpy, nb::make_tuple(plan.keys.size()), "int64",
                                 plan.keys.data(), plan.keys.size() * sizeof(int64_t));
  result["remapped_indices"] = CopyHostArray(numpy, indices.shape, "int32",
      plan.remapped_indices.data(), plan.remapped_indices.size() * sizeof(int32_t));
  result["publication_words"] = CopyHostArray(numpy,
      nb::make_tuple(plan.publication_words.size()), "int32",
      plan.publication_words.data(), plan.publication_words.size() * sizeof(uint32_t));
  result["field_base"] = CopyHostArray(numpy, nb::make_tuple(6), "int64",
                                        plan.field_base.data(), 6 * sizeof(int64_t));
  result["mask_shape_zyx"] = CopyHostArray(numpy, nb::make_tuple(3), "int32",
                                           plan.mask_shape.data(), 3 * sizeof(int));
  result["key_count"] = plan.keys.size();
  result["staged_bytes_d4"] = plan.keys.size() * 4 * sizeof(float);
  return result;
}


// Validate the host row table once before using it for ownership or device gathers.
void BindPackedDftGather(BeamzBuffer& indices, const HostBuffer& index_host,
                         const HostBuffer& weights, const HostBuffer& counts,
                         const HostBuffer& masks, int monitors) {
  if (indices.rank != 2) return;
  const auto& w = weights.descriptor;
  const auto& c = counts.descriptor;
  const auto& m = masks.descriptor;
  const int64_t neighbors = indices.dims[1];
  if (monitors < 1 || neighbors < 1 ||
      indices.element_type != kBeamzS32 || w.element_type != kBeamzF32 ||
      w.rank != 2 || w.dims[0] != indices.dims[0] || w.dims[1] != neighbors ||
      c.rank != 2 || c.element_type != kBeamzS32 || c.dims[0] != monitors ||
      c.dims[1] != 5 || m.rank != 2 || m.element_type != kBeamzF32 ||
      m.dims[0] != monitors || m.dims[1] != 6 ||
      !index_host.data || !counts.data || !masks.data)
    throw std::invalid_argument("invalid packed DFT gather descriptors");
  int64_t next = (int64_t(6) * monitors + neighbors - 1) / neighbors;
  if (next > indices.dims[0])
    throw std::invalid_argument("packed DFT gather omits its row header");
  const auto* rows = static_cast<const int32_t*>(index_host.data);
  const auto* sizes = static_cast<const int32_t*>(counts.data);
  const auto* active = static_cast<const float*>(masks.data);
  int maximum = 0;
  for (int i = 0; i < monitors; ++i) {
    const int points = sizes[5LL * i + 1];
    if (points < 1 || sizes[5LL * i] < 1)
      throw std::invalid_argument("packed DFT gather requires positive point and frequency counts");
    maximum = std::max(maximum, points);
    for (int axis = 0; axis < 6; ++axis) {
      const int64_t slot = 6LL * i + axis;
      const bool used = active[slot] != 0.0f;
      if (rows[slot] != (used ? next : -1))
        throw std::invalid_argument("packed DFT gather row offsets do not cover the declared points");
      if (used) next += points;
    }
  }
  if (next != indices.dims[0])
    throw std::invalid_argument("packed DFT gather row capacity differs");
  indices.row_stride = maximum;
}

// The sliding window shares the ordinary field addresses outside its kernel.
size_t RollingGuardBytes(const BeamzBuffer& buffer,size_t index,int band){
  if(!band||buffer.rank!=3||buffer.element_type!=kBeamzF32||
     !buffer.dims[0]||!buffer.dims[1]||!buffer.dims[2])return 0;
  int planes=band+4;
  if(index>=6){
    const int term=index>=kCpmlEPsiInputOffset?int(index)-kCpmlEPsiInputOffset:int(index)-kCpmlHPsiInputOffset;
    if(term<0||term>=6)return 0;
    if(beamz::cuda::moving::kCpmlAxes[term]==0)planes=int(buffer.dims[0]);
  }
  return size_t(planes)*buffer.plane_stride*sizeof(float);
}

#if BEAMZ_CUDA_NATIVE_MOVING
beamz::cuda::moving::CpmlWidths MovingCpmlWidths(const HostBuffer* host, bool cpml) {
  beamz::cuda::moving::CpmlWidths widths{};
  if(!cpml)return widths;
  for(int phase=0;phase<2;++phase){
    const auto& meta=host[phase?kCpmlEInputOffset+6:12];
    const auto& b=meta.descriptor;
    if(!meta.data||b.rank!=2||b.element_type!=kBeamzS32||b.dims[0]!=6||b.dims[1]!=5)
      throw std::invalid_argument("moving CPML metadata requires int32[6,5] host storage");
    const auto* data=static_cast<const int32_t*>(meta.data);
    for(int term=0;term<6;++term)widths[6*phase+term]={data[5*term+2],data[5*term+3]};
  }
  return widths;
}

unsigned TransparentCpmlFaces(const HostBuffer* host, int boundary_code) {
  if (boundary_code != (12 << 8)) return 0;
  unsigned faces = 63;
  for (int phase = 0; phase < 2; ++phase) {
    const int start = phase ? kCpmlEInputOffset + 7 : 13;
    for (int term = 0; term < 6; ++term) {
      const int axis = beamz::cuda::moving::kCpmlAxes[term];
      for (int kind = 0; kind < 3; ++kind) {
        const auto& profile = host[start + 3 * term + kind];
        const auto* values = static_cast<const float*>(profile.data);
        for (int side = 0; side < 2; ++side) {
          const unsigned mask = 1u << (2 * axis + side);
          if (!(faces & mask)) continue;
          if (!values || profile.descriptor.element_type != kBeamzF32 ||
              profile.bytes != 24 * sizeof(float) ||
              !std::all_of(values + 12 * side, values + 12 * (side + 1),
                           [kind](float v) { return v == float(kind == 2); }))
            faces &= ~mask;
        }
      }
    }
  }
  return faces;
}

unsigned SyntheticFaces(const std::array<int, 3>& cut) {
  return cut[0] < 0 ? 0u :
      (unsigned(cut[1] >= 0) | (unsigned(cut[2] >= 0) << 1)) << (2 * cut[0]);
}

void AppendMovingSourceCells(std::vector<beamz::cuda::moving::SourceCell>& points,
                             const BeamzBuffer* fields,int timing,int component,
                             const HostBuffer& indices) {
  if(timing<0||timing>2||component<0||component>2||!indices.data||
     indices.descriptor.rank!=1||indices.descriptor.element_type!=kBeamzS32)
    throw std::invalid_argument("invalid moving sparse source indices");
  const auto& field=fields[(timing==1?0:3)+component];
  const int64_t nx=field.dims[2],ny=field.dims[1],nz=field.dims[0];
  if(nx<1||ny<1||nz<1||nx>INT32_MAX/ny||nx*ny>INT32_MAX/nz)
    throw std::invalid_argument("moving sparse source field exceeds int32");
  const auto* cells=static_cast<const int32_t*>(indices.data);
  for(size_t i=0;i<indices.bytes/sizeof(int32_t);++i){
    const int index=cells[i];
    if(index<0||index>=nx*ny*nz)throw std::invalid_argument("moving sparse source outside its field");
    points.push_back({{int(index/(nx*ny)),int((index/nx)%ny),int(index%nx)},timing==0});
  }
}

// Include regular currents whenever an irregular current uses this certificate.
std::array<int,2> MovingSourceZSpan(const std::vector<beamz::cuda::moving::SourceCell>& points,
                               const BeamzSourceGroupLaunch* groups,const int32_t*const* starts) {
  if(points.empty())return {};
  int64_t first=INT32_MAX,end=0;
  for(const auto& cell:points){
    first=std::min(first,int64_t(cell.point[0]));
    end=std::max(end,int64_t(cell.point[0])+1);
  }
  for(int g=0;g<9;++g)for(int64_t i=0;i<groups[g].coefficients.dims[0];++i){
    const int64_t z=starts[g][3*i];
    first=std::min(first,z);end=std::max(end,z+groups[g].coefficients.dims[1]);
  }
  return {int(std::max<int64_t>(0,first)),int(std::min<int64_t>(INT32_MAX,end))};
}

void AppendMovingRecorderKeys(BeamzNativeDeepPlan& plan,const BeamzBuffer* fields,
                              int component,const HostBuffer& indices) {
  if(!indices.bytes)return;
  if(component<0||component>=6||!indices.data||indices.descriptor.rank!=2||
     indices.descriptor.element_type!=kBeamzS32)
    throw std::invalid_argument("invalid moving recorder gather descriptor");
  const auto& f=fields[component];
  const int64_t extent=f.dims[0]*f.dims[1]*f.dims[2],base=plan.field_base[component];
  const auto* data=static_cast<const int32_t*>(indices.data);
  for(size_t i=0;i<indices.bytes/sizeof(int32_t);++i){
    const int index=data[i];
    if(index<0||index>=extent)throw std::invalid_argument("moving recorder gather outside its field");
    if(plan.keys.empty()||plan.keys.back()!=base+index)plan.keys.push_back(base+index);
  }
}

void SortMovingKeys(BeamzNativeDeepPlan& plan) {
  std::sort(plan.keys.begin(),plan.keys.end());
  plan.keys.erase(std::unique(plan.keys.begin(),plan.keys.end()),plan.keys.end());
  if(plan.keys.size()>INT32_MAX)throw std::invalid_argument("moving unique observations exceed int32");
}

bool RecordVolumesAtBoundaries(double frames_per_step,int depth) {
  // Each cut can cost one temporal block, so only cut for sparse acquisitions.
  return frames_per_step>0 && frames_per_step*depth<=0.125;
}

std::vector<int32_t> MovingMonitorRows(const BeamzBuffer&indices,const BeamzBuffer&counts,const int32_t*data){
  if (indices.rank == 2) return {};
  if(indices.rank!=4||indices.element_type!=kBeamzS32||indices.dims[1]!=6||indices.dims[3]<1||
     counts.rank!=2||counts.element_type!=kBeamzS32||counts.dims[0]!=indices.dims[0]||counts.dims[1]!=5)
    throw std::invalid_argument("moving monitor counts require an int32[M,5] table matching the indices");
  if(indices.dims[0]&&!data)throw std::invalid_argument("moving monitor counts require host storage");
  std::vector<int32_t>offsets(size_t(indices.dims[0])+1,0);size_t total=0;
  for(int64_t m=0;m<indices.dims[0];++m){
    const int np=data[5*m+1];if(np<0||np>indices.dims[2])throw std::invalid_argument("moving monitor point count exceeds its index support");
    total+=size_t(np)*6*size_t(indices.dims[3]);
    if(total>size_t(INT32_MAX))throw std::invalid_argument("moving compact index offsets exceed int32");
    offsets[size_t(m)+1]=int32_t(total);
  }
  if(total==size_t(indices.dims[0])*6*size_t(indices.dims[2])*size_t(indices.dims[3]))offsets.clear();
  return offsets;
}

nb::dict MovingResourcePlan(nb::sequence inputs,int depth,int raw_slots,nb::sequence scatters,nb::sequence recorders,bool tensor,size_t budget,int minimum_overlap,int scheduling_sms,int macros,std::array<int,3> cut,int boundary_code,int rolling_band){
  if(macros<2||macros>kSlabMacros||macros%2)throw std::invalid_argument("macro slab count must be even and within the compiled limit");
  if(minimum_overlap==0)minimum_overlap=raw_slots;
  if(minimum_overlap<2||minimum_overlap>raw_slots)throw std::invalid_argument("minimum macro overlap exceeds the requested overlap range");
  if(raw_slots<2||raw_slots>16)throw std::invalid_argument("raw bank count must be2..16");
  if(!beamz::cuda::moving::ValidDepth(depth))throw std::invalid_argument("temporal depth must be a multiple of four from 4 to 32");
  if(tensor&&depth!=4)throw std::invalid_argument("tensor moving execution uses four-step blocks");
  constexpr size_t extra=12+kSourceGroupCount*kSourceGroupBufferCount+kMonitorInputCount+kMstepPlanBufferCount;
  const size_t count=nb::len(inputs);
  const bool cpml=count==kCpmlGraphInputCount+extra;
  if(!cpml&&count!=kYeeGraphInputCount+extra)throw std::invalid_argument("moving plan requires a prepared Yee packet");
  const size_t source=(cpml?kCpmlGraphInputCount:kYeeGraphInputCount)+12;
  const size_t monitor=source+kSourceGroupCount*kSourceGroupBufferCount;
  auto numpy=nb::module_::import_("numpy");std::vector<HostBuffer> host;host.reserve(count);
  std::vector<BeamzBuffer> fields;fields.reserve(count);
  for(size_t i=0;i<count;++i){host.push_back(Describe(inputs[i],numpy,false));fields.push_back(host.back().descriptor);}
  if (cut != std::array<int, 3>{-1, -1, -1}) {
    const int axis = cut[0];
    if (axis < 0 || axis > 2 || cut[1] < -1 || cut[2] < -1 ||
        (cut[1] < 0 && cut[2] < 0) || (cut[1] >= 0 && cut[2] >= 0 && cut[1] >= cut[2]))
      throw std::invalid_argument("invalid temporal resource-plan partition cut");
    for (int c = 0; c < 6; ++c)
      if (std::max(cut[1], cut[2]) > fields[c].dims[axis])
        throw std::invalid_argument("temporal resource-plan cut exceeds field extent");
  }
  const auto* counts = static_cast<const int32_t*>(host[monitor+4].data);
  const int monitors = int(fields[monitor+4].dims[0]);
  BindPackedDftGather(fields[monitor], host[monitor], host[monitor+1], host[monitor+4], host[monitor+3], monitors);
  const auto rows=MovingMonitorRows(fields[monitor],fields[monitor+4],counts);
  if(host[monitor].bytes&&!host[monitor].data)throw std::invalid_argument("moving monitor indices require host storage");
  auto obs=BeamzBuildNativeMovingKeyPlan(fields.data(),fields[monitor],static_cast<const int32_t*>(host[monitor].data),counts,monitors);
  size_t recorder_bytes=0;
  double volume_frames_per_step=0;
  size_t volume_count=0;
  for(nb::handle entry:recorders){
    const auto spec=nb::cast<nb::sequence>(entry);
    if(nb::len(spec)!=6)throw std::invalid_argument("invalid moving recorder descriptor");
    const auto indices=Describe(spec[3],numpy,false);
    AppendMovingRecorderKeys(obs,fields.data(),nb::cast<int>(spec[0]),indices);
    if(indices.bytes)recorder_bytes+=indices.bytes;
    else{
      const int interval=nb::cast<int>(spec[1]);
      if(interval<1)throw std::invalid_argument("invalid moving recorder cadence");
      volume_frames_per_step+=1.0/interval;
      ++volume_count;
    }
  }
  const bool record_volume=volume_count&&!RecordVolumesAtBoundaries(volume_frames_per_step,depth);
  if(record_volume)recorder_bytes+=volume_count*sizeof(beamz::cuda::moving::VolumeRecording);
  if(nb::len(recorders))SortMovingKeys(obs);
  const size_t index_count=rows.empty()?host[monitor].bytes/sizeof(int32_t):size_t(rows.back());
  const size_t index_width=BeamzMovingIndexWidth(BeamzDftNeighbors(fields[monitor]),obs.keys.size());
  int scatter_sweep_planes=0;
  BeamzNativeMovingPlan p;{
  BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};
  for(int g=0;g<9;++g){groups[g].coefficients=fields[source+3*g];const auto&positions=fields[source+3*g+2];starts[g]=static_cast<const int32_t*>(host[source+3*g+2].data);
    if(groups[g].coefficients.dims[0]&&(!starts[g]||positions.element_type!=kBeamzS32||positions.rank!=2||positions.dims[0]!=groups[g].coefficients.dims[0]||positions.dims[1]!=3))throw std::invalid_argument("moving source starts require int32[G,3] host storage");
  }
  std::vector<beamz::cuda::moving::SourceCell> sparse;
  for(nb::handle entry:scatters){
    const auto spec=nb::cast<nb::sequence>(entry);
    if(nb::len(spec)!=6)throw std::invalid_argument("invalid moving sparse source descriptor");
    AppendMovingSourceCells(sparse,fields.data(),nb::cast<int>(spec[0]),nb::cast<int>(spec[1]),Describe(spec[2],numpy,false));
  }
  if(rolling_band){
    const auto span=MovingSourceZSpan(sparse,groups,starts);
    scatter_sweep_planes=span[1]-span[0];
  }
  p=BeamzBuildNativeMovingPlan(fields.data(),depth,{},groups,starts,MovingCpmlWidths(host.data(),cpml),bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES),sparse,record_volume,tensor,obs.keys,cpml && cut[0]>=0 ? TransparentCpmlFaces(host.data(),boundary_code)&SyntheticFaces(cut) : 0u);
  }
  BeamzCompactNativeMovingUniformTiles(p,false);
  size_t lookup_bytes=0;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
  if(!obs.keys.empty()){
    lookup_bytes=BeamzBuildNativeMovingRowObservation<false>(fields.data(),obs.keys).device_bytes;
  }
#endif
  const size_t phase_bytes=size_t(depth)*fields[monitor+2].dims[0]*fields[monitor+2].dims[1]*12;
  const size_t scatter_bytes=nb::len(scatters)*sizeof(beamz::cuda::moving::Scatter);
  const size_t face_bytes=tensor?0:sizeof(int32_t)*beamz::cuda::moving::FaceLines(p.counts);
  const size_t field_metadata=p.metadata_bytes+face_bytes;
  // Keys, recorders, source geometry and physical tile policies are shared by
  // all legal overlaps at this depth. Recompute only the slab liveness bounds.
  BeamzNativeMovingMacroSlab slab;
  size_t observation_bytes,extra_bytes;
  for(;;--raw_slots){
    slab=BeamzBuildNativeMovingMacroSlab(p,macros,raw_slots,false,obs.keys.empty()?0:scheduling_sms,cut);
    p.metadata_bytes=field_metadata+size_t(p.epochs)*(slab.max_frontier-p.max_frontier)*sizeof(uint64_t);
    p.trail_floats=beamz::cuda::moving::TrailOwners(slab.max_frontier,tensor)*((tensor?9:6)*depth-3)*p.surface_points;
    observation_bytes=index_count*index_width+rows.size()*4+size_t(depth)*slab.observation_slots*obs.keys.size()*4+lookup_bytes;
    extra_bytes=p.metadata_bytes+p.trail_floats*4+observation_bytes+slab.device_bytes()+(p.tile_ids.size()*sizeof(int32_t)+face_bytes)*macros+phase_bytes+scatter_bytes+recorder_bytes;
    if(extra_bytes<=budget||raw_slots==minimum_overlap)break;
  }
  // Both schedules consume the same frozen material, sources and DFT maps.
  // Only their live field storage, raw observations and tile indexing differ.
  nb::dict r;
  // A rolling band may spend one additional live frontier of field values.
  // The caller always caps the total against the available device memory.
  if(scatter_sweep_planes)r["scatter_sweep_planes"]=scatter_sweep_planes;
  r["field_frontier_bytes"]=tensor?0:size_t(slab.max_frontier)*(6*depth-3)*p.surface_points*sizeof(float);
  if(rolling_band){
    if(rolling_band<32||rolling_band>512||rolling_band%32)throw std::invalid_argument("invalid CUDA rolling band");
    bool admitted=cpml&&boundary_code==(12<<8)&&!tensor&&depth%8==0;
    uint32_t reference=0;
    for(int c=0;admitted&&c<3;++c){
      const auto&decay=host[6+c];const auto&drive=host[9+c];
      admitted=decay.data&&drive.data&&!decay.descriptor.rank&&!drive.descriptor.rank&&
        decay.descriptor.element_type==kBeamzF32&&drive.descriptor.element_type==kBeamzF32&&
        *static_cast<const float*>(decay.data)==1.f&&fields[kCpmlEInputOffset+3+c].rank==2&&
        fields[kCpmlEInputOffset+3+c].element_type==kBeamzS32;
      if(admitted){uint32_t bits;std::memcpy(&bits,drive.data,4);if(c)admitted=bits==reference;else reference=bits;}
    }
    for(int i=0;admitted&&i<6;++i)admitted=fields[kCpmlHPsiInputOffset+i].element_type==kBeamzF32&&fields[kCpmlEPsiInputOffset+i].element_type==kBeamzF32;
    if(admitted){
      std::array<int,3> shape{};
      for(int axis=0;axis<3;++axis){shape[axis]=int(fields[0].dims[axis]);for(int c=1;c<6;++c)shape[axis]=std::min(shape[axis],int(fields[c].dims[axis]));}
      if(*std::min_element(shape.begin(),shape.end())>=24){
        const size_t shared_bytes=p.tile_flags.size()+sizeof(int32_t)+index_count*index_width+rows.size()*4+
            size_t(depth)*obs.keys.size()*4+lookup_bytes+phase_bytes+scatter_bytes+recorder_bytes;
        const auto bytes=[&](int band){
          size_t size=shared_bytes+beamz::cuda::rolling::MakePlan(shape,band).mask_bytes;
          for(size_t i=0;i<fields.size();++i)size+=RollingGuardBytes(fields[i],i,band);
          return size;
        };
        nb::dict alternatives;
        for(int band=32;band<rolling_band;band*=2)alternatives[nb::int_(band)]=bytes(band);
        const auto preferred=bytes(rolling_band);
        alternatives[nb::int_(rolling_band)]=preferred;
        r["rolling_extra_bytes"]=preferred;
        r["rolling_band_bytes"]=alternatives;
      }
    }
  }
  r["depth"]=depth;r["counts_zyx"]=nb::make_tuple(p.counts[0],p.counts[1],p.counts[2]);r["tile_count"]=p.tile_ids.size();r["rank_count"]=p.bulk_counts.size();r["max_frontier"]=slab.max_frontier;r["rank_frontier"]=p.max_frontier;r["raw_slots"]=raw_slots;r["observation_slots"]=obs.keys.empty()?0:slab.observation_slots;r["observation_end"]=slab.observation_end;r["macro_level_stride"]=slab.macro_level_stride;r["macro_levels"]=slab.level_count;r["epochs"]=p.epochs;r["surface_points"]=p.surface_points;r["trail_bytes"]=p.trail_floats*4;r["metadata_bytes"]=p.metadata_bytes;r["observation_bytes"]=observation_bytes;r["observation_keys"]=obs.keys.size();r["direct_lookup_bytes"]=lookup_bytes;r["extra_gpu_bytes"]=extra_bytes;r["temporal_dft_phase_bytes"]=phase_bytes;r["macro_slab_count"]=macros;
  r["monitor_index_bytes"]=index_count*index_width;r["monitor_index_offsets_bytes"]=rows.size()*4;
  std::vector<int64_t>policy_counts(6,0);
  for(size_t rank=0;rank<p.bulk_counts.size();++rank)if(!p.policy_offsets.empty())for(int policy=0;policy<6;++policy)policy_counts[policy]+=p.policy_offsets[7*rank+policy+1]-p.policy_offsets[7*rank+policy];
  r["policy_tile_counts"]=nb::cast(policy_counts);r["policy_host_bytes"]=(p.policy_offsets.size()+2*p.bulk_counts.size())*4;r["compact_noio_host_bytes"]=p.bulk_counts.size()*4;r["compact_observer_host_bytes"]=p.bulk_counts.size()*4;
  size_t bulk=0;for(int count:p.bulk_counts)bulk+=count;r["bulk_tile_count"]=bulk;return r;
}
#endif

// The prefix of every row is aligned, with its remainder packed after all
// prefixes. The logical field and its compact representation have equal size.
void CopyShardRows(const BeamzBuffer& field, void* dense, cudaMemcpyKind kind,
                   cudaStream_t stream, size_t first_row, size_t rows,
                   int source_device = -1, int destination_device = -1) {
  const size_t nx = field.dims[2], head = field.row_stride;
  const size_t total_rows = size_t(field.dims[0]) * field.dims[1];
  const size_t chunk = std::max(size_t(1), (64u << 20) / (nx * sizeof(float)));
  const bool upload = kind == cudaMemcpyHostToDevice;
  for (size_t row = 0; row < rows; row += chunk) {
    const size_t height = std::min(rows - row, chunk);
    for (int tail = 0; tail < 2; ++tail) {
      const size_t width = tail ? nx - head : head;
      if (!width) continue;
      auto* segment = static_cast<float*>(field.data) +
          (tail ? total_rows * head : 0) + (first_row + row) * width;
      auto* full = static_cast<float*>(dense) + row * nx + (tail ? head : 0);
      if (source_device != destination_device) {
        cudaMemcpy3DPeerParms p{};
        p.srcDevice = source_device;
        p.dstDevice = destination_device;
        p.srcPtr = {segment, width * 4, width * 4, height};
        p.dstPtr = {full, nx * 4, nx * 4, height};
        p.extent = {width * 4, height, 1};
        Check(cudaMemcpy3DPeerAsync(&p, stream), "copy compact peer face");
      } else {
        Check(cudaMemcpy2DAsync(upload ? segment : full, 4 * (upload ? width : nx),
                                upload ? full : segment, 4 * (upload ? nx : width),
                                4 * width, height, kind, stream), "copy compact field rows");
      }
    }
    if (kind != cudaMemcpyDeviceToDevice)
      Check(cudaStreamSynchronize(stream), "finish compact host transfer");
  }
}

void CopyShardField(const BeamzBuffer& field, void* dense, cudaMemcpyKind kind,
                    cudaStream_t stream) {
  CopyShardRows(field, dense, kind, stream, 0, size_t(field.dims[0]) * field.dims[1]);
}

class NativeSession {
 public:
  NativeSession(nb::sequence inputs, nb::dict attributes, bool preserve_inputs,
                std::string kind, bool initialize_scratch)
      : attributes_(Attributes(attributes)), preserve_(preserve_inputs),
        mstep_(kind == "mstep"), generated_(kind == "generated" || kind == "sharded"),
        shard_layout_(kind == "sharded"),
        numpy_(nb::module_::import_("numpy")) {
    moving_macros_=AttributeInt(attributes,"moving_macros",kSlabMacros);
    if(moving_macros_<2||moving_macros_>kSlabMacros||moving_macros_%2)
      throw std::invalid_argument("invalid temporal macro count");
    allow_peer_packets_=AttributeInt(attributes,"allow_peer_packets",1)!=0;
    rolling_band_=AttributeInt(attributes,"rolling_band",0);
    if(rolling_band_&&(!BEAMZ_CUDA_NATIVE_MOVING||!mstep_||!attributes_.cpml_enabled||attributes_.metric_kind||
       attributes_.boundary_code!=(12<<8)||rolling_band_<32||rolling_band_>512||rolling_band_%32))
      throw std::invalid_argument("invalid rolling CUDA storage contract");
    clamp_source_tails_ = AttributeInt(attributes, "clamp_source_tails", 0) != 0;
    (void)initialize_scratch; // Compatibility argument; unused workspace stays deferred.
    if (shard_layout_ && preserve_) throw std::invalid_argument("sharded layout requires owned fields");
    if (!mstep_ && !generated_ && kind != "program")
      throw std::invalid_argument("kind must be mstep, program, generated or sharded");
    if (mstep_ && ((attributes_.metric_kind < 0 || attributes_.metric_kind > 2) ||
        AttributeInt(attributes, "temporal_steps", 2) != 2 ||
        AttributeInt(attributes, "program_layout", 0) != 0))
      throw std::invalid_argument("mstep requires the Cartesian two-step layout");
    graph_count_ = attributes_.cpml_enabled ? kCpmlGraphInputCount : kYeeGraphInputCount;
    state_count_ = attributes_.cpml_enabled ? kCpmlGraphOutputCount : kYeeGraphOutputCount;
    const int mstep_count = graph_count_ + 12 + 27 + 16 + 5;
    const int program_count = graph_count_ + 27 + 16;
    const size_t input_count = nb::len(inputs);
    if ((mstep_ && input_count != static_cast<size_t>(mstep_count)) ||
        (!mstep_ && input_count != static_cast<size_t>(mstep_count) &&
         input_count != static_cast<size_t>(program_count)))
      throw std::invalid_argument("input count does not match the prepared native ABI");
    workspace_count_ = input_count == static_cast<size_t>(mstep_count) ? 12 : 0;
    source_offset_ = graph_count_ + workspace_count_;
    monitor_offset_ = source_offset_ + 27;
    for (int c = 0; c < 6; ++c) persistent_.insert(c);
    if (attributes_.cpml_enabled) {
      for (int c = 0; c < 6; ++c) {
        persistent_.insert(kCpmlHPsiInputOffset + c);
        persistent_.insert(kCpmlEPsiInputOffset + c);
      }
    }
    for (int c = 0; c < 3; ++c) persistent_.insert(monitor_offset_ + 7 + c);
    persistent_.insert(monitor_offset_ + 13);
    persistent_.insert(monitor_offset_ + 14);
    persistent_.insert(monitor_offset_ + 15);
    host_.reserve(input_count);
    base_inputs_.reserve(input_count);
    allocations_.reserve(input_count);
    std::vector<nb::object> original_inputs;original_inputs.reserve(input_count);
    std::map<std::string,int> immutable_inputs;
    const size_t electric=attributes_.cpml_enabled?kCpmlEInputOffset:12;

    for (size_t i = 0; i < input_count; ++i) {
      original_inputs.push_back(nb::borrow<nb::object>(inputs[i]));const auto&original_input=original_inputs.back();
      // Moving reconstructs its own plan; one-bank ordinary ignores workspace.
      const bool defer_plan=workspace_count_&&i>=size_t(monitor_offset_+16);
      const bool defer_scratch=workspace_count_&&
          ((!mstep_&&i>=size_t(graph_count_)&&i<size_t(graph_count_+6))||(i>=size_t(graph_count_+6)&&i<size_t(graph_count_+12))||defer_plan);
      if(defer_scratch)deferred_scratch_.insert(int(i));
      auto view = Describe(original_input, numpy_, false);
      if(view.literal_zero&&!(persistent_.count(i)||(workspace_count_&&i>=size_t(graph_count_)&&i<size_t(graph_count_+12))||defer_plan))throw std::invalid_argument("zero descriptors are admitted only for evolving state and workspace");

      const bool zero_initial = (persistent_.count(i) || defer_scratch) && (view.literal_zero||BeamzNativeAllPositiveZero(view.data, view.bytes));
      if (zero_initial) zero_reset_.insert(int(i));
      if(rolling_band_ && i>=electric+3 && i<electric+6){
        const auto& code=view.descriptor;
        const auto& table=base_inputs_[i-3];
        const auto& field=base_inputs_[i-electric];
        if(code.element_type!=kBeamzS32 || code.rank!=2 || code.dims[0]!=1 ||
           table.element_type!=kBeamzF32 || table.rank!=1)
          throw std::invalid_argument("rolling requires block-coded electric coefficients");
        int64_t cells=1;
        for(int axis=0;axis<3;++axis){
          if(field.dims[axis]<1||cells>INT32_MAX/field.dims[axis])
            throw std::invalid_argument("invalid rolling support");
          cells*=field.dims[axis];
        }
        if(code.dims[1]!=(cells+31)/32)
          throw std::invalid_argument("invalid rolling coefficient header count");
        BeamzValidateMovingBlockHeaders(static_cast<const int32_t*>(view.data),cells,table.dims[0]);
      }
      const bool capture=defer_scratch||(!preserve_&&persistent_.count(i)!=0)||
          (BEAMZ_CUDA_NATIVE_MOVING&&attributes_.cpml_enabled&&(i==12||i==electric+6))||
          (BEAMZ_CUDA_NATIVE_MOVING&&mstep_&&((i>=6&&i<12&&view.descriptor.rank==0)||(!rolling_band_&&i>=electric&&i<electric+6)||
           i==size_t(monitor_offset_)||i==size_t(monitor_offset_+4)||
           (i>=size_t(source_offset_)&&i<size_t(monitor_offset_)&&(i-source_offset_)%3==2)));
      const bool immutable=!persistent_.count(i)&&!defer_plan&&
          !(i>=size_t(graph_count_)&&i<size_t(graph_count_+workspace_count_))&&
          !(i>=size_t(monitor_offset_+10)&&i<size_t(monitor_offset_+13));
      if(immutable&&view.bytes){
        const uintptr_t identity=reinterpret_cast<uintptr_t>(original_input.ptr());
        std::string key(reinterpret_cast<const char*>(&identity),sizeof(identity));
        // Keep the alias classes identical between ordinary and moving plans.
        key.push_back(capture||(i>=electric&&i<electric+6));
        key.append(reinterpret_cast<const char*>(&view.bytes),sizeof(view.bytes));BeamzAppendBufferKey(&key,view.descriptor);
        const auto previous=immutable_inputs.find(key);
        if(previous!=immutable_inputs.end()){
          host_.push_back(host_[previous->second]);base_inputs_.push_back(base_inputs_[previous->second]);
          allocations_.push_back(std::make_unique<DeviceMemory>(0));++immutable_aliases_;immutable_gpu_alias_bytes_+=view.bytes;
          if(capture)immutable_host_alias_bytes_+=view.bytes;
          continue;
        }
        immutable_inputs.emplace(std::move(key),int(i));
      }
      host_.push_back(zero_initial?std::move(view):Describe(original_input,numpy_,capture));
      const bool alias_final=mstep_&&i>=size_t(graph_count_)&&i<size_t(graph_count_+6);
      auto descriptor = host_.back().descriptor;
      const size_t guard=RollingGuardBytes(descriptor,i,rolling_band_);
      auto memory = std::make_unique<DeviceMemory>((defer_scratch||alias_final)?0:host_.back().bytes+guard);
      descriptor.data = alias_final?base_inputs_[i-graph_count_].data:static_cast<char*>(memory->data)+guard;
      if (shard_layout_ && i < 6 && descriptor.rank == 3 && descriptor.element_type == kBeamzF32) {
        const int nx = int(descriptor.dims[2]);
        if (nx >= 64 && nx % 32) {
          descriptor.compact_ny = -1;
          descriptor.row_stride = nx / 32 * 32;
          descriptor.plane_stride = descriptor.row_stride * int(descriptor.dims[1]);
        }
      }
      if(!defer_scratch&&!alias_final&&host_.back().bytes){
        if(host_.back().literal_zero)Check(cudaMemsetAsync(memory->data,0,memory->bytes,stream_.value),"initialize literal zero state");
        else if (BeamzSplitRows(descriptor))
          CopyShardField(descriptor, const_cast<void*>(host_.back().data), cudaMemcpyHostToDevice, stream_.value);
        else Check(cudaMemcpyAsync(descriptor.data, host_.back().data, host_.back().bytes,
                              cudaMemcpyHostToDevice, stream_.value), "upload prepared input");
      }
      resident_bytes_ += memory->bytes;
      allocations_.push_back(std::move(memory));
      base_inputs_.push_back(descriptor);
      // A mapped input is staging storage. Finish this bounded upload before
      // dropping its file pages, while private snapshots retain their bytes.
      // Never discard private modifications in copy-on-write caller mappings.
      const auto mode=nb::getattr(original_input,"mode",nb::none());
      if(nb::isinstance<nb::str>(mode)&&nb::cast<std::string>(mode)=="r"&&nb::hasattr(original_input,"_mmap")){
        const auto mapping=original_input.attr("_mmap");
        if(!mapping.is_none()&&nb::hasattr(mapping,"madvise")){
          const auto mmap=nb::module_::import_("mmap");
          if(nb::hasattr(mmap,"MADV_DONTNEED")){
            Check(cudaStreamSynchronize(stream_.value), "finish bounded mapped input upload");
            mapping.attr("madvise")(mmap.attr("MADV_DONTNEED"));
          }
        }
      }
    }
#if BEAMZ_CUDA_NATIVE_MOVING
    if (mstep_ && attributes_.cpml_enabled)
      neutral_faces_ = TransparentCpmlFaces(host_.data(), attributes_.boundary_code);
#endif
    BindPackedDftGather(base_inputs_[monitor_offset_], host_[monitor_offset_],
                        host_[monitor_offset_ + 1], host_[monitor_offset_ + 4],
                        host_[monitor_offset_ + 3], attributes_.monitor_count);
    ValidateClock(monitor_offset_ + 13, kBeamzF32);
    ValidateClock(monitor_offset_ + 14, kBeamzS32);
    ValidateClock(monitor_offset_ + 15, kBeamzS32);
    std::memcpy(&initial_step_, host_[monitor_offset_ + 14].data, sizeof(int));
    std::memcpy(&initial_elapsed_, host_[monitor_offset_ + 15].data, sizeof(int));
    std::memcpy(&origin_, host_[monitor_offset_ + 13].data, sizeof(float));
    if (initial_step_ < 0 || initial_elapsed_ < 0 || !std::isfinite(origin_))
      throw std::invalid_argument("native clocks require nonnegative steps and finite origin");
    clock_time_ = std::make_unique<DeviceMemory>(sizeof(float));
    resident_bytes_ += sizeof(float);
    if (preserve_) {
      for (int index : persistent_) {
        if(zero_reset_.count(index))continue;
        auto snapshot = std::make_unique<DeviceMemory>(host_[index].bytes);
        if (host_[index].bytes)
          Check(cudaMemcpyAsync(snapshot->data, base_inputs_[index].data,
                                host_[index].bytes, cudaMemcpyDeviceToDevice,
                                stream_.value), "create preserved device state");
        preserve_bytes_ += snapshot->bytes;
        snapshots_.emplace_back(index, std::move(snapshot));
      }
    }
    for(int g=0;g<9;++g){const int offset=source_offset_+3*g;const auto&coeff=base_inputs_[offset];
      if(coeff.dims[0]<=1||!((attributes_.coincident_source_group_mask>>g)&1))continue;
      const auto&starts=base_inputs_[offset+2];
      if(coeff.rank!=4||starts.rank!=2||starts.element_type!=kBeamzS32||starts.dims[0]!=coeff.dims[0]||starts.dims[1]!=3)
        throw std::invalid_argument("coincident sources require matching int32 starts");
      const auto*positions=static_cast<const int32_t*>(host_[offset+2].data);
      for(int64_t source=1;source<coeff.dims[0];++source)for(int axis=0;axis<3;++axis)
        if(positions[3*source+axis]!=positions[axis])throw std::invalid_argument("coincident source starts differ");
    }
    BindInitial();
    step_ = initial_step_;
    elapsed_ = initial_elapsed_;
    UpdateClocks(0);
    Check(cudaStreamSynchronize(stream_.value), "finish native preparation");
#if BEAMZ_CUDA_NATIVE_MOVING
    RetireUploadedHostArrays();
#endif
  }


  ~NativeSession(){
    if(closed_)return;
    int previous;
    if(cudaGetDevice(&previous)!=cudaSuccess||cudaSetDevice(device_)!=cudaSuccess)return;
    if(cudaStreamSynchronize(stream_.value)==cudaSuccess)
      BeamzReleaseStreamGraphs(device_,stream_.value);
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
    for(const auto& stream:moving_policy_streams_)
      if(stream&&stream->value)cudaStreamSynchronize(stream->value);
#endif
    cudaSetDevice(previous);
  }

  void Reset() {
    auto guard = Guard();
    RequireOpen();
    nb::gil_scoped_release release;
    Check(cudaStreamSynchronize(stream_.value), "synchronize before reset");
    if(preserve_)RestorePreservedState();
    else for(int index:persistent_){
      if(!host_[index].bytes)continue;
      if(zero_reset_.count(index))
        Check(cudaMemsetAsync(base_inputs_[index].data,0,host_[index].bytes,stream_.value),"reset captured zero state");
      else if (BeamzSplitRows(base_inputs_[index]))
        CopyShardField(base_inputs_[index], const_cast<void*>(host_[index].data), cudaMemcpyHostToDevice, stream_.value);
      else
        Check(cudaMemcpyAsync(base_inputs_[index].data,host_[index].data,host_[index].bytes,
                              cudaMemcpyHostToDevice,stream_.value),"reset captured host state");
    }
    BindInitial();
    step_ = initial_step_;
    elapsed_ = initial_elapsed_;
    UpdateClocks(0);
    Check(cudaStreamSynchronize(stream_.value), "finish native reset");
  }

  void ConfigureShard(nb::sequence geometries, std::array<int,2> neighbors) {
    auto guard = Guard();
    RequireOpen();
    if (mstep_ || preserve_ || shard_ || nb::len(geometries) != 2)
      throw std::invalid_argument("shards require a fresh private ordinary session");
    DeviceScope device_scope;
    Check(cudaSetDevice(device_), "select shard allocation device");
    auto shard = std::make_unique<ShardResources>();
    shard->neighbors = neighbors;
    std::array<bool,2> stage{};
    for (int side = 0; side < 2; ++side) {
      const int peer = neighbors[side];
      if (peer < -2) throw std::invalid_argument("invalid shard neighbor ordinal");
      stage[side] = peer == -2 || (peer >= 0 && !HasDirectExchange(device_,peer));
    }
    for (int phase = 0; phase < 2; ++phase) {
      const auto host = Describe(geometries[phase], numpy_, false);
      const auto& shape = host.descriptor;
      if (shape.rank != 2 || shape.element_type != kBeamzS32 ||
          shape.dims[0] != 9 || shape.dims[1] != 3 || !host.data)
        throw std::invalid_argument("shard geometry must be a 9 x 3 int32 array");
      const auto* geometry = static_cast<const int32_t*>(host.data);
      if (geometry[0] < 0 || geometry[0] > 2 || geometry[1] < 0 || geometry[2] != 0 ||
          (phase && (geometry[0] != shard->axis || geometry[1] != shard->origin)))
        throw std::invalid_argument("invalid owned shard coordinates");
      shard->axis = geometry[0];
      shard->origin = geometry[1];
      auto& local = shard->phases[phase];
      std::copy_n(geometry, local.host_geometry.size(), local.host_geometry.begin());
      for (int index = 0; index < 6; ++index) {
        const auto& field = current_[(phase * 3 + index) % 6];
        for (int d = 0; d < 3; ++d) {
          const auto logical = geometry[3 + index * 3 + d];
          if (field.dims[d] < 1 ||
              (d != shard->axis && field.dims[d] != logical) ||
              (d == shard->axis && shard->origin + field.dims[d] > logical))
            throw std::invalid_argument("owned field does not fit its global Yee support");
        }
      }
      if (attributes_.cpml_enabled) {
        const int material = phase == 0 ? 6 : kCpmlEInputOffset;
        const auto& meta = host_[material + 6];
        if (!meta.data || meta.descriptor.element_type != kBeamzS32 ||
            meta.descriptor.rank != 2 || meta.descriptor.dims[0] != 6 ||
            meta.descriptor.dims[1] != 5)
          throw std::invalid_argument("invalid packed shard CPML metadata");
        const auto* terms = static_cast<const int32_t*>(meta.data);
        for (int term = 0; term < 6; ++term) {
          const int axis = terms[5 * term + 1];
          if (axis < 0 || axis > 2) throw std::invalid_argument("invalid shard CPML axis");
          const int component = term / 2;
          const auto& psi = psi_current_[phase * 6 + term];
          const auto& target = current_[phase * 3 + component];
          const int stop = geometry[3 + component * 3 + axis];
          const int low = terms[5 * term + 2], high = terms[5 * term + 3];
          const int base = geometry[21 + term];
          const int begin = axis == shard->axis ? shard->origin : 0;
          const int end = begin + int(target.dims[axis]);
          if (low < 0 || high < 0 || low + int64_t(high) > stop || base < 0)
            throw std::invalid_argument("invalid shard CPML geometry");
          const auto covers = [&](int from, int to, int shift) {
            const int start = std::max(begin, from), finish = std::min(end, to);
            return start >= finish || (base <= start + shift &&
                finish + shift <= base + psi.dims[axis]);
          };
          if (!covers(0, low, 0) || !covers(stop - high, stop, low + high - stop))
            throw std::invalid_argument("owned CPML storage misses active recurrence entries");
          for (int d = 0; d < 3; ++d)
            if (d != axis && psi.dims[d] != target.dims[d])
              throw std::invalid_argument("shard CPML transverse dimensions differ from its field");
        }
      }
      local.geometry_storage = std::make_unique<DeviceMemory>(host.bytes);
      local.geometry = shape;
      local.geometry.data = local.geometry_storage->data;
      Check(cudaMemcpy(local.geometry.data, host.data, host.bytes, cudaMemcpyHostToDevice),
            "upload shard geometry");
      shard->bytes += host.bytes;
      Check(cudaEventCreateWithFlags(&local.packed, cudaEventDisableTiming), "create shard pack event");
      Check(cudaEventCreateWithFlags(&local.copied, cudaEventDisableTiming), "create shard copy event");
      for (int component = 0; component < 3; ++component) {
        const auto& field = current_[(1 - phase) * 3 + component];
        auto face = field;
        face.dims[shard->axis] = 1;
        face.row_stride = int(face.dims[2]);
        face.plane_stride = int(face.dims[1] * face.dims[2]);
        const size_t bytes = size_t(face.dims[0]) * face.plane_stride * sizeof(float);
        const bool exchanged = component != 2 - shard->axis;
        if (exchanged && stage[1-phase]) {
          local.receive[component] = std::make_unique<DeviceMemory>(bytes);
          face.data = local.receive[component]->data;
          shard->bytes += bytes;
        }
        if (exchanged && shard->axis != 0 && stage[phase]) {
          local.send[component] = std::make_unique<DeviceMemory>(bytes);
          shard->bytes += bytes;
        }
        local.faces[2 * component] = local.faces[2 * component + 1] = face;
      }
    }
    Check(cudaStreamSynchronize(stream_.value), "finish shard preparation");
    resident_bytes_ += shard->bytes;
    shard_ = std::move(shard);
  }

  static double RunSharded(nb::sequence owners, int steps) {
    if (steps < 1 || nb::len(owners) < 2)
      throw std::invalid_argument("sharded CUDA requires at least two partitions and one step");
    std::vector<NativeSession*> shards;
    for (auto owner : owners) shards.push_back(nb::cast<NativeSession*>(owner));
    auto sorted = shards;
    std::sort(sorted.begin(), sorted.end(), std::less<NativeSession*>{});
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
      throw std::invalid_argument("each shard needs independent field ownership");
    std::vector<std::unique_lock<std::mutex>> guards;
    for (auto* shard : sorted) guards.push_back(shard->Guard());
    DeviceScope device_scope;
    std::vector<BeamzNativeLaunchStorage> launches(shards.size());
    const int axis = shards.front()->shard_ ? shards.front()->shard_->axis : -1;
    int origin = 0;
    for (size_t rank = 0; rank < shards.size(); ++rank) {
      auto& shard = *shards[rank];
      shard.RequireOpen();
      if (axis < 0 || !shard.shard_ || shard.shard_->axis != axis ||
          shard.shard_->origin != origin ||
          shard.step_ != shards.front()->step_ || shard.elapsed_ != shards.front()->elapsed_ ||
          int64_t(shard.step_) + steps > INT32_MAX ||
          int64_t(shard.elapsed_) + steps > INT32_MAX)
        throw std::invalid_argument("shards must have adjacent ownership and synchronized clocks");
      for (const auto& field : shard.current_)
        if (field.dims[axis] > INT32_MAX - origin)
          throw std::invalid_argument("shard extents exceed int32");
      for (int side = 0; side < 2; ++side) {
        const int peer = int(rank) + 2*side - 1;
        const int actual = peer >= 0 && peer < int(shards.size()) ? shards[peer]->device_ : -1;
        const int expected = shard.shard_->neighbors[side];
        if (expected != -2 && expected != actual)
          throw std::invalid_argument("shard neighbor differs from prepared transfer route");
      }
      origin += int(shard.current_[0].dims[axis]);
      const auto a = shard.attributes_;
      BeamzNativePrepare(a, shard.inputs_.data(), shard.outputs_.data(),
          shard.source_offset_, shard.monitor_offset_, shard.workspace_count_,
          false, &launches[rank]);
      auto& program = launches[rank].program;
      for (int phase = 0; phase < 2; ++phase) {
        auto& launch = phase == 0 ? program.h_ab : program.e_ab;
        const auto& local = shard.shard_->phases[phase];
        launch.shard_geometry = local.geometry;
        std::copy_n(local.faces, 6, launch.shard_halos);
        if (!BeamzValidateNativeShard(launch))
          throw std::invalid_argument("invalid prepared shard phase");
      }
      for (const auto& group : launches[rank].groups)
        if (group.coefficients.dims[0] && int64_t(shard.step_) + steps > group.waveforms.dims[1])
          throw std::invalid_argument("sharded execution exceeds prepared source waveforms");
    }
    for (size_t rank = 1; rank < shards.size(); ++rank)
      EnablePeerPair(shards[rank-1]->device_, shards[rank]->device_);
    std::vector<bool> direct(shards.size() - 1);
    for (size_t rank = 0; rank + 1 < shards.size(); ++rank) {
      auto& low = *shards[rank];
      auto& high = *shards[rank + 1];
      const bool prepared = low.shard_->neighbors[1] >= 0 || high.shard_->neighbors[0] >= 0;
      direct[rank] = (prepared || (low.shard_layout_ && high.shard_layout_)) && HasDirectExchange(low.device_, high.device_);
      if (!direct[rank]) continue;
      for (int c = 0; c < 3; ++c) {
        auto h = low.current_[c], e = high.current_[3 + c];
        h.rank = e.rank = 4;
        h.dims[3] = h.dims[axis] - 1;
        e.dims[3] = 0;
        launches[rank].program.h_ab.shard_halos[2*c + 1] = e;
        launches[rank + 1].program.e_ab.shard_halos[2*c] = h;
      }
    }
    const auto start = std::chrono::steady_clock::now();
    nb::gil_scoped_release release;
    auto execute = [&] {
      for (int step = 0; step < steps; ++step) {
        for (size_t rank = 0; rank < shards.size(); ++rank) {
          auto& shard = *shards[rank];
          const auto& program = launches[rank].program;
          if (!program.source_groups[0].coefficients.dims[0] &&
              !program.source_groups[1].coefficients.dims[0] &&
              !program.source_groups[2].coefficients.dims[0]) continue;
          Check(cudaSetDevice(shard.device_), "select pre-electric source device");
          Check(BeamzEnqueueSourcePhase(shard.stream_.value, program.h_ab, program.e_ab,
                program.source_groups, program.source_group_count, 0, step),
                "inject pre-electric source on owned cells");
        }
        for (int phase = 0; phase < 2; ++phase) {
          const int direction = phase == 0 ? 1 : -1;
          for (int rank = 0; rank < int(shards.size()); ++rank) {
            auto& shard = *shards[rank];
            if (rank - direction < 0 || rank - direction >= int(shards.size())) continue;
            Check(cudaSetDevice(shard.device_), "select halo packing device");
            auto& local = shard.shard_->phases[phase];
            for (int c = 0; c < 3; ++c) {
              if (direct[std::min(rank,rank-direction)] || !local.send[c]) continue;
              auto face = local.faces[2 * c];
              face.data = local.send[c]->data;
              const auto field = shard.current_[(1 - phase) * 3 + c];
              const int plane = phase == 0 ? 0 : int(field.dims[axis]) - 1;
              Check(BeamzEnqueueShardFace(shard.stream_.value, field, face, axis, plane),
                    "pack outgoing halo");
            }
            Check(cudaEventRecord(local.packed, shard.stream_.value), "record outgoing halo");
          }
          for (int rank = 0; rank < int(shards.size()); ++rank) {
            auto& shard = *shards[rank];
            const int neighbor = rank + direction;
            if (neighbor < 0 || neighbor >= int(shards.size())) continue;
            const auto& source = *shards[neighbor];
            auto& local = shard.shard_->phases[phase];
            const auto& other = source.shard_->phases[phase];
            Check(cudaSetDevice(shard.device_), "select halo receiving device");
            Check(cudaStreamWaitEvent(shard.stream_.value, other.packed, 0), "wait for outgoing halo");
            if (direct[std::min(rank,neighbor)]) continue;
            for (int c = 0; c < 3; ++c) {
              if (!local.receive[c]) continue;
              const auto bytes = local.receive[c]->bytes;
              if (bytes != size_t(other.faces[2*c].dims[0]) * other.faces[2*c].plane_stride * sizeof(float))
                throw std::invalid_argument("shard peer faces have different sizes");
              const auto* field = static_cast<const char*>(source.current_[(1 - phase) * 3 + c].data);
              const void* data = other.send[c] ? other.send[c]->data :
                  field + (phase == 0 ? 0 : (source.current_[c].dims[0] - 1) * bytes);
              auto* target = local.receive[c]->data;
              const auto& layout = source.current_[(1 - phase) * 3 + c];
              if (axis == 0 && BeamzSplitRows(layout)) {
                const size_t rows = layout.dims[1];
                CopyShardRows(layout, target, cudaMemcpyDeviceToDevice, shard.stream_.value,
                              phase == 0 ? 0 : (layout.dims[0] - 1) * rows, rows,
                              source.device_, shard.device_);
                continue;
              }
              const auto status = source.device_ == shard.device_
                  ? cudaMemcpyAsync(target, data, bytes, cudaMemcpyDeviceToDevice, shard.stream_.value)
                  : cudaMemcpyPeerAsync(target, shard.device_, data, source.device_, bytes, shard.stream_.value);
              Check(status, "copy halo face");
            }
            Check(cudaEventRecord(local.copied, shard.stream_.value), "record halo receipt");
          }
          for (int rank = 0; rank < int(shards.size()); ++rank) {
            auto& shard = *shards[rank];
            Check(cudaSetDevice(shard.device_), "select sharded Yee update device");
            auto& program = launches[rank].program;
            const auto& launch = phase == 0 ? program.h_ab : program.e_ab;
            Check(static_cast<cudaError_t>(BeamzEnqueueNativeShard(shard.stream_.value, launch, shard.shard_->phases[phase].host_geometry.data())),
                  "advance owned Yee fields");
            // A direct read finishes before the reverse Yee phase can update its donor plane.
            const int recipient = rank - direction;
            if (recipient >= 0 && recipient < int(shards.size()) && !direct[std::min(rank,recipient)])
              Check(cudaStreamWaitEvent(shard.stream_.value,
                  shards[recipient]->shard_->phases[phase].copied, 0), "retain halo until its copy completes");
            auto source_launch = program.h_ab;
            if (rank) source_launch.metallic_edges &= ~(1 << (2 * axis));
            if (rank + 1 < int(shards.size())) source_launch.metallic_edges &= ~(1 << (2 * axis + 1));
            Check(BeamzEnqueueSourcePhase(shard.stream_.value, source_launch, program.e_ab,
                  program.source_groups, program.source_group_count, phase + 1, step),
                  "inject source on owned cells");
            if (phase == 1 && program.monitors)
              Check(BeamzEnqueueDftGroups(shard.stream_.value, program.h_ab, program.e_ab,
                      *program.monitors, step), "acquire sharded DFT rows");
          }
        }
      }
    };
    try {
      execute();
      for (auto* shard : shards) {
        Check(cudaSetDevice(shard->device_), "select sharded clock device");
        shard->UpdateClocks(steps);
        Check(cudaStreamSynchronize(shard->stream_.value), "finish sharded solve");
      }
      for (auto* shard : shards) {
        shard->step_ += steps;
        shard->elapsed_ += steps;
        shard->last_nsteps_ = steps;
        ++shard->runs_;
      }
    } catch (...) {
      for (auto* shard : shards)
        if (cudaSetDevice(shard->device_) == cudaSuccess) cudaStreamSynchronize(shard->stream_.value);
      throw;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }

  void RestoreState(nb::sequence values, float origin, int step, int elapsed) {
    auto guard = Guard(); RequireOpen();
    if (preserve_ || !std::isfinite(origin) || step < 0 || elapsed < 0 || elapsed > step)
      throw std::invalid_argument("invalid native continuation clocks or ownership");
    // The persistent inputs are ordered as fields, CPML terms, DFT rows, then
    // clocks. Validate every leaf before touching the resident state.
    if (nb::len(values) != persistent_.size() - 3)
      throw std::invalid_argument("native continuation has the wrong number of state buffers");
    std::vector<HostBuffer> checkpoint;
    checkpoint.reserve(nb::len(values));
    const int clock = monitor_offset_ + 13;
    for (int index : persistent_) {
      if (index >= clock) break;
      auto value = Describe(values[checkpoint.size()], numpy_, false);
      const auto& expected = base_inputs_[index];
      const auto& actual = value.descriptor;
      if (value.bytes != host_[index].bytes || actual.rank != expected.rank ||
          actual.element_type != expected.element_type ||
          !std::equal(actual.dims, actual.dims + actual.rank, expected.dims))
        throw std::invalid_argument("native continuation dtype/shape differs from the prepared state");
      checkpoint.push_back(std::move(value));
    }

    nb::gil_scoped_release release;
    BindInitial();
    size_t leaf = 0;
    for (int index : persistent_) {
      if (index >= clock) break;
      const auto& value = checkpoint[leaf++];
      auto* target = static_cast<unsigned char*>(base_inputs_[index].data);
      if (value.literal_zero) {
        if (value.bytes)
          Check(cudaMemsetAsync(target, 0, value.bytes, stream_.value), "clear continuation state");
        continue;
      }
      if (BeamzSplitRows(base_inputs_[index])) {
        CopyShardField(base_inputs_[index], const_cast<void*>(value.data), cudaMemcpyHostToDevice, stream_.value);
        continue;
      }
      const auto* source = static_cast<const unsigned char*>(value.data);
      constexpr size_t transfer = 64u << 20;
      for (size_t offset = 0; offset < value.bytes; offset += transfer) {
        const size_t bytes = std::min(transfer, value.bytes - offset);
        Check(cudaMemcpyAsync(target + offset, source + offset, bytes,
                              cudaMemcpyHostToDevice, stream_.value), "upload continuation state");
        Check(cudaStreamSynchronize(stream_.value), "finish continuation upload chunk");
      }
    }
    Check(cudaMemcpyAsync(base_inputs_[clock].data, &origin, sizeof(origin),
                          cudaMemcpyHostToDevice, stream_.value), "restore observation origin");
    Check(cudaMemcpyAsync(base_inputs_[clock + 1].data, &step, sizeof(step),
                          cudaMemcpyHostToDevice, stream_.value), "restore absolute step");
    Check(cudaMemcpyAsync(base_inputs_[clock + 2].data, &elapsed, sizeof(elapsed),
                          cudaMemcpyHostToDevice, stream_.value), "restore observation step");
    step_ = step; elapsed_ = elapsed;
    UpdateClocks(0);
    Check(cudaStreamSynchronize(stream_.value), "finish continuation restore");
  }

  nb::dict Run(int nsteps) {
    if (nsteps < 0) nsteps = attributes_.nsteps;
    return RunSchedule({nsteps});
  }

  nb::dict RunSchedule(const std::vector<int>& chunks) {
    if(mstep_)throw std::invalid_argument("One-bank temporal sessions require run_moving; ordinary execution uses kind=generated or program");
    auto guard = Guard();
    RequireOpen();
    if (shard_ || shard_layout_) throw std::invalid_argument("owned shards must be advanced together");
    if (chunks.empty()) throw std::invalid_argument("native schedule cannot be empty");
    int64_t total_steps = 0;
    for (int count : chunks) {
      if (count < 1) throw std::invalid_argument("native chunk steps must be positive");
      if (!recordings_.empty() && count > recording_chunk_steps_)
        throw std::invalid_argument("native recorder chunk exceeds staging capacity");
      total_steps += count;
      if (total_steps > INT32_MAX) throw std::invalid_argument("native schedule exceeds int32");
    }
    const int start_step = preserve_ ? initial_step_ : step_;
    const int start_elapsed = preserve_ ? initial_elapsed_ : elapsed_;
    if (int64_t(start_step) + total_steps > INT32_MAX ||
        int64_t(start_elapsed) + total_steps > INT32_MAX)
      throw std::invalid_argument("native continuation clock exceeds int32");
    for (int group = 0; group < 9; ++group) {
      const auto& coefficients = base_inputs_[source_offset_ + 3 * group];
      const auto& waveforms = base_inputs_[source_offset_ + 3 * group + 1];
      if (!clamp_source_tails_ && coefficients.dims[0] && int64_t(start_step) + total_steps > waveforms.dims[1])
        throw std::invalid_argument("nsteps exceeds a prepared source waveform");
    }
    float cuda_ms = 0, physics_ms = 0, copy_ms = 0;
    const auto started = std::chrono::steady_clock::now();
    {
      nb::gil_scoped_release release;
      while (chunk_starts_.size() < chunks.size()) {
        chunk_starts_.push_back(std::make_unique<Event>());
        chunk_ready_.push_back(std::make_unique<Event>());
        chunk_finished_.push_back(std::make_unique<Event>());
      }
      Check(cudaEventRecord(start_.value, stream_.value), "record native start");
      if (preserve_) {
        BindInitial();RestorePreservedState();
        step_ = initial_step_;
        elapsed_ = initial_elapsed_;
      }
      for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        const int nsteps = chunks[chunk];
        if (chunk)
          Check(cudaEventRecord(chunk_starts_[chunk]->value, stream_.value),
                "record chunk preparation start");
        Check(cudaEventRecord(chunk_ready_[chunk]->value, stream_.value),
              "record chunk preparation end");
        BindOutputs();
        auto a = attributes_;
        a.nsteps = nsteps;
        BeamzNativeLaunchStorage launch;
        BeamzNativePrepare(a, inputs_.data(), outputs_.data(), source_offset_,
                            monitor_offset_, workspace_count_, false, &launch);
        if (tensor_enabled_) launch.program.e_ab.tensor = &tensor_;
        const int error = LaunchOrdinary(launch.program, nsteps);
        Check(static_cast<cudaError_t>(error), "execute native program");
        CopyRecordings(nsteps);
        Check(cudaEventRecord(chunk_finished_[chunk]->value, stream_.value),
              "record chunk physics end");
        UpdateClocks(nsteps);
        step_ += nsteps;
        elapsed_ += nsteps;
        last_bank_ = 0;
      }
      Check(cudaEventRecord(end_.value, stream_.value), "record native end");
      Check(cudaEventSynchronize(end_.value), "synchronize native result");
      Check(cudaEventElapsedTime(&cuda_ms, start_.value, end_.value), "measure native time");
      for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        float physics = 0, preparation = 0;
        Check(cudaEventElapsedTime(&physics, chunk_ready_[chunk]->value,
                                    chunk_finished_[chunk]->value), "measure physics time");
        Check(cudaEventElapsedTime(&preparation,
                                    chunk ? chunk_starts_[chunk]->value : start_.value,
                                    chunk_ready_[chunk]->value), "measure input preparation");
        physics_ms += physics;
        copy_ms += preparation;
      }
      last_nsteps_ = static_cast<int>(total_steps);
      ++runs_;
    }
    const double host_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    nb::dict result;
    result["cuda_ms"] = cuda_ms;
    result["physics_cuda_ms"] = physics_ms;
    result["prepare_cuda_ms"] = copy_ms;
    result["host_ms"] = host_ms;
    result["nsteps"] = static_cast<int>(total_steps);
    result["chunks"] = nb::cast(chunks);
    result["output_bank"] = last_bank_;
    result["physical_field_bank"] =
        current_[0].data == base_inputs_[0].data ? 0 : 1;
    result["preserve_inputs"] = preserve_;
    result["initialize_scratch"] = false;
    result["scratch_zero_bytes"] = 0;
    result["preserving_copy_bytes"] = preserve_ ? preserve_bytes_ : 0;
    return result;
  }

  nb::dict ConfigureDiagnostics(nb::sequence specifications, nb::sequence axes,
                                double domain_measure, std::array<int, 3> owned, nb::sequence interpolation) {
    auto guard = Guard(); RequireOpen();
    const size_t terms = nb::len(specifications);
    if ((terms != 6 && (terms != 9 || !tensor_enabled_)) || nb::len(axes) != terms ||
        !std::isfinite(domain_measure) || domain_measure <= 0)
      throw std::invalid_argument("diagnostics require six field materials, optional tensor terms, and a positive domain measure");
    std::vector<std::unique_ptr<DeviceMemory>> allocations;
    std::vector<BeamzDiagnosticMaterial> materials(terms);
    auto nodes = current_[3];
    if (terms == 9) {
      ++nodes.dims[2];
      for (int component = 0; component < 3; ++component)
        for (int axis = 0; axis < 3; ++axis)
          if (current_[3 + component].dims[axis] != nodes.dims[axis] - (axis == 2 - component))
            throw std::invalid_argument("tensor diagnostics require complete Yee supports");
    }
    const size_t interpolation_count = nb::len(interpolation);
    if (interpolation_count && (interpolation_count != 3 || terms != 9))
      throw std::invalid_argument("tensor diagnostic interpolation requires three width arrays");
    const int cut_axis = owned[0];
    if (cut_axis < -1 || cut_axis > 2 ||
        (cut_axis >= 0 && (terms != 6 || owned[1] < 0 || (owned[2] != -1 && owned[2] <= owned[1]))))
      throw std::invalid_argument("invalid diagnostics ownership interval");
    size_t bytes = 0;
    auto upload = [&](const HostBuffer& host) -> void* {
      if (!host.bytes) return nullptr;
      auto memory = std::make_unique<DeviceMemory>(host.bytes);
      Check(cudaMemcpyAsync(memory->data, host.data, host.bytes,
                            cudaMemcpyHostToDevice, stream_.value), "upload diagnostics");
      auto* pointer = memory->data;
      bytes += memory->bytes;
      allocations.push_back(std::move(memory));
      return pointer;
    };
    for (size_t component = 0; component < terms; ++component) {
      auto field = component < 6 ? current_[component] : nodes;
      if (cut_axis >= 0) {
        const int64_t end = owned[2] < 0 ? field.dims[cut_axis] : owned[2];
        if (end > field.dims[cut_axis] || end <= owned[1])
          throw std::invalid_argument("diagnostics ownership exceeds field support");
        field.dims[cut_axis] = end - owned[1];
      }
      if (field.rank != 3 || field.element_type != kBeamzF32)
        throw std::invalid_argument("diagnostics require rank-three FP32 fields");
      const int64_t count = field.dims[0] * field.dims[1] * field.dims[2];
      if (count < 1 || count > INT32_MAX)
        throw std::invalid_argument("diagnostic field size must fit positive int32 indexing");
      const auto spec = nb::cast<nb::dict>(specifications[component]);
      if (!spec.contains("values") || !spec.contains("ends"))
        throw std::invalid_argument("diagnostic material requires values and ends");
      const auto values = DiagnosticArray(spec["values"], numpy_, "float32");
      const auto ends = DiagnosticArray(spec["ends"], numpy_, "int32");
      const size_t size = values.bytes / sizeof(float);
      if (!size || size > INT32_MAX)
        throw std::invalid_argument("diagnostic material size is out of range");
      const auto* host_values = static_cast<const float*>(values.data);
      for (size_t index = 0; index < size; ++index)
        if (!std::isfinite(host_values[index]))
          throw std::invalid_argument("diagnostic materials must be finite");
      auto& material = materials[component]; material.count = int(size);
      if (ends.bytes) {
        if (ends.bytes / sizeof(int32_t) != size)
          throw std::invalid_argument("diagnostic RLE values/ends size differs");
        const auto* boundaries = static_cast<const int32_t*>(ends.data);
        int32_t previous = 0;
        for (size_t index = 0; index < size; ++index) {
          if (boundaries[index] <= previous || boundaries[index] > count)
            throw std::invalid_argument("diagnostic RLE ends must increase within the field");
          previous = boundaries[index];
        }
        if (previous != count)
          throw std::invalid_argument("diagnostic RLE must cover the full field");
        material.encoding = 2;
      } else if (size == 1) material.encoding = 0;
      else if (int64_t(size) == count) material.encoding = 1;
      else throw std::invalid_argument("diagnostic material is neither scalar, dense nor complete RLE");
      material.values = static_cast<const float*>(upload(values));
      material.ends = static_cast<const int32_t*>(upload(ends));
      const auto widths = nb::cast<nb::sequence>(axes[component]);
      if (nb::len(widths) != 3)
        throw std::invalid_argument("diagnostic axes require z, y, x widths");
      for (int axis = 0; axis < 3; ++axis) {
        const auto width = DiagnosticArray(widths[axis], numpy_, "float64");
        if (width.bytes / sizeof(double) != size_t(field.dims[axis]))
          throw std::invalid_argument("diagnostic width length differs from field support");
        const auto* host_widths = static_cast<const double*>(width.data);
        for (int64_t index = 0; index < field.dims[axis]; ++index)
          if (!std::isfinite(host_widths[index]) ||
              (interpolation_count ? host_widths[index] < 0 : host_widths[index] <= 0))
            throw std::invalid_argument("invalid diagnostic integration width");
        material.widths[axis] = static_cast<const double*>(upload(width));
      }
    }
    std::array<const double*, 3> interpolation_widths;
    for (int component = 0; component < 3; ++component) {
      const int axis = 2 - component;
      auto& widths = interpolation_widths[component];
      widths = materials[3 + component].widths[axis];
      if (interpolation_count) {
        const auto width = DiagnosticArray(interpolation[component], numpy_, "float64");
        const auto count = current_[3 + component].dims[axis];
        const auto* host = static_cast<const double*>(width.data);
        if (width.bytes / sizeof(double) != size_t(count) ||
            std::any_of(host, host + count, [](double x) { return !std::isfinite(x) || x <= 0; }))
          throw std::invalid_argument("invalid tensor diagnostic interpolation widths");
        widths = static_cast<const double*>(upload(width));
      }
    }
    auto partials = std::make_unique<DeviceMemory>(
        terms * kBeamzDiagnosticBlocks * sizeof(BeamzDiagnosticPartial));
    bytes += partials->bytes;
    Check(cudaStreamSynchronize(stream_.value), "finish diagnostic preparation");
    resident_bytes_ = resident_bytes_ - diagnostic_bytes_ + bytes;
    diagnostic_bytes_ = bytes; diagnostic_domain_measure_ = domain_measure;
    diagnostic_materials_ = std::move(materials);
    diagnostic_ownership_ = owned;
    diagnostic_interpolation_ = interpolation_widths;
    diagnostic_allocations_ = std::move(allocations);
    diagnostic_partials_ = std::move(partials);
    nb::dict result; result["gpu_bytes"] = bytes; return result;
  }

  size_t ConfigureEffects(nb::sequence tensor, nb::sequence scatters,
                          nb::sequence recorders, int chunk_steps, bool gather_at_boundaries=false) {
    auto guard = Guard(); RequireOpen();
    if (runs_ || tensor_enabled_ || !effects_allocations_.empty())
      throw std::invalid_argument("effects are configured once on a supported unused session");
    if (gather_at_boundaries && (!mstep_ || !rolling_band_))
      throw std::invalid_argument("boundary sampling requires rolling CUDA");
    if (chunk_steps < 1 || chunk_steps > 256)
      throw std::invalid_argument("invalid native effect chunk capacity");
    std::vector<std::unique_ptr<DeviceMemory>> allocations;
    std::vector<BeamzNativeScatter> scatter_plans;
    std::vector<NativeRecording> recordings;
#if BEAMZ_CUDA_NATIVE_MOVING
    std::vector<beamz::cuda::moving::SourceCell> source_cells;
#endif
    BeamzTensorLaunch tensor_plan{};
    size_t resident = 0;
    auto upload = [&](const HostBuffer& host) {
      auto result = host.descriptor;
      if (!host.bytes) return result;
      auto memory = std::make_unique<DeviceMemory>(host.bytes);
      if (!host.literal_zero) {
        Check(cudaMemcpyAsync(memory->data, host.data, host.bytes,
              cudaMemcpyHostToDevice, stream_.value), "upload native effect");
        Check(cudaStreamSynchronize(stream_.value), "finish native effect upload");
      }
      result.data = memory->data;
      resident += memory->bytes;
      allocations.push_back(std::move(memory));
      return result;
    };
    auto count_field = [&](int component) {
      const auto& field = base_inputs_.at(component);
      if (field.rank != 3 || field.element_type != kBeamzF32)
        throw std::invalid_argument("native effects require rank-three FP32 fields");
      return field.dims[0] * field.dims[1] * field.dims[2];
    };
    auto checked_indices = [](const HostBuffer& host, int64_t count) {
      if (host.descriptor.element_type != kBeamzS32 || host.literal_zero)
        throw std::invalid_argument("effect gathers require int32 indices");
      const auto* indices = static_cast<const int32_t*>(host.data);
      for (size_t i = 0; i < host.bytes / sizeof(int32_t); ++i)
        if (indices[i] < 0 || indices[i] >= count)
          throw std::invalid_argument("effect gather index exceeds its field");
    };
    const auto tensor_size = nb::len(tensor);
    if (tensor_size) {
      if (tensor_size != 8)
        throw std::invalid_argument("native tensor effect requires eight descriptors");
      // Final three descriptors describe uninitialized per-step curl scratch.
      for (size_t i = 0; i < tensor_size; ++i) {
        auto buffer = upload(Describe(tensor[i], numpy_, false));
        if (i == 0) tensor_plan.cross = buffer;
        else if (i == 1) tensor_plan.origin = buffer;
        else if (i < 5) tensor_plan.widths[i - 2] = buffer;
        else tensor_plan.curls[i - 5] = buffer;
      }
    }
    for (nb::handle entry : scatters) {
      const auto spec = nb::cast<nb::sequence>(entry);
      if (nb::len(spec) != 6)
        throw std::invalid_argument("invalid native scatter descriptor");
      const int timing = nb::cast<int>(spec[0]), component = nb::cast<int>(spec[1]);
      if (timing < 0 || timing > 2 || component < 0 || component > 2)
        throw std::invalid_argument("invalid native scatter timing/component");
      const auto indices = Describe(spec[2], numpy_, false);
      const auto ends = Describe(spec[3], numpy_, false);
      const auto coefficients = Describe(spec[4], numpy_, false);
      const auto waveform = Describe(spec[5], numpy_, false);
      checked_indices(indices, count_field((timing == 1 ? 0 : 3) + component));
      const auto n = indices.descriptor.dims[0], ncoeff = coefficients.descriptor.dims[0];
      if (indices.descriptor.rank != 1 || !n || ends.descriptor.rank != 1 ||
          ends.descriptor.element_type != kBeamzS32 ||
          coefficients.descriptor.rank != 1 || coefficients.descriptor.element_type != kBeamzF32 ||
          waveform.descriptor.rank != 1 || waveform.descriptor.element_type != kBeamzF32 ||
          !waveform.bytes || waveform.literal_zero || coefficients.literal_zero ||
          (ends.bytes ? ends.descriptor.dims[0] != n || ends.literal_zero : ncoeff != n))
        throw std::invalid_argument("invalid native scatter shape/type");
      const auto* cells = static_cast<const int32_t*>(indices.data);
      for (int64_t i = 1; i < n; ++i)
        if (cells[i - 1] >= cells[i])
          throw std::invalid_argument("native scatter cells must be distinct and ordered");
      if (ends.bytes) {
        const auto* offsets = static_cast<const int32_t*>(ends.data);
        int last = 0;
        for (int64_t i = 0; i < n; ++i) {
          if (offsets[i] <= last || offsets[i] > ncoeff)
            throw std::invalid_argument("invalid native scatter offsets");
          last = offsets[i];
        }
        if (last != ncoeff) throw std::invalid_argument("native scatter offsets are incomplete");
      }
#if BEAMZ_CUDA_NATIVE_MOVING
      if(mstep_)AppendMovingSourceCells(source_cells,base_inputs_.data(),timing,component,indices);
#endif
      scatter_plans.push_back({upload(indices), upload(ends), upload(coefficients),
                               upload(waveform), timing, component});
    }
    for (nb::handle entry : recorders) {
      const auto spec = nb::cast<nb::sequence>(entry);
      if (nb::len(spec) != 6)
        throw std::invalid_argument("invalid native recorder descriptor");
      const int component = nb::cast<int>(spec[0]), interval = nb::cast<int>(spec[1]);
      const float sign = nb::cast<float>(spec[2]);
      if (component < 0 || component > 5 || interval < 1 || std::abs(sign) != 1)
        throw std::invalid_argument("invalid native recorder component/cadence");
      const auto indices = Describe(spec[3], numpy_, false);
      const auto weights = Describe(spec[4], numpy_, false);
      checked_indices(indices, count_field(component));
      // The host history can exceed int32 total elements. Only a bounded
      // interval of history resides on the GPU, with positive int32 points.
      auto destination=DescribeRecordingDestination(spec[5],numpy_);
      const auto capacity=destination.frames, points=destination.points;
      if (capacity < 0 || capacity > INT32_MAX || points < 1 || points > INT32_MAX ||
          (indices.bytes
              ? indices.descriptor.rank != 2 || weights.descriptor.rank != 2 ||
                indices.descriptor.dims[0] != points || weights.descriptor.dims[0] != points ||
                weights.descriptor.dims[1] != indices.descriptor.dims[1] ||
                weights.descriptor.element_type != kBeamzF32 || weights.literal_zero
              : weights.bytes || points != count_field(component)))
        throw std::invalid_argument("invalid native recorder gather plan");
      if (gather_at_boundaries && !indices.bytes)
        throw std::invalid_argument("boundary gathers require interpolation indices");
      const int64_t staging_count = std::min(capacity, int64_t(gather_at_boundaries?1:1+(chunk_steps-1)/interval));
      HostBuffer scratch;
      scratch.descriptor = {nullptr, 2, kBeamzF32, {staging_count, points, 1, 1}};
      scratch.bytes = size_t(staging_count) * points * sizeof(float);
      scratch.literal_zero = true;
      recordings.push_back({{upload(indices), upload(weights), upload(scratch),
                             component, interval, sign}, std::move(destination.owner), destination.data, capacity,
                            mstep_&&!gather_at_boundaries&&indices.bytes?Describe(spec[3],numpy_,true):HostBuffer{}});
    }
    tensor_enabled_ = tensor_size != 0;
    recording_chunk_steps_ = chunk_steps;
    tensor_ = tensor_plan;
    scatters_ = std::move(scatter_plans);
#if BEAMZ_CUDA_NATIVE_MOVING
    moving_source_cells_=std::move(source_cells);
#endif
    recordings_ = std::move(recordings);
    boundary_gathers_ = gather_at_boundaries;
    effects_allocations_ = std::move(allocations);
    effects_bytes_ = resident;
    resident_bytes_ += resident;
    return resident;
  }

  void RebindRecordings(nb::sequence histories) {
    auto guard=Guard();RequireOpen();
    if(nb::len(histories)!=recordings_.size())throw std::invalid_argument("native recorder destination count differs");
    std::vector<RecordingDestination> destinations;
    destinations.reserve(recordings_.size());
    for(size_t i=0;i<recordings_.size();++i){
      auto destination=DescribeRecordingDestination(histories[i],numpy_);
      const auto& record=recordings_[i];
      if(destination.frames!=record.capacity||destination.points!=record.plan.frames.dims[1])
        throw std::invalid_argument("native recorder destination shape differs");
      destinations.push_back(std::move(destination));
    }
    for(size_t i=0;i<recordings_.size();++i){
      recordings_[i].owner=std::move(destinations[i].owner);
      recordings_[i].history=destinations[i].data;
    }
  }

  nb::dict FieldDiagnostics() {
    auto guard = Guard(); RequireOpen();
    if (!diagnostic_partials_)
      throw std::invalid_argument("configure_diagnostics must precede field_diagnostics");
    std::vector<BeamzDiagnosticPartial> partials(diagnostic_materials_.size() * kBeamzDiagnosticBlocks);
    {
      nb::gil_scoped_release release;
      for (int component = 0; component < 6; ++component) {
        auto field = current_[component];
        const auto [axis, begin, end] = diagnostic_ownership_;
        if (axis >= 0) {
          const int64_t strides[] = {field.plane_stride, field.row_stride, 1};
          field.data = static_cast<float*>(field.data) + begin * strides[axis];
          field.dims[axis] = (end < 0 ? field.dims[axis] : end) - begin;
        }
        Check(BeamzEnqueueFieldDiagnostics(stream_.value, field,
            diagnostic_materials_[component], diagnostic_domain_measure_,
            static_cast<BeamzDiagnosticPartial*>(diagnostic_partials_->data) +
                component * kBeamzDiagnosticBlocks), "reduce native field diagnostics");
      }
      if (diagnostic_materials_.size() == 9) {
        auto nodes = current_[3];
        ++nodes.dims[2];
        constexpr int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        for (int term = 0; term < 3; ++term) {
          const int left = pairs[term][0], right = pairs[term][1];
          Check(BeamzEnqueueTensorDiagnostics(
              stream_.value, current_[3 + left], 2 - left, current_[3 + right], 2 - right,
              nodes, diagnostic_materials_[6 + term],
              diagnostic_interpolation_[left], diagnostic_interpolation_[right], diagnostic_domain_measure_,
              static_cast<BeamzDiagnosticPartial*>(diagnostic_partials_->data) +
                  (6 + term) * kBeamzDiagnosticBlocks), "reduce native tensor energy");
        }
      }
      Check(cudaMemcpyAsync(partials.data(), diagnostic_partials_->data,
          partials.size() * sizeof(BeamzDiagnosticPartial), cudaMemcpyDeviceToHost,
          stream_.value), "read diagnostic partials");
      Check(cudaStreamSynchronize(stream_.value), "finish native field diagnostics");
    }
    nb::list sums, maxima, finite;
    for (size_t component = 0; component < diagnostic_materials_.size(); ++component) {
      double sum = 0; float maximum = 0; uint32_t nonfinite = 0;
      for (int block = 0; block < kBeamzDiagnosticBlocks; ++block) {
        const auto& value = partials[component * kBeamzDiagnosticBlocks + block];
        sum += value.sum; maximum = std::max(maximum, value.maximum);
        nonfinite |= value.nonfinite;
      }
      sums.append(float(sum));
      maxima.append(nonfinite & 1 ? std::numeric_limits<float>::quiet_NaN() : maximum);
      finite.append(nonfinite == 0);
    }
    nb::dict result; result["sums"] = sums; result["max_fields"] = maxima;
    result["finite"] = finite; result["gpu_bytes"] = diagnostic_bytes_;
    return result;
  }

  int64_t ConfigureMonitorDiagnostics(int real, int imag,
      const std::vector<std::vector<int64_t>>& rows, size_t budget) {
    auto guard = Guard();
    RequireOpen();
    ClearMonitorDiagnosticsUnlocked();
    if (rows.empty()) return 0;
    const auto valid_output = [&](int index) {
      return index >= 0 && size_t(index) < outputs_.size() &&
          outputs_[index].rank == 1 && outputs_[index].element_type == kBeamzF32;
    };
    if (!valid_output(real) || !valid_output(imag) ||
        outputs_[real].dims[0] != outputs_[imag].dims[0])
      throw std::invalid_argument("monitor diagnostics need matching packed FP32 DFT vectors");
    size_t previous_values = 0, max_partials = 0;
    int64_t end = 0;
    std::vector<MonitorDiagnosticRow> ranges;
    for (const auto& row : rows) {
      if (row.size() != 3 || row[0] < end || row[1] <= 0 || row[2] <= 0 ||
          row[1] > std::numeric_limits<int>::max() || row[2] > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid monitor diagnostic range");
      const int64_t count = row[1] * row[2];
      if (row[0] > outputs_[real].dims[0] || count > outputs_[real].dims[0] - row[0])
        throw std::invalid_argument("monitor diagnostic range exceeds DFT arena");
      const size_t partials = row[1] * (1 + (row[2] - 1) / kBeamzDftDiagnosticTile);
      if (partials > std::numeric_limits<int>::max()) return -1;
      ranges.push_back({size_t(row[0]), previous_values, int(row[1]), int(row[2])});
      max_partials = std::max(max_partials, partials);
      previous_values += count;
      end = row[0] + count;
    }
    if (previous_values > budget / (2 * sizeof(float))) return -1;
    const size_t previous_bytes = previous_values * 2 * sizeof(float);
    const size_t partial_bytes = max_partials * sizeof(BeamzDftNorms);
    const size_t bytes = previous_bytes + partial_bytes + sizeof(BeamzDftResidual);
    size_t free_bytes = 0, total_bytes = 0;
    Check(cudaMemGetInfo(&free_bytes, &total_bytes), "read monitor diagnostic capacity");
    if (bytes > budget || bytes > free_bytes) return -1;
    auto previous = std::make_unique<DeviceMemory>(previous_bytes);
    auto partials = std::make_unique<DeviceMemory>(partial_bytes);
    auto residual = std::make_unique<DeviceMemory>(sizeof(BeamzDftResidual));
    monitor_diagnostic_rows_ = std::move(ranges);
    monitor_previous_ = std::move(previous);
    monitor_partials_ = std::move(partials);
    monitor_residual_ = std::move(residual);
    monitor_real_ = real;
    monitor_imag_ = imag;
    monitor_previous_values_ = previous_values;
    resident_bytes_ += bytes;
    monitor_diagnostic_bytes_ = bytes;
    return bytes;
  }

  nb::tuple MonitorDiagnostics() {
    auto guard = Guard();
    RequireOpen();
    if (!monitor_residual_) return nb::make_tuple(nb::none(), true);
    BeamzDftResidual result{};
    const bool compare = monitor_diagnostic_ready_;
    {
      nb::gil_scoped_release release;
      auto* previous = static_cast<float*>(monitor_previous_->data);
      auto* residual = static_cast<BeamzDftResidual*>(monitor_residual_->data);
      Check(cudaMemsetAsync(residual, 0, sizeof(*residual), stream_.value),
            "reset monitor diagnostic reduction");
      for (const auto& row : monitor_diagnostic_rows_)
        Check(BeamzEnqueueDftDiagnostics(stream_.value,
            static_cast<const float*>(outputs_[monitor_real_].data) + row.offset,
            static_cast<const float*>(outputs_[monitor_imag_].data) + row.offset,
            previous + row.previous, previous + monitor_previous_values_ + row.previous,
            row.frequencies, row.points, compare,
            static_cast<BeamzDftNorms*>(monitor_partials_->data), residual),
            "reduce native DFT change");
      Check(cudaMemcpyAsync(&result, residual, sizeof(result), cudaMemcpyDeviceToHost,
                            stream_.value), "read monitor diagnostic reduction");
      Check(cudaStreamSynchronize(stream_.value), "finish monitor diagnostics");
    }
    double change = 0;
    std::memcpy(&change, &result.maximum_bits, sizeof(change));
    monitor_diagnostic_ready_ = true;
    return nb::make_tuple(compare ? nb::cast(change) : nb::none(), !result.nonfinite);
  }

  void ClearMonitorDiagnostics() {
    auto guard = Guard();
    RequireOpen();
    ClearMonitorDiagnosticsUnlocked();
  }

  void ClearMonitorDiagnosticsUnlocked() {
    monitor_previous_.reset();
    monitor_partials_.reset();
    monitor_residual_.reset();
    monitor_diagnostic_rows_.clear();
    monitor_diagnostic_ready_ = false;
    resident_bytes_ -= monitor_diagnostic_bytes_;
    monitor_diagnostic_bytes_ = 0;
  }

  nb::object Probe(const std::vector<std::vector<int>>&requests){
    auto guard=Guard();RequireOpen();
    auto array=numpy_.attr("empty")(nb::make_tuple(requests.size()),"dtype"_a="int32");
    const auto host=Describe(array,numpy_,false);auto*target=static_cast<int32_t*>(const_cast<void*>(host.data));
    for(size_t i=0;i<requests.size();++i){const auto&r=requests[i];
      if(r.size()!=2||r[0]<0||size_t(r[0])>=outputs_.size()||!(r[0]<18||(r[0]>=30&&r[0]<=32)))throw std::invalid_argument("probe requires physical state or DFT outputs");
      const auto&value=outputs_[r[0]];int64_t size=1;for(int a=0;a<value.rank;++a)size*=value.dims[a];
      if(r[1]<0||r[1]>=size||value.element_type!=kBeamzF32)throw std::invalid_argument("probe index/type differs from state");
      const int offset=value.rank==3?BeamzPhysicalIndex(value,r[1]):r[1];
      Check(cudaMemcpyAsync(target+i,static_cast<const float*>(value.data)+offset,4,cudaMemcpyDeviceToHost,stream_.value),"probe native state");
    }
    Check(cudaStreamSynchronize(stream_.value),"finish state probe");return array;
  }

  nb::list Download() {
    auto guard = Guard();
    RequireOpen();
    nb::list result;
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto& host = host_[output_aliases_[i]];
      nb::object array = numpy_.attr("empty")(host.shape, "dtype"_a=host.dtype);
      const auto destination = Describe(array, numpy_, false);
      if(deferred_scratch_.count(output_aliases_[i])){if(host.bytes){if(zero_reset_.count(output_aliases_[i]))std::memset(const_cast<void*>(destination.data),0,host.bytes);else std::memcpy(const_cast<void*>(destination.data),host.data,host.bytes);}}
      else if (BeamzSplitRows(outputs_[i]))
        CopyShardField(outputs_[i], const_cast<void*>(destination.data), cudaMemcpyDeviceToHost, stream_.value);
      else if (host.bytes)
        Check(cudaMemcpyAsync(const_cast<void*>(destination.data), outputs_[i].data,
                              host.bytes, cudaMemcpyDeviceToHost, stream_.value),
              "download native output");
      result.append(array);
    }
    Check(cudaStreamSynchronize(stream_.value), "finish native download");
    return result;
  }

  nb::object DownloadDeviceState(size_t budget) {
    auto guard = Guard();
    RequireOpen();
    std::vector<std::pair<size_t, size_t>> leaves;
    size_t bytes = 0;
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const int input = output_aliases_[i];
      if (!persistent_.count(input)) continue;
      if (outputs_[i].element_type != kBeamzF32 && outputs_[i].element_type != kBeamzBF16)
        throw std::invalid_argument("native CUDA checkpoints require FP32 or BF16 buffers");
      bytes = (bytes + 255) & ~size_t(255);
      leaves.emplace_back(i, bytes);
      bytes += host_[input].bytes;
    }
    const auto live = checkpoint_bytes_->load(std::memory_order_relaxed);
    if (live > budget || bytes > budget - live) return nb::none();
    DeviceScope device_scope;
    Check(cudaSetDevice(device_), "select checkpoint CUDA device");
    auto checkpoint = std::make_unique<CheckpointStorage>();
    checkpoint->device = device_;
    checkpoint->bytes = bytes;
    checkpoint->live_bytes = checkpoint_bytes_;
    cudaError_t status;
    {
      nb::gil_scoped_release release;
      status = cudaMalloc(&checkpoint->data, std::max(bytes, size_t{1}));
      if (status == cudaSuccess) {
        checkpoint_bytes_->fetch_add(bytes, std::memory_order_relaxed);
        auto* destination = static_cast<unsigned char*>(checkpoint->data);
        for (const auto& [output, offset] : leaves) {
          const auto length = host_[output_aliases_[output]].bytes;
          if (BeamzSplitRows(outputs_[output]))
            CopyShardField(outputs_[output], destination + offset, cudaMemcpyDeviceToDevice, stream_.value);
          else if (length)
            Check(cudaMemcpyAsync(destination + offset, outputs_[output].data,
                                  length, cudaMemcpyDeviceToDevice, stream_.value),
                  "copy native CUDA checkpoint");
        }
        Check(cudaStreamSynchronize(stream_.value), "finish native CUDA checkpoint");
      }
    }
    if (status == cudaErrorMemoryAllocation) {
      cudaGetLastError();
      return nb::none();
    }
    Check(status, "allocate native CUDA checkpoint");
    auto* destination = static_cast<unsigned char*>(checkpoint->data);
    nb::capsule owner(checkpoint.get(), [](void* address) noexcept {
      delete static_cast<CheckpointStorage*>(address);
    });
    checkpoint.release();
    nb::list result;
    for (const auto& [output, offset] : leaves) {
      const auto& value = outputs_[output];
      size_t shape[4]{};
      for (int axis = 0; axis < value.rank; ++axis)
        shape[axis] = size_t(value.dims[axis]);
      const auto dtype = value.element_type == kBeamzBF16
          ? nb::dlpack::dtype{uint8_t(nb::dlpack::dtype_code::Bfloat), 16, 1}
          : nb::dtype<float>();
      result.append(nb::ndarray<nb::jax, nb::device::cuda, nb::c_contig>(
          destination + offset, value.rank, shape, owner, nullptr,
          dtype, nb::device::cuda::value, device_).cast());
    }
    return result;
  }

  void DownloadInto(int output_index, nb::handle destination_value) {
    auto guard = Guard();
    RequireOpen();
    if (output_index < 0 ||
        static_cast<size_t>(output_index) >= outputs_.size())
      throw std::invalid_argument("native output index is out of range");
    // Describe retains the caller's ndarray (including a NumPy memmap) while
    // the GIL is released. No full-sized host staging array is allocated.
    const auto destination = Describe(destination_value, numpy_, false);
    if (!nb::cast<bool>(destination.array.attr("flags").attr("writeable")))
      throw std::invalid_argument("native download destination must be writable");
    const auto& expected = host_[output_aliases_[output_index]];
    const auto& actual = destination.descriptor;
    if (actual.element_type != expected.descriptor.element_type ||
        actual.rank != expected.descriptor.rank ||
        destination.bytes != expected.bytes)
      throw std::invalid_argument("native download destination dtype/shape differs");
    for (int axis = 0; axis < actual.rank; ++axis)
      if (actual.dims[axis] != expected.descriptor.dims[axis])
        throw std::invalid_argument("native download destination dtype/shape differs");
    if(deferred_scratch_.count(output_aliases_[output_index])){
      nb::gil_scoped_release release;if(expected.bytes){if(zero_reset_.count(output_aliases_[output_index]))std::memset(const_cast<void*>(destination.data),0,expected.bytes);else std::memcpy(const_cast<void*>(destination.data),expected.data,expected.bytes);}return;
    }
    auto* target = static_cast<unsigned char*>(const_cast<void*>(destination.data));
    const auto* source = static_cast<const unsigned char*>(outputs_[output_index].data);
    if (BeamzSplitRows(outputs_[output_index])) {
      nb::gil_scoped_release release;
      CopyShardField(outputs_[output_index], target, cudaMemcpyDeviceToHost, stream_.value);
      return;
    }
    constexpr size_t kTransferBytes = 64u << 20;
    {
      nb::gil_scoped_release release;
      for (size_t offset = 0; offset < expected.bytes; offset += kTransferBytes) {
        const size_t bytes = std::min(kTransferBytes, expected.bytes - offset);
        Check(cudaMemcpyAsync(target + offset, source + offset, bytes,
                              cudaMemcpyDeviceToHost, stream_.value),
              "download native output into caller storage");
        // Bound pageable-host staging even when the destination spans GiBs.
        Check(cudaStreamSynchronize(stream_.value), "finish native download chunk");
      }
      if (!expected.bytes)
        Check(cudaStreamSynchronize(stream_.value), "finish empty native download");
    }
  }

  // Each rank owns [begin,end) in global cells and carries a causal halo on
  // every internal face. The first window starts from consistent initial data.
  // Subsequent windows consume only the neighbor's finished physical fields.
#if BEAMZ_CUDA_NATIVE_MOVING
  static double RunTemporalPartitions(nb::sequence owners, const std::vector<std::array<int, 3>>& spans,
                                      int axis, int steps, int window, int depth, int slots,
                                      bool asynchronous, const std::vector<bool>& packet_links) {
    const int macros = depth > 0 ? window / depth : 0;
    const int count = int(nb::len(owners));
    if (count < 2 || int(spans.size()) != count || axis < 0 || axis > 2 || steps <= 0 ||
        !beamz::cuda::moving::ValidDepth(depth) || macros < 2 || macros > kSlabMacros || macros % 2 ||
        window != macros * depth || slots < 2 || slots > 16)
      throw std::invalid_argument("invalid temporal partition dimensions or window");
    if (!packet_links.empty() && packet_links.size() != size_t(count - 1))
      throw std::invalid_argument("packet_links must contain one entry per temporal interface");
    DeviceScope restore;
    std::vector<NativeSession*> ranks;
    for (auto owner : owners) ranks.push_back(nb::cast<NativeSession*>(owner));
    auto ordered = ranks;
    std::sort(ordered.begin(), ordered.end(), std::less<NativeSession*>{});
    if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end())
      throw std::invalid_argument("temporal partitions require distinct sessions");
    std::vector<std::unique_lock<std::mutex>> locks;
    for (auto* rank : ordered) locks.push_back(rank->Guard());
    const bool tensor = ranks.front()->tensor_enabled_;
    const int exchange_width = tensor ? 2 * window : window;
    if (tensor) asynchronous = false;
    struct Stage {
      int device = CurrentDevice();
      Stream stream;
      Event sent, unpacked;
      DeviceMemory send, receive;
      explicit Stage(size_t bytes) : send(bytes), receive(bytes) {}
      ~Stage() { cudaSetDevice(device); }
    };
    using Face = TemporalFace;
    struct Part {
      int device = -1;
      ~Part() {
        if (device >= 0) cudaSetDevice(device);
      }
      std::vector<Face> up, down;
      std::unique_ptr<TemporalPeerTransport> peer_transport;
      std::unique_ptr<Stage> upper_stage, lower_stage;
      std::unique_ptr<Event> low, high, exchanged;
      std::unique_ptr<Stream> transfer;
      BeamzNativeLaunchStorage launch{};
      int low_done, high_needed;
    };
    std::vector<Part> parts(count);
    for (int i = 0; i < count; ++i) {
      auto& s = *ranks[i];
      auto& p = parts[i];
      const auto [origin, begin, end] = spans[i];
      s.RequireOpen();
      const bool fixed_cpml = s.attributes_.boundary_code == (12 << 8);
      const int low_guard = fixed_cpml && !(s.neutral_faces_ & (1 << (2 * axis))) ? 13 : 1;
      const int high_guard = fixed_cpml && !(s.neutral_faces_ & (2 << (2 * axis))) ? 13 : 1;
      if (origin < 0 || begin < origin || end <= begin ||
          (i ? begin != spans[i - 1][2] : begin != 0 || origin != 0) ||
          (i && begin - origin < exchange_width + low_guard) ||
          (i + 1 < count && s.current_[0].dims[axis] - (end - origin) < exchange_width + high_guard) ||
          ((i || i + 1 < count) && end - begin < exchange_width + 1) || s.step_ != ranks[0]->step_ ||
          s.elapsed_ != ranks[0]->elapsed_ || int64_t(std::max(s.step_, s.elapsed_)) + steps > INT32_MAX || !s.mstep_ ||
          s.preserve_ || s.tensor_enabled_ != tensor || s.boundary_gathers_ ||
          (!s.recordings_.empty() && s.recording_chunk_steps_ < window))
        throw std::invalid_argument("incompatible temporal partition or insufficient halo");
      Check(cudaSetDevice(s.device_), "select temporal partition");
      if (!s.clamp_source_tails_) {
        for (int g = 0; g < 9; ++g)
          if (s.base_inputs_[s.source_offset_ + 3 * g].dims[0] &&
              int64_t(s.step_) + steps > s.base_inputs_[s.source_offset_ + 3 * g + 1].dims[1])
            throw std::invalid_argument("temporal partition exceeds source waveform");
      }
      p.device = s.device_;
      s.EnsureMovingResources(depth, slots, 0, macros, {axis, i?begin-origin:-1, i+1<count?end-origin:-1});
      cudaStream_t st = s.stream_.value;
      Check(cudaMemsetAsync(s.moving_workspace_.bad, 0, sizeof(int), st),
            "clear temporal partition diagnostic");
      p.low = std::make_unique<Event>();
      p.high = std::make_unique<Event>();
      p.exchanged = std::make_unique<Event>();
      p.transfer = std::make_unique<Stream>();
      s.BindOutputs();
      auto attrs = s.attributes_;
      attrs.nsteps = window;
      BeamzNativePrepare(attrs, s.inputs_.data(), s.outputs_.data(), s.source_offset_, s.monitor_offset_,
                         s.workspace_count_, false, &p.launch);
      if (tensor) p.launch.program.e_ab.tensor = &s.tensor_;
      const auto& plan = s.moving_plan_;
      const auto& slab = s.moving_macro_slab_;
      // At version v a tile touches coordinates around tile*owned-v.
      // Include the staggered neighbor and a guard tile on either side.
      const int tile = plan.owned[axis];
      int transverse = 0;
      for (int a = 0; a < 3; ++a)
        if (a != axis) transverse += plan.counts[a] - 1;
      p.low_done = std::clamp((begin - origin + window + depth + 3) / tile + 2 + transverse +
                                  (macros - 1) * slab.macro_level_stride,
                              1, slab.level_count);
      p.high_needed = std::clamp((end - origin - window - 3) / tile - 1, 0, slab.level_count);
    }
    for (int i = 1; i < count; ++i) EnablePeerPair(ranks[i - 1]->device_, ranks[i]->device_);
    auto faces = [&](int from, int to, int cut, bool from_lower, std::vector<Face>& descriptors) {
      size_t bytes = 0;
      auto& s = *ranks[from];
      auto& d = *ranks[to];
      const int n = s.attributes_.cpml_enabled ? 18 : 6;
      if (n != (d.attributes_.cpml_enabled ? 18 : 6))
        throw std::invalid_argument("inconsistent partition CPML");
      auto append = [&](const BeamzBuffer& src, const BeamzBuffer& dst, int a, int b, int width) {
        if (width <= 0 || !src.dims[0] || !src.dims[1] || !src.dims[2]) return;
        if (src.rank != 3 || dst.rank != 3 || (src.element_type != kBeamzF32 && src.element_type != kBeamzBF16) ||
            dst.element_type != src.element_type || src.compact_slots || dst.compact_slots || a < 0 || b < 0 ||
            a + width > src.dims[axis] || b + width > dst.dims[axis])
          throw std::invalid_argument("unsupported temporal exchange buffer");
        for (int k = 0; k < 3; ++k)
          if (k != axis && src.dims[k] != dst.dims[k])
            throw std::invalid_argument("cross-section mismatch");
        const int element_bytes = src.element_type == kBeamzBF16 ? 2 : 4;
        cudaMemcpy3DPeerParms q{};
        q.srcDevice = s.device_;
        q.dstDevice = d.device_;
        q.srcPtr = {src.data, size_t(element_bytes * src.row_stride), size_t(element_bytes * src.dims[2]),
                    size_t(src.plane_stride / src.row_stride)};
        q.dstPtr = {dst.data, size_t(element_bytes * dst.row_stride), size_t(element_bytes * dst.dims[2]),
                    size_t(dst.plane_stride / dst.row_stride)};
        q.srcPos = {size_t(axis == 2 ? element_bytes * a : 0), size_t(axis == 1 ? a : 0), size_t(axis == 0 ? a : 0)};
        q.dstPos = {size_t(axis == 2 ? element_bytes * b : 0), size_t(axis == 1 ? b : 0), size_t(axis == 0 ? b : 0)};
        q.extent = {size_t(element_bytes * (axis == 2 ? width : src.dims[2])), size_t(axis == 1 ? width : src.dims[1]),
                    size_t(axis == 0 ? width : src.dims[0])};
        descriptors.push_back({q, bytes});
        bytes += q.extent.width * q.extent.height * q.extent.depth;
      };
      for (int j = 0; j < n; ++j) {
        const int field = j < 6 ? j : (j - 6) / 2;
        const bool axial = field % 3 == 2 - axis;
        const int width = tensor ? exchange_width : j < 6 ? window - (from_lower ? (field < 3 && axial) : (field < 3 || axial))
                               : window - (from_lower ? (field >= 3 || axial) : 1 + (field >= 3 && axial));
        const int first = from_lower ? cut - width : cut;
        const int source_origin = spans[from][0], destination_origin = spans[to][0];
        const auto& src = s.outputs_[j];
        const auto& dst = d.outputs_[j];
        if (j < 6 || beamz::cuda::moving::kCpmlAxes[(j - 6) % 6] != axis) {
          append(src, dst, first - source_origin, first - destination_origin, width);
        } else if (!(s.attributes_.boundary_code >> 8)) {
          // Normal-axis CPML history is packed into two physical intervals.
          // A thick absorber can cross an internal cut, so exchange each real
          // intersection using its packed offsets on the two owners.
          const auto sw = s.moving_workspace_.cpml_widths[j - 6];
          const auto dw = d.moving_workspace_.cpml_widths[j - 6];
          const int starts[] = {source_origin,
              source_origin + int(s.outputs_[field].dims[axis]) - sw.high};
          const int targets[] = {destination_origin,
              destination_origin + int(d.outputs_[field].dims[axis]) - dw.high};
          const int source_widths[] = {sw.low, sw.high};
          const int destination_widths[] = {dw.low, dw.high};
          for (int side = 0; side < 2; ++side) {
            const int lo = std::max({first, starts[side], targets[side]});
            const int hi = std::min({first + width, starts[side] + source_widths[side], targets[side] + destination_widths[side]});
            append(src, dst, lo - starts[side] + (side ? sw.low : 0),
                   lo - targets[side] + (side ? dw.low : 0), hi - lo);
          }
        }
      }
      return bytes;
    };
    for (int i = 0; i + 1 < count; ++i) {
      auto& p = parts[i];
      const int a = ranks[i]->device_, b = ranks[i + 1]->device_;
      const size_t up = faces(i, i + 1, spans[i][2], true, p.up);
      const size_t down = faces(i + 1, i, spans[i][2], false, p.down);
      if (axis == 0 && !HasDirectExchange(a,b) && !p.up.empty() && p.up.size()==p.down.size() &&
          std::all_of(p.up.begin(),p.up.end(),Face::Linear) && std::all_of(p.down.begin(),p.down.end(),Face::Linear)) {
        const auto aligned = [](const Face& f) {
          return f.Bytes() % sizeof(unsigned) == 0 &&
              f.region.srcPtr.pitch * f.region.srcPtr.ysize % sizeof(unsigned) == 0 &&
              f.region.dstPtr.pitch * f.region.dstPtr.ysize % sizeof(unsigned) == 0;
        };
        p.peer_transport=std::make_unique<TemporalPeerTransport>(
            a,b,(packet_links.empty() || packet_links[i]) &&
                ranks[i]->allow_peer_packets_ && ranks[i+1]->allow_peer_packets_ &&
                std::all_of(p.up.begin(), p.up.end(), aligned) &&
                std::all_of(p.down.begin(), p.down.end(), aligned));
      } else if (axis != 0 && !HasDirectExchange(a, b)) {
        Check(cudaSetDevice(a), "select upper staging device");
        p.upper_stage = std::make_unique<Stage>(std::max(up, down));
        Check(cudaSetDevice(b), "select lower staging device");
        parts[i + 1].lower_stage = std::make_unique<Stage>(std::max(up, down));
      }
    }
    auto direct_copy = [](const std::vector<Face>& faces, cudaStream_t stream) {
      for (const Face& f : faces) {
        const auto& q = f.region;
        if (q.srcDevice == q.dstDevice) {
          cudaMemcpy3DParms local{};
          local.srcPtr = q.srcPtr;
          local.dstPtr = q.dstPtr;
          local.srcPos = q.srcPos;
          local.dstPos = q.dstPos;
          local.extent = q.extent;
          local.kind = cudaMemcpyDeviceToDevice;
          Check(cudaMemcpy3DAsync(&local, stream), "copy local temporal overlap");
        } else
          Check(cudaMemcpy3DPeerAsync(&q, stream), "copy peer temporal overlap");
      }
    };
    auto local_stage = [](const std::vector<Face>& faces, Stage& stage, bool pack) {
      for (const Face& f : faces) {
        const auto& q = f.region;
        auto* ptr = static_cast<char*>(pack ? stage.send.data : stage.receive.data) + f.offset;
        cudaPitchedPtr linear{ptr, q.extent.width, q.extent.width, q.extent.height};
        cudaMemcpy3DParms local{};
        local.extent = q.extent;
        local.kind = cudaMemcpyDeviceToDevice;
        if (pack) {
          local.srcPtr = q.srcPtr;
          local.srcPos = q.srcPos;
          local.dstPtr = linear;
        } else {
          local.srcPtr = linear;
          local.dstPtr = q.dstPtr;
          local.dstPos = q.dstPos;
        }
        Check(cudaMemcpy3DAsync(&local, stage.stream.value), "stage temporal halo");
      }
    };
    auto exchange = [&]() {
      for (int i = 0; i + 1 < count; ++i) {
        auto& p = parts[i];
        Check(cudaSetDevice(ranks[i]->device_), "select temporal exchange device");
        auto st = p.transfer->value;
        Check(cudaStreamWaitEvent(st, p.high->value, 0), "wait for upper producer");
        Check(cudaStreamWaitEvent(st, parts[i + 1].low->value, 0), "wait for lower producer");
        if (p.peer_transport) {
          p.peer_transport->Exchange(p.up,p.down,*p.high,*parts[i+1].low,p.device,st);
        } else if (!p.upper_stage) {
          direct_copy(p.up, st);
          direct_copy(p.down, st);
        } else {
          auto& left = *p.upper_stage;
          auto& right = *parts[i + 1].lower_stage;
          // Stage strided faces on each GPU. The non-peer PCIe route then
          // transports one contiguous allocation per direction.
          auto send = [&](Stage& src, Stage& dst, const std::vector<Face>& fs) {
            Check(cudaSetDevice(src.device), "select sending GPU");
            auto stream = src.stream.value;
            Check(cudaStreamWaitEvent(stream, p.high->value, 0), "wait for upper producer");
            Check(cudaStreamWaitEvent(stream, parts[i + 1].low->value, 0), "wait for lower producer");
            local_stage(fs, src, true);
            const auto& last = fs.back();
            auto e = last.region.extent;
            Check(cudaMemcpyPeerAsync(dst.receive.data, dst.device, src.send.data, src.device,
                                      last.offset + e.width * e.height * e.depth, stream),
                  "transfer staged temporal halo");
            Check(cudaEventRecord(src.sent.value, stream), "publish staged halo");
          };
          auto receive = [&](Stage& src, Stage& dst, const std::vector<Face>& fs) {
            Check(cudaSetDevice(dst.device), "select receiving GPU");
            Check(cudaStreamWaitEvent(dst.stream.value, src.sent.value, 0), "wait for staged halo");
            local_stage(fs, dst, false);
            Check(cudaEventRecord(dst.unpacked.value, dst.stream.value), "publish unpacked halo");
          };
          send(left, right, p.up);
          send(right, left, p.down);
          receive(left, right, p.up);
          receive(right, left, p.down);
          Check(cudaStreamWaitEvent(st, left.unpacked.value, 0), "join upper stage");
          Check(cudaStreamWaitEvent(st, right.unpacked.value, 0), "join lower stage");
        }
        Check(cudaEventRecord(p.exchanged->value, st), "publish temporal exchange");
      }
    };
    auto started = std::chrono::steady_clock::now();
    nb::gil_scoped_release release;
    const int initial_step = ranks[0]->step_;
    try {
      for (int offset = 0; offset <= steps - window; offset += window) {
        for (int i = 0; i < count; ++i) {
          auto& s = *ranks[i];
          auto& p = parts[i];
          const auto& slab = s.moving_macro_slab_;
          const auto st = s.stream_.value;
          Check(cudaSetDevice(s.device_), "enqueue temporal partition");
          std::vector<int> cuts = {0, slab.level_count};
          const int low = i && asynchronous && !s.rolling_band_ ? p.low_done : slab.level_count;
          const int high = i + 1 < count && asynchronous && !s.rolling_band_ ? p.high_needed : 0;
          if (i) cuts.push_back(low);
          if (i + 1 < count) cuts.push_back(high);
          std::sort(cuts.begin(), cuts.end());
          cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
          for (size_t j = 0; j + 1 < cuts.size(); ++j) {
            const int first = cuts[j], last = cuts[j + 1];
            if (offset && i && !first)
              Check(cudaStreamWaitEvent(st, parts[i - 1].exchanged->value, 0),
                    "wait for lower temporal face");
            if (offset && i + 1 < count && first == high)
              Check(cudaStreamWaitEvent(st, p.exchanged->value, 0), "wait for upper temporal face");
            const auto enqueue = [&] {
              if(s.rolling_band_){
                for(int m=0;m<macros;++m){
                  int e=BeamzEnqueueNativeAsymmetric(st,p.launch.program,s.other_.data(),m*depth,s.moving_workspace_);
                  if(e)return static_cast<cudaError_t>(e);
                  if(slab.observation_end){
                    e=s.EnqueueMovingDft(p.launch.program,m*depth,0);if(e)return static_cast<cudaError_t>(e);
                    e=s.EnqueueMovingRecorders(m*depth,0);if(e)return static_cast<cudaError_t>(e);
                  }
                }
                return cudaSuccess;
              }
              int cur = first;
              for (int m = 0; slab.observation_end && m < macros; ++m) {
                int stop = slab.observation_end + m * slab.macro_level_stride;
                if (stop <= first || stop > last) continue;
                int e = BeamzEnqueueNativeMacroSlab(st, p.launch.program, s.other_.data(), 0,
                                                    s.moving_workspace_, cur, stop, cur == 0);
                if (e) return static_cast<cudaError_t>(e);
                const int raw_step = (m % slab.observation_slots) * depth;
                e = s.EnqueueMovingDft(p.launch.program, m * depth, raw_step);
                if (e) return static_cast<cudaError_t>(e);
                e = s.EnqueueMovingRecorders(m * depth, raw_step);
                if (e) return static_cast<cudaError_t>(e);
                cur = stop;
              }
              if (cur < last)
                return static_cast<cudaError_t>(BeamzEnqueueNativeMacroSlab(
                    st, p.launch.program, s.other_.data(), 0, s.moving_workspace_, cur, last, cur == 0));
              return cudaSuccess;
            };
            std::string key = "distributed-frontier";
            key.append(reinterpret_cast<const char*>(&s.moving_plan_id_), sizeof(s.moving_plan_id_));
            key.append(reinterpret_cast<const char*>(&s.current_[0].data), sizeof(s.current_[0].data));
            for (const int value : {first, last})
              key.append(reinterpret_cast<const char*>(&value), sizeof(value));
            key += s.RecordingKey(window);
            Check(s.moving_direct_submission_
                      ? enqueue()
                      : BeamzLaunchGraph(st, key, (s.attributes_.cuda_flags & kBeamzGraphCache) != 0,
                                         s.attributes_.graph_cache_capacity, enqueue),
                  "advance temporal frontier");
            if (i && last == low) Check(cudaEventRecord(p.low->value, st), "publish lower temporal face");
          }
          Check(cudaEventRecord(p.high->value, st), "publish upper temporal face");
          s.CopyRecordings(window);
          s.UpdateClocks(window);
          s.step_ += window;
        }
        exchange();
      }
      // Drain neighbor transfers before the unblocked tail or a subsequent call.
      const int tail = steps % window;
      for (int i = 0; i < count; ++i) {
        auto& s = *ranks[i];
        Check(cudaSetDevice(s.device_), "select temporal tail");
        const auto st = s.stream_.value;
        if (steps >= window && i)
          Check(cudaStreamWaitEvent(st, parts[i - 1].exchanged->value, 0), "drain lower temporal face");
        if (steps >= window && i + 1 < count)
          Check(cudaStreamWaitEvent(st, parts[i].exchanged->value, 0), "drain upper temporal face");
        if (tail) {
          const auto& program = parts[i].launch.program;
          const BeamzNativeStepEffects effects{&NativeSession::EnqueueEffects, &s, {}};
          for (int step = 0; step < tail; ++step)
            Check(static_cast<cudaError_t>((s.recordings_.empty() && s.scatters_.empty())
                    ? BeamzEnqueueProgramStepAt(st, program, step)
                    : BeamzEnqueueProgramStepAtWithEffects(st, program, step, effects)),
                  "advance temporal tail");
          s.CopyRecordings(tail);
          s.UpdateClocks(tail);
          Check(cudaEventRecord(parts[i].low->value, st), "publish lower tail");
          Check(cudaEventRecord(parts[i].high->value, st), "publish upper tail");
        }
      }
      if (tail) exchange();
      for (int i = 0; i < count; ++i) {
        auto& s = *ranks[i];
        Check(cudaSetDevice(s.device_), "select final temporal partition");
        if (tail && i)
          Check(cudaStreamWaitEvent(s.stream_.value, parts[i - 1].exchanged->value, 0),
                "retire lower halo");
        if (tail && i + 1 < count)
          Check(cudaStreamWaitEvent(s.stream_.value, parts[i].exchanged->value, 0), "retire upper halo");
        Check(cudaStreamSynchronize(s.stream_.value), "complete temporal partition");
        int bad = 0;
        Check(cudaMemcpy(&bad, s.moving_workspace_.bad, sizeof(bad), cudaMemcpyDeviceToHost),
              "check temporal partition");
        if (bad) throw std::runtime_error("temporal partition ownership check failed");
      }
    } catch (...) {
      for (auto* s : ranks) {
        s->step_ = initial_step;
        if (cudaSetDevice(s->device_) == cudaSuccess) cudaDeviceSynchronize();
      }
      throw;
    }
    for (auto* s : ranks) {
      s->step_ = initial_step + steps;
      s->elapsed_ += steps;
      s->last_nsteps_ = steps;
      ++s->runs_;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  }
#endif
  nb::dict RunMoving(const std::vector<int>&requested_chunks,int depth,int raw_slots,int scheduling_sms){
    if(raw_slots<2||raw_slots>16)throw std::invalid_argument("raw bank count must be2..16");
#if !BEAMZ_CUDA_NATIVE_MOVING
    (void)requested_chunks;(void)depth;throw std::invalid_argument("native moving support not compiled");
#else
    auto guard=Guard();RequireOpen();
    if(!beamz::cuda::moving::ValidDepth(depth))throw std::invalid_argument("temporal depth must be a multiple of four from 4 to 32");
    // A reusable graph advances one complete slab. Recording has its own staging limit.
    const int chunk_capacity=recordings_.empty()?kSlabMacros*depth:recording_chunk_steps_;
    auto chunks=BeamzAlignNativeMovingChunks(requested_chunks,depth,chunk_capacity);
    const int64_t total=std::accumulate(chunks.begin(),chunks.end(),int64_t{0});
    const int first_step=preserve_?initial_step_:step_,first_elapsed=preserve_?initial_elapsed_:elapsed_;
    if(int64_t(first_step)+total>INT32_MAX||int64_t(first_elapsed)+total>INT32_MAX)throw std::invalid_argument("moving continuation exceeds int32 clocks");
    const bool volume_boundaries=MovingVolumesAtBoundaries(depth);
    if(volume_boundaries)chunks=MovingVolumeChunks(first_step,first_step+int(total),depth,chunk_capacity);
    if (!clamp_source_tails_) {
      for (int group = 0; group < 9; ++group) {
        const int offset = source_offset_ + 3 * group;
        if (base_inputs_[offset].dims[0] && int64_t(first_step) + total > base_inputs_[offset + 1].dims[1])
          throw std::invalid_argument("moving exceeds prepared source waveforms");
      }
    }
    // A half-filled slab would otherwise launch independent temporal blocks.
    // Freeze an even number of blocks so both field banks return to their
    // starting bindings. Later calls can replay the same slab any number of times.
    const int slab_steps=moving_macros_*depth;
    EnsureMovingResources(depth,raw_slots,scheduling_sms,moving_macros_);
    const auto started=std::chrono::steady_clock::now();
    float cuda_ms=0,physics_ms=0,prepare_ms=0;
    {
      nb::gil_scoped_release release;
      while(chunk_starts_.size()<chunks.size()){chunk_starts_.push_back(std::make_unique<Event>());chunk_ready_.push_back(std::make_unique<Event>());chunk_finished_.push_back(std::make_unique<Event>());}
      Check(cudaEventRecord(start_.value,stream_.value),"record moving start");
      if(preserve_){
        BindInitial();RestorePreservedState();
        step_=initial_step_;elapsed_=initial_elapsed_;
      }
      // Operational epoch/tags are deliberately NOT restored with physical
      // state, so replay/reset cannot accept a stale previous-macro producer.
      Check(cudaMemsetAsync(moving_workspace_.bad,0,sizeof(int32_t),stream_.value),"clear moving invocation diagnostic");
      for(size_t chunk=0;chunk<chunks.size();++chunk){
        if(chunk)Check(cudaEventRecord(chunk_starts_[chunk]->value,stream_.value),"record moving chunk preparation");
        Check(cudaEventRecord(chunk_ready_[chunk]->value,stream_.value),"record moving physics start");
        const int count=chunks[chunk];const auto initial_fields=current_,initial_other=other_;
        BindOutputs();auto a=attributes_;a.nsteps=count;
        BeamzNativeLaunchStorage identity;
        BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&identity);
        if(tensor_enabled_)identity.program.e_ab.tensor=&tensor_;
        // The immutable temporal plan belongs to this session. Only the
        // program bindings, output bank and recorder phase vary between calls.
        std::string key=BeamzGraphKey("native-temporal",stream_.value,identity.program);
        key.append(reinterpret_cast<const char*>(&moving_plan_id_),sizeof(moving_plan_id_));
        for(const auto& field:other_)BeamzAppendBufferKey(&key,field);
        key+=RecordingKey(count);
        const auto enqueue=[&]()->cudaError_t{
          for(int offset=0;offset+depth<=count;){
            BindOutputs();BeamzNativeLaunchStorage launch;
            BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&launch);
            if(tensor_enabled_)launch.program.e_ab.tensor=&tensor_;
            if(!rolling_band_&&count-offset>=slab_steps){
              const auto& slab=moving_macro_slab_;
              int level_begin=0;
              if(slab.observation_end){
                for(int m=0;m<slab.macro_count;++m){
                  const int level_end=slab.observation_end+m*slab.macro_level_stride;
                  int error=BeamzEnqueueNativeMacroSlab(stream_.value,launch.program,other_.data(),offset,moving_workspace_,level_begin,level_end,m==0);
                  if(error)return static_cast<cudaError_t>(error);
                  const int raw_step=(m%slab.observation_slots)*depth;
                  error=EnqueueMovingDft(launch.program,offset+m*depth,raw_step);if(error)return static_cast<cudaError_t>(error);
                  error=EnqueueMovingRecorders(offset+m*depth,raw_step);if(error)return static_cast<cudaError_t>(error);
                  level_begin=level_end;
                }
              }
              if(level_begin<slab.level_count){
                const int error=BeamzEnqueueNativeMacroSlab(stream_.value,launch.program,other_.data(),offset,moving_workspace_,level_begin,slab.level_count,level_begin==0);
                if(error)return static_cast<cudaError_t>(error);
              }
              offset+=slab_steps;
            }else{
              int error=BeamzEnqueueNativeAsymmetric(stream_.value,launch.program,other_.data(),offset,moving_workspace_);if(error)return static_cast<cudaError_t>(error);
              error=EnqueueMovingDft(launch.program,offset);if(error)return static_cast<cudaError_t>(error);
              error=EnqueueMovingRecorders(offset,0);if(error)return static_cast<cudaError_t>(error);
              std::swap(current_,other_);offset+=depth;
            }
          }
          BindOutputs();BeamzNativeLaunchStorage tail;
          BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&tail);
          if(tensor_enabled_)tail.program.e_ab.tensor=&tensor_;
          const BeamzNativeStepEffects effects{&NativeSession::EnqueueEffects,this,{}};
          for(int offset=(count/depth)*depth;offset<count;++offset){
            const int error=scatters_.empty()&&recordings_.empty()?BeamzEnqueueProgramStepAt(stream_.value,tail.program,offset):
              BeamzEnqueueProgramStepAtWithEffects(stream_.value,tail.program,offset,effects);
            if(error)return static_cast<cudaError_t>(error);
          }
          if(volume_boundaries&&count%depth==0){
            const auto error=EnqueueBoundaryVolumeFrames(tail.program,count);
            if(error!=cudaSuccess)return error;
          }
          return BeamzNativeAdvanceClocks(stream_.value,static_cast<int*>(base_inputs_[monitor_offset_+14].data),static_cast<int*>(base_inputs_[monitor_offset_+15].data),static_cast<const float*>(base_inputs_[monitor_offset_+13].data),static_cast<float*>(clock_time_->data),attributes_.dt,count);
        };
        Check(moving_direct_submission_ ? enqueue() :
              BeamzLaunchGraph(stream_.value,key,(a.cuda_flags&kBeamzGraphCache)!=0,a.graph_cache_capacity,enqueue),
              "execute native temporal schedule");
        if(!recordings_.empty())CopyRecordings(count);
        // Capture closures are skipped on cache hits. Commit field parity
        // independently of capture/replay; packed psi always remains in place.
        current_=initial_fields;other_=initial_other;
        if((count/depth)&1)std::swap(current_,other_);
        step_+=count;elapsed_+=count;
        Check(cudaEventRecord(chunk_finished_[chunk]->value,stream_.value),"record moving chunk end");
      }
      BindOutputs();last_bank_=0;last_nsteps_=static_cast<int>(total);++runs_;
      Check(cudaEventRecord(end_.value,stream_.value),"record moving end");Check(cudaEventSynchronize(end_.value),"finish moving solve");
      Check(cudaEventElapsedTime(&cuda_ms,start_.value,end_.value),"measure moving solve");
      for(size_t chunk=0;chunk<chunks.size();++chunk){float physics=0,prepare=0;
        Check(cudaEventElapsedTime(&physics,chunk_ready_[chunk]->value,chunk_finished_[chunk]->value),"measure moving chunk physics");
        Check(cudaEventElapsedTime(&prepare,chunk?chunk_starts_[chunk]->value:start_.value,chunk_ready_[chunk]->value),"measure moving preparation");physics_ms+=physics;prepare_ms+=prepare;
      }
      int32_t bad=0;Check(cudaMemcpy(&bad,moving_workspace_.bad,sizeof(bad),cudaMemcpyDeviceToHost),"read moving ownership diagnostic");
      if(bad)throw std::runtime_error("native moving producer/staging diagnostic failed: "+std::to_string(bad));
    }
    nb::dict result;result["cuda_ms"]=cuda_ms;result["physics_cuda_ms"]=physics_ms;result["prepare_cuda_ms"]=prepare_ms;
    result["host_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    result["nsteps"]=static_cast<int>(total);result["chunks"]=nb::cast(chunks);result["requested_chunks"]=nb::cast(requested_chunks);result["output_bank"]=0;
    result["physical_field_bank"]=current_[0].data==base_inputs_[0].data?0:1;
    result["preserve_inputs"]=preserve_;result["initialize_scratch"]=false;result["scratch_zero_bytes"]=0;
    result["preserving_copy_bytes"]=preserve_?preserve_bytes_:0;result["algorithm"]="asymmetric_moving";result["moving_depth"]=depth;
    int macros=0,tails=0;for(int count:chunks){macros+=count/depth;tails+=count%depth;}
    result["moving_macro_count"]=macros;result["ordinary_tail_steps"]=tails;
    result["physics_timing_scope"]="moving_rank_kernels_sources_raw_observations_DFT_tails_and_clock_advance";
    result["diagnostic_read_after_cuda_timer"]=true;return result;
#endif
  }

  nb::dict RunDeep4(const std::vector<int>& chunks) {
    (void) chunks;
    throw std::runtime_error("deep D4 is not compiled into this native component");
  }

  nb::dict Clocks() {
    auto guard = Guard();
    RequireOpen();
    int step = 0, elapsed = 0;
    float time = 0, origin = 0;
    Check(cudaMemcpyAsync(&origin, base_inputs_[monitor_offset_ + 13].data,
                          sizeof(float), cudaMemcpyDeviceToHost, stream_.value), "download observation origin");
    Check(cudaMemcpyAsync(&step, base_inputs_[monitor_offset_ + 14].data,
                          sizeof(int), cudaMemcpyDeviceToHost, stream_.value), "download step");
    Check(cudaMemcpyAsync(&elapsed, base_inputs_[monitor_offset_ + 15].data,
                          sizeof(int), cudaMemcpyDeviceToHost, stream_.value), "download elapsed");
    Check(cudaMemcpyAsync(&time, clock_time_->data, sizeof(float),
                          cudaMemcpyDeviceToHost, stream_.value), "download time");
    Check(cudaStreamSynchronize(stream_.value), "finish clock download");
    nb::dict result;
    result["current_step"] = step;
    result["elapsed_steps"] = elapsed;
    result["time"] = time;
    result["t"] = time;
    result["observation_origin"] = origin;
    return result;
  }

  nb::dict Metadata() const {
    auto guard = Guard();
    nb::dict result;
    result["kind"] = mstep_ ? "mstep" : generated_ ? "generated" : "program";
    result["input_count"] = host_.size();
    result["output_count"] = output_aliases_.size();
    result["resident_bytes"] = closed_ ? 0 : resident_bytes_ + preserve_bytes_;
    result["checkpoint_device_bytes"] = checkpoint_bytes_->load(std::memory_order_relaxed);
    result["allocated_payload_bytes"] = resident_bytes_ + preserve_bytes_;
    result["diagnostics_gpu_bytes"] = diagnostic_bytes_;
    result["monitor_diagnostic_gpu_bytes"] = monitor_diagnostic_bytes_;
    result["effects_gpu_bytes"] = effects_bytes_;
    result["preserve_snapshot_bytes"] = preserve_bytes_;
    result["immutable_input_alias_count"]=immutable_aliases_;
    result["immutable_gpu_alias_bytes"]=immutable_gpu_alias_bytes_;
    result["immutable_host_alias_bytes"]=immutable_host_alias_bytes_;
    size_t persistent_bytes = 0, scratch_bytes = 0, final_bytes = 0, phase_bytes = 0;
    for (int index : persistent_) persistent_bytes += host_[index].bytes;
    if (workspace_count_)
      for (int c = 0; c < 6; ++c) {
        scratch_bytes += host_[graph_count_ + 6 + c].bytes;
        final_bytes += host_[graph_count_ + c].bytes;
      }
    for (int c = 0; c < 3; ++c) phase_bytes += host_[monitor_offset_ + 10 + c].bytes;
    result["persistent_state_bytes"] = persistent_bytes;
    result["scratch_bytes"] = scratch_bytes;
    size_t deferred_bytes=0,deferred_plan_bytes=0;
    for(int index:deferred_scratch_){
      if(index>=monitor_offset_+16)deferred_plan_bytes+=host_[index].bytes;
      else deferred_bytes+=host_[index].bytes;
    }
    result["deferred_plan_bytes"]=deferred_plan_bytes;
    result["bounded_mapped_upload"]=true;
    result["deferred_scratch_bytes"]=deferred_bytes;
    size_t deferred_scratch_only=0,deferred_final_only=0;
    for(int index:deferred_scratch_){
      if(index>=graph_count_+6&&index<graph_count_+12)deferred_scratch_only+=host_[index].bytes;
      if(index>=graph_count_&&index<graph_count_+6)deferred_final_only+=host_[index].bytes;
    }
    result["deferred_scratch_only_bytes"]=deferred_scratch_only;
    result["deferred_final_workspace_bytes"]=deferred_final_only;
    result["allocated_scratch_bytes"]=scratch_bytes-deferred_scratch_only;
    result["aliased_final_workspace_bytes"]=mstep_?final_bytes:0;
    result["allocated_final_workspace_bytes"]=mstep_?0:final_bytes-deferred_final_only;
    result["final_workspace_bytes"] = final_bytes;
    result["phase_workspace_bytes"] = phase_bytes;
    result["preserving_copy_bytes_per_run"] = preserve_ ? preserve_bytes_ : 0;
    result["scratch_zero_bytes_per_run"] = 0;
    result["final_phase_zero_bytes_per_run"] = 0;
    result["workspace_policy"] =
        "unused_workspace_deferred_no_scratch_fill";
    result["execution_policy"] = preserve_ ? "preserve_initial_state" : "owned_resident";
    result["preserve_inputs"] = preserve_;
    result["initialize_scratch"] = false;
    result["runs"] = runs_;
    result["last_nsteps"] = last_nsteps_;
    result["output_bank"] = last_bank_;
    result["physical_field_bank"] =
        current_[0].data == base_inputs_[0].data ? 0 : 1;
    result["closed"] = closed_;
    result["moving_compiled"]=bool(BEAMZ_CUDA_NATIVE_MOVING);
#if BEAMZ_CUDA_NATIVE_MOVING
    result["moving_parallel_policies_compiled"]=bool(BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES);result["moving_parallel_policies_active"]=moving_workspace_.parallel_policies;result["moving_ready"]=moving_ready_;result["retired_coefficient_host_bytes"]=retired_coefficient_host_bytes_;result["retired_upload_host_bytes"]=retired_upload_host_bytes_;result["moving_policies_compiled"]=bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES);result["moving_policy_host_bytes"]=(moving_plan_.policy_offsets.size()+moving_plan_.uniform_noio_counts.size()+moving_plan_.uniform_observer_counts.size())*4;result["moving_compact_noio_host_bytes"]=moving_plan_.uniform_noio_counts.size()*4;result["moving_compact_noio_tiles"]=moving_uniform_noio_;result["moving_uniform_observer_tiles"]=moving_uniform_observer_;result["moving_compact_observer_host_bytes"]=moving_plan_.uniform_observer_counts.size()*4;result["moving_extra_bytes"]=moving_extra_bytes_;
    result["moving_raw_ring_slots"]=moving_macro_slab_.overlap_slots;result["moving_observation_slots"]=moving_workspace_.observation.key_count?moving_macro_slab_.observation_slots:0;result["moving_observation_end"]=moving_macro_slab_.observation_end;result["moving_macro_level_stride"]=moving_macro_slab_.macro_level_stride;
    result["moving_macro_slab_count"]=moving_ready_?moving_macro_slab_.macro_count:kSlabMacros;result["moving_macro_slab_levels"]=moving_macro_slab_.level_count;
    result["moving_observation_bytes"]=moving_observation_bytes_;result["moving_direct_lookup_bytes"]=moving_lookup_bytes_;result["moving_trail_bytes"]=moving_plan_.trail_floats*4;
    result["moving_dft_frequency_blocks"]=6*moving_dft_tiles_;
    result["moving_monitor_index_bytes"]=moving_monitor_index_bytes_;result["moving_monitor_index_offsets_bytes"]=moving_monitor_index_offsets_bytes_;
    result["moving_source_presence_mask"]=moving_workspace_.source_presence_mask;
    std::array<int64_t,3> source_tiles{};
    for(size_t rank=0;rank<moving_plan_.bulk_counts.size();++rank){
     if(!moving_plan_.policy_offsets.empty())for(int policy=3;policy<=5;++policy)source_tiles[policy-3]+=moving_plan_.policy_offsets[rank*7+policy+1]-moving_plan_.policy_offsets[rank*7+policy];
     else source_tiles[2]+=moving_plan_.rank_offsets[rank+1]-moving_plan_.rank_offsets[rank]-moving_plan_.bulk_counts[rank];
    }
    result["moving_source_policy_tiles"]=nb::make_tuple(source_tiles[0],source_tiles[1],source_tiles[2]);
    result["moving_source_lowering"]="ordered_coincident_runtime";
    result["moving_geometry_lowering"]="canonical_geometry";
    nb::list source_counts;for(int c=0;c<6;++c)source_counts.append(base_inputs_[source_offset_+3*(3+c)].dims[0]);
    result["moving_source_counts"]=source_counts;
    result["moving_source_pending_barriers"]=5;
    result["moving_source_pending_shared_bytes"]=58744;result["moving_original_shared_bytes"]=51812;
    result["moving_uniform_noio_tiles"]=moving_uniform_noio_;result["moving_noio_uniform_isotropic_only"]=true;result["moving_uniform_bulk_tiles"]=moving_uniform_bulk_;result["moving_uniform_certificate_prefix_bytes"]=moving_uniform_prefix_bytes_;
    result["moving_uniform_certificate_host_scratch_bytes"]=moving_uniform_prefix_bytes_;
    result["rolling_band"]=rolling_band_;
    result["moving_depth"]=moving_workspace_.depth;result["moving_rank_count"]=moving_workspace_.rank_count;
    result["moving_epoch_policy"]="monotone_operational_epoch_not_restored_with_physics";
#endif
    result["deep_d4_compiled"] = false;
    return result;
  }

  void Close() {
    auto guard = Guard();
    if (closed_) return;
    DeviceScope device_scope;
    Check(cudaSetDevice(device_), "select native session device for close");
    Check(cudaStreamSynchronize(stream_.value), "close native session");
    shard_.reset();
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
    for(const auto&stream:moving_policy_streams_)if(stream&&stream->value)Check(cudaStreamSynchronize(stream->value),"close moving policy stream");
#endif
    BeamzReleaseStreamGraphs(device_,stream_.value);
    snapshots_.clear();
    diagnostic_allocations_.clear(); diagnostic_partials_.reset();
    diagnostic_materials_.clear(); diagnostic_bytes_ = 0;
    ClearMonitorDiagnosticsUnlocked();
#if BEAMZ_CUDA_NATIVE_MOVING
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
    moving_policy_done_.clear();moving_policy_fork_.reset();moving_policy_streams_.clear();
#endif
    moving_allocations_.clear();moving_ready_=false;
#endif
    effects_allocations_.clear();
    recordings_.clear();
    scatters_.clear();
    effects_bytes_ = 0;
    allocations_.clear();
    for(auto& host:host_){host.array=nb::object{};host.data=nullptr;}
    clock_time_.reset();
    chunk_starts_.clear(); chunk_ready_.clear(); chunk_finished_.clear();
    for (Event* event : {&start_, &prepared_, &physics_, &end_}) {
      Check(cudaEventDestroy(event->value), "destroy native event");
      event->value = nullptr;
    }
    Check(cudaStreamDestroy(stream_.value), "destroy native stream");
    stream_.value = nullptr;
    closed_ = true;
  }

 private:
  bool MovingVolumesAtBoundaries(int depth) const {
    if(boundary_gathers_)return true;
#if BEAMZ_CUDA_NATIVE_MOVING
    double density=0;
    for(const auto& record:recordings_)
      if(!record.plan.indices.data)density+=1.0/record.plan.interval;
    return RecordVolumesAtBoundaries(density,depth);
#else
    return false;
#endif
  }

  std::vector<int> MovingVolumeChunks(int first,int last,int depth,int capacity) const {
    std::vector<int> chunks;
    for(int step=first;step<last;){
      int next=last;
      for(const auto& record:recordings_){
        if(record.plan.indices.data && !boundary_gathers_)continue;
        const int64_t frame=int64_t(step)/record.plan.interval+1;
        if(frame<=record.capacity)next=int(std::min(int64_t(next),frame*record.plan.interval));
      }
      const auto span=BeamzAlignNativeMovingChunks({next-step},depth,capacity);
      chunks.insert(chunks.end(),span.begin(),span.end());
      step=next;
    }
    return chunks;
  }

  cudaError_t EnqueueBoundaryVolumeFrames(const BeamzProgramLaunch& program,int count) {
    for(const auto& record:recordings_){
      if(record.plan.indices.data && !boundary_gathers_)continue;
      const auto interval=record.plan.interval;
      const int end=step_+count;
      if(end%interval||end/interval>record.capacity)continue;
      const auto error=BeamzEnqueueNativeRecorder(stream_.value,program,record.plan,
                                                 end/interval-step_/interval-1);
      if(error!=cudaSuccess)return error;
    }
    return cudaSuccess;
  }

  int RecordingCount(const NativeRecording& record, int nsteps) const {
    const int interval = record.plan.interval;
    const int start = step_ / interval;
    return int(std::max(int64_t{0}, std::min(record.capacity,
        int64_t((step_ + nsteps) / interval)) - start));
  }

  std::string RecordingKey(int nsteps) const {
    // Freeze only relative sample positions into a graph. Ordinary chunks
    // with no acquisition keep the normal graph-cache entry.
    std::string recorder_key;
    bool recording = false;
    for (const auto& record : recordings_) {
      const int count = RecordingCount(record, nsteps);
      const int phase = count ? step_ % record.plan.interval : 0;
      recorder_key.append(reinterpret_cast<const char*>(&count), sizeof(count));
      recorder_key.append(reinterpret_cast<const char*>(&phase), sizeof(phase));
      recording |= count != 0;
    }
    return recording?recorder_key:std::string{};
  }

  int LaunchOrdinary(const BeamzProgramLaunch& program, int nsteps) {
    const auto recorder_key=RecordingKey(nsteps);
    if (scatters_.empty() && recorder_key.empty()) {
      return generated_ ? BeamzLaunchTracedProgram(stream_.value, program)
                        : BeamzLaunchProgram(stream_.value, program);
    }
    const BeamzNativeStepEffects effects{
        &NativeSession::EnqueueEffects, this, "native-effects-v1" + recorder_key};
    if (generated_)
      return BeamzLaunchTracedProgramEffects(stream_.value, program, &effects);
    return BeamzLaunchProgramEffects(stream_.value, program, &effects);
  }

  static cudaError_t EnqueueEffects(cudaStream_t stream,
                                     const BeamzProgramLaunch& program,
                                     int stage, int offset, const void* context) {
    const auto& owner = *static_cast<const NativeSession*>(context);
    if (stage < 3) {
      for (const auto& scatter : owner.scatters_) if (scatter.timing == stage) {
        const auto error = BeamzEnqueueNativeScatter(stream, program, scatter, offset);
        if (error != cudaSuccess) return error;
      }
    } else {
      const int step = owner.step_ + offset + 1;
      for (const auto& record : owner.recordings_) {
        const int interval = record.plan.interval;
        if (step % interval || step / interval > record.capacity) continue;
        const int frame = step / interval - owner.step_ / interval - 1;
        const auto error = BeamzEnqueueNativeRecorder(stream, program, record.plan, frame);
        if (error != cudaSuccess) return error;
      }
    }
    return cudaSuccess;
  }

  void CopyRecordings(int nsteps) {
    for (const auto& record : recordings_) {
      const int count = RecordingCount(record, nsteps);
      if (!count) continue;
      const size_t points = record.plan.frames.dims[1];
      Check(cudaMemcpyAsync(record.history + size_t(step_ / record.plan.interval) * points,
                            record.plan.frames.data, size_t(count) * points * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_.value), "drain native recorder frames");
    }
  }

  bool tensor_enabled_ = false, boundary_gathers_ = false;
  int recording_chunk_steps_ = 256;
  BeamzTensorLaunch tensor_{};
  std::vector<NativeRecording> recordings_;
  std::vector<BeamzNativeScatter> scatters_;
  std::vector<std::unique_ptr<DeviceMemory>> effects_allocations_;
  size_t effects_bytes_ = 0;

// Included in NativeSession's private section by the private candidate patcher.
#if BEAMZ_CUDA_NATIVE_MOVING
  void RetireUploadedHostArrays(){
    const size_t electric=attributes_.cpml_enabled?kCpmlEInputOffset:12;
    const bool pending=mstep_&&!moving_ready_;
    std::set<PyObject*> owners,coefficients;
    for(size_t i=0;i<host_.size();++i){
      auto& host=host_[i];
      if(!host.array.is_valid())continue;
      const bool material=i>=electric&&i<electric+6;
      const bool cpml=attributes_.cpml_enabled&&(i==12||i==electric+6);
      const bool source_start=i>=size_t(source_offset_)&&i<size_t(monitor_offset_)&&(i-source_offset_)%3==2;
      if(((!preserve_&&persistent_.count(i))||deferred_scratch_.count(i))&&!zero_reset_.count(i))continue;
      if(cpml||(i==size_t(monitor_offset_)&&pending))continue;
      if(pending&&((material&&!rolling_band_)||(i>=6&&i<12&&host.descriptor.rank==0)||source_start||i==size_t(monitor_offset_+4)))continue;
      if(owners.insert(host.array.ptr()).second)retired_upload_host_bytes_+=host.bytes;
      if(mstep_&&material&&coefficients.insert(host.array.ptr()).second)retired_coefficient_host_bytes_+=host.bytes;
      host.array=nb::object{};host.data=nullptr;
    }
  }
  void EnsureMovingResources(int depth,int raw_slots,int scheduling_sms,int macros=kSlabMacros,std::array<int,3> cut={-1,-1,-1}){
    // Acquired slices sample the completed staggered fields. They need no per-step temporal observation keys.
    std::vector<NativeRecording> empty;
    auto& moving_recordings=boundary_gathers_?empty:recordings_;
    if(moving_ready_){if(cut!=moving_cut_||macros!=moving_macro_slab_.macro_count||depth!=moving_workspace_.depth||raw_slots!=moving_macro_slab_.overlap_slots||scheduling_sms!=moving_macro_slab_.scheduling_sms)throw std::invalid_argument("moving depth and wavefront schedule are fixed per session");return;}
    // A failed upload must not leave a half-built plan on a retry.
    if(!moving_allocations_.empty()){
      Check(cudaStreamSynchronize(stream_.value),"retire incomplete moving upload");
      moving_allocations_.clear();
    }
    // Large multi-stream wavefront graphs can fault inside the driver on A100 SXM4.
    // Direct submission retains policy-stream concurrency and the same dependencies.
    cudaDeviceProp prop{};
    Check(cudaGetDeviceProperties(&prop,device_),"query temporal submission capability");
    moving_direct_submission_=std::string(prop.name)=="NVIDIA A100-SXM4-80GB";
    moving_workspace_={};moving_temporal_dft_phases_=nullptr;moving_index_offsets_=nullptr;moving_workspace_.raw_slots=raw_slots;
    if(!beamz::cuda::moving::ValidDepth(depth))throw std::invalid_argument("temporal depth must be a multiple of four from 4 to 32");
    if(tensor_enabled_&&depth!=4)throw std::invalid_argument("tensor moving execution uses four-step blocks");
    if(!mstep_||workspace_count_!=12||(attributes_.metric_kind<0||attributes_.metric_kind>2))throw std::invalid_argument("moving requires a Cartesian FP32 Mstep packet");
    int minimum[3]{INT32_MAX,INT32_MAX,INT32_MAX};
    constexpr int dz[6]={0,0,1,1,1,0},dy[6]={0,1,0,1,0,1},dx[6]={1,0,0,0,1,1};
    for(int c=0;c<6;++c)for(int a=0;a<3;++a)minimum[a]=std::min(minimum[a],int(base_inputs_[c].dims[a]));
    for(int c=0;c<6;++c){const auto&field=base_inputs_[c];const auto&final=other_[c];
      if(field.rank!=3||field.element_type!=kBeamzF32||final.rank!=3||final.element_type!=kBeamzF32)throw std::invalid_argument("moving requires canonical FP32 field descriptors");
      const int add[3]={dz[c],dy[c],dx[c]};
      for(int a=0;a<3;++a)if(field.dims[a]!=minimum[a]+add[a]||final.dims[a]!=field.dims[a])throw std::invalid_argument("moving requires canonical staggered supports");
      for(const auto*f:{&field,&final})if(f->row_stride!=f->dims[2]||f->plane_stride!=f->dims[1]*f->dims[2]||f->compact_slots)throw std::invalid_argument("moving requires dense canonical field pitches");
    }
    bool lossless=true;uint32_t source_bits[3]{};
    for(int c=0;c<3;++c){
      if(base_inputs_[6+c].rank||base_inputs_[9+c].rank){lossless=false;continue;}
      for(int index:{6+c,9+c})if(base_inputs_[index].element_type!=kBeamzF32)throw std::invalid_argument("moving requires FP32 H coefficients");
      std::memcpy(&moving_workspace_.h_decay[c],host_[6+c].data,4);std::memcpy(&moving_workspace_.h_source[c],host_[9+c].data,4);
      std::memcpy(&source_bits[c],host_[9+c].data,4);
      lossless&=moving_workspace_.h_decay[c]==1.f;
      if(!std::isfinite(moving_workspace_.h_decay[c])||!std::isfinite(moving_workspace_.h_source[c]))throw std::invalid_argument("moving requires finite H coefficients");
    }
    const size_t electric=attributes_.cpml_enabled?kCpmlEInputOffset:12;
    const bool packed_e=base_inputs_[electric+3].element_type==kBeamzS32;
    if(packed_e&&lossless&&source_bits[0]==source_bits[1]&&source_bits[1]==source_bits[2])moving_workspace_.data_flags|=beamz::cuda::moving::kLosslessIsotropicH;
    const int layout=base_inputs_[electric+3].rank;
    if(packed_e)for(int c=0;c<3;++c){const auto&table=base_inputs_[electric+c];const auto&codes=base_inputs_[electric+3+c];
      if(table.rank!=1||table.element_type!=kBeamzF32||table.dims[0]<1||table.dims[0]>INT32_MAX||codes.element_type!=kBeamzS32||codes.rank!=layout||(layout!=1&&layout!=2))throw std::invalid_argument("moving requires homogeneous byte/block packed E coefficients");
    }
    if(packed_e)moving_workspace_.data_flags|=layout==1?beamz::cuda::moving::kBytePackedE:beamz::cuda::moving::kBlockPackedE;
    for(int g=0;g<9;++g){const int index=source_offset_+3*g;const auto&coeff=base_inputs_[index];
      if(!coeff.dims[0])continue;
      if(!((attributes_.coincident_source_group_mask>>g)&1))continue;
      const auto*starts=static_cast<const int32_t*>(host_[index+2].data);
      for(int s=1;s<coeff.dims[0];++s)for(int a=0;a<3;++a)if(starts[3*s+a]!=starts[a])throw std::invalid_argument("moving coincident source starts differ");
    }
    BindOutputs();auto a=attributes_;a.nsteps=std::min(depth,attributes_.nsteps);BeamzNativeLaunchStorage validate;
    BeamzNativePrepare(a,inputs_.data(),outputs_.data(),source_offset_,monitor_offset_,workspace_count_,false,&validate);
    if(tensor_enabled_)validate.program.e_ab.tensor=&tensor_;
    Check(static_cast<cudaError_t>(BeamzValidateNativeProgram(validate.program)),"validate moving packet");
    moving_workspace_.source_presence_mask=BeamzNativeMovingSourcePresenceMask(validate.program.source_groups,validate.program.source_group_count);
    using beamz::cuda::moving::kCpmlAxes;
    moving_workspace_.cpml_widths=MovingCpmlWidths(host_.data(),attributes_.cpml_enabled);
    for(int family=0;family<2;++family){const auto&p=family?validate.program.e_ab:validate.program.h_ab;
      for(int term=0;term<p.nterms;++term){const int axis=kCpmlAxes[term];const auto&target=p.outputs[term/2];const auto&psi=p.inputs[31+term];
        const auto [low,high]=moving_workspace_.cpml_widths[6*family+term];
        const int64_t packed=int64_t(low)+high;
        if(low<0||high<0||packed>target.dims[axis]||
           (p.uniform_cpml_thickness>0&&(low!=p.uniform_cpml_thickness||high!=p.uniform_cpml_thickness)))
          throw std::invalid_argument("moving CPML metadata disagrees with the Yee supports");
        if(psi.rank!=3||(psi.element_type!=kBeamzF32&&psi.element_type!=kBeamzBF16)||
           p.outputs[3+term].element_type!=psi.element_type||p.outputs[3+term].data!=psi.data||psi.compact_slots)
          throw std::invalid_argument("moving requires in-place FP32 or BF16 packed psi");
        for(int d=0;d<3;++d)if(psi.dims[d]!=(d==axis?packed:target.dims[d]))throw std::invalid_argument("moving packed psi shape differs");
        if(!packed)continue;
        if(psi.row_stride!=psi.dims[2]||psi.plane_stride!=psi.dims[1]*psi.dims[2])throw std::invalid_argument("moving psi must be canonical packed slabs");
        for(int k=0;k<3;++k){const auto&profile=p.inputs[13+3*term+k];
          if(profile.rank!=3||profile.element_type!=kBeamzF32)throw std::invalid_argument("moving profile type/rank differs");
          for(int d=0;d<3;++d)if(profile.dims[d]!=(d==axis?packed:1))throw std::invalid_argument("moving profile dimensions differ");
        }
      }
    }
    const auto* host_counts = static_cast<const int32_t*>(host_[monitor_offset_+4].data);
    moving_dft_tiles_=0;
    if(base_inputs_[monitor_offset_+2].dims[1]>=beamz::cuda::temporal::kDftFrequencyBatch)
     for(int m=0;m<attributes_.monitor_count;++m){
      moving_dft_tiles_+=beamz::cuda::temporal::DftWorkFor(host_counts[5*m],host_counts[5*m+1]).blocks;
    }
    const auto index_rows=MovingMonitorRows(base_inputs_[monitor_offset_],base_inputs_[monitor_offset_+4],host_counts);
    auto observation=BeamzBuildNativeMovingKeyPlan(base_inputs_.data(),base_inputs_[monitor_offset_],static_cast<const int32_t*>(host_[monitor_offset_].data),host_counts,attributes_.monitor_count);
    if(attributes_.monitor_count&&observation.keys.empty())throw std::invalid_argument("moving monitor has no valid raw samples");
    const bool volume_boundaries=MovingVolumesAtBoundaries(depth);
    bool record_volume=false;
    for(const auto& record:moving_recordings){
      AppendMovingRecorderKeys(observation,base_inputs_.data(),record.plan.component,record.host_indices);
      record_volume|=!volume_boundaries&&!record.host_indices.bytes;
    }
    if(!moving_recordings.empty())SortMovingKeys(observation);
    std::array<int,2> source_z{};
    {
    const int32_t*starts[9]{};for(int g=0;g<9;++g)starts[g]=static_cast<const int32_t*>(host_[source_offset_+3*g+2].data);
    // Only synthetic shard walls can discard their private normal histories.
    // A physical identity profile must still clear restored nonzero CPML state.
    moving_plan_=BeamzBuildNativeMovingPlan(base_inputs_.data(),depth,{},validate.groups,starts,moving_workspace_.cpml_widths,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES),moving_source_cells_,record_volume,tensor_enabled_,observation.keys,neutral_faces_ & SyntheticFaces(cut));
    if(rolling_band_)source_z=MovingSourceZSpan(moving_source_cells_,validate.groups,starts);
    }
    std::vector<beamz::cuda::moving::Scatter> moving_scatters;
    moving_scatters.reserve(scatters_.size());
    for(int group=0;group<9;++group){
      for(const auto& s:scatters_)if(3*s.timing+s.component==group){
        moving_scatters.push_back({static_cast<const int32_t*>(s.indices.data),
          static_cast<const int32_t*>(s.ends.data),static_cast<const float*>(s.coefficients.data),
          static_cast<const float*>(s.waveform.data),int(s.indices.dims[0]),int(s.waveform.dims[0])});
      }
      moving_workspace_.scatter_offsets[group+1]=int(moving_scatters.size());
    }
    std::vector<beamz::cuda::moving::VolumeRecording> volumes;
    if(record_volume)for(int component=0;component<6;++component){
      for(const auto& record:moving_recordings)if(record.plan.component==component&&!record.host_indices.bytes)
        volumes.push_back({static_cast<float*>(record.plan.frames.data),int(record.plan.frames.dims[1]),
                           record.plan.interval,int(record.capacity),record.plan.sign});
      moving_workspace_.volume_offsets[component+1]=int(volumes.size());
    }
    size_t recorder_bytes=volumes.size()*sizeof(beamz::cuda::moving::VolumeRecording);
    for(const auto& record:moving_recordings)recorder_bytes+=record.host_indices.bytes;
    // Wavefront tiles use a frozen host palette. Rolling tiles trace the immutable device upload.
    const void*table_data[3]{},*code_data[3]{};
    for(int c=0;c<3;++c){table_data[c]=host_[electric+c].data;code_data[c]=host_[electric+3+c].data;}
    const auto certificate=packed_e&&!rolling_band_?BeamzCertifyNativeMovingUniformE(base_inputs_.data(),base_inputs_.data()+electric,base_inputs_.data()+electric+3,table_data,code_data,moving_plan_):BeamzNativeMovingUniformE{};
    const bool cached_cpml=attributes_.boundary_code==(12<<8);
    size_t certified_bulk=0,certified_noio=0,certified_observer=0;
    if(certificate.supported){
      std::copy(certificate.values.begin(),certificate.values.end(),moving_workspace_.common_e);
      for(size_t tile=0;tile<moving_plan_.tile_flags.size();++tile){
        if(!certificate.admitted[tile])continue;
        const uint8_t flags=moving_plan_.tile_flags[tile];
        if(flags==0){moving_plan_.tile_flags[tile]|=64;++certified_bulk;continue;}
        if(!(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH)||(beamz::cuda::moving::Policy(flags)<1||beamz::cuda::moving::Policy(flags)>5))continue;
        // Count only real E update trajectories, not empty high-side boxes.
        const int q[3]={int(tile/(moving_plan_.counts[1]*moving_plan_.counts[2])),int((tile/moving_plan_.counts[2])%moving_plan_.counts[1]),int(tile%moving_plan_.counts[2])};
        bool has_e_update=false;
        for(int c=0;c<3;++c){int64_t first=1,last=depth;const auto&f=base_inputs_[3+c];
          for(int axis=0;axis<3;++axis){const int64_t origin=int64_t(q[axis])*moving_plan_.owned[axis];
            first=std::max<int64_t>(first,origin-f.dims[axis]+1);
            last=std::min<int64_t>(last,origin+moving_plan_.owned[axis]-1);
          }
          has_e_update|=first<=last;
        }
        if(has_e_update){
          const int policy=beamz::cuda::moving::Policy(flags);
          if(policy==1&&!cached_cpml)continue;
          moving_plan_.tile_flags[tile]|=64;
          if(policy==1)++certified_noio;
          else if(policy==2)++certified_observer;
        }
      }
    }
    moving_uniform_bulk_=certified_bulk;moving_uniform_noio_=certified_noio;moving_uniform_observer_=certified_observer;moving_uniform_prefix_bytes_=certificate.peak_scratch_bytes;
    const bool compact_enabled=!tensor_enabled_&&bool(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH);
    BeamzCompactNativeMovingUniformTiles(moving_plan_,compact_enabled);
    size_t compact_count=0;for(int count:moving_plan_.uniform_noio_counts)compact_count+=count;
    if(BEAMZ_CUDA_NATIVE_MOVING_POLICIES&&compact_enabled&&compact_count!=certified_noio)throw std::logic_error("compact NoIO certificate count differs");
    size_t observer_count=0;for(int count:moving_plan_.uniform_observer_counts)observer_count+=count;
    if(BEAMZ_CUDA_NATIVE_MOVING_POLICIES&&compact_enabled&&observer_count!=certified_observer)throw std::logic_error("compact observer certificate count differs");
    moving_index_width_=int(BeamzMovingIndexWidth(BeamzDftNeighbors(base_inputs_[monitor_offset_]),observation.keys.size()));
    BeamzNativeRowObservation sparse;
    moving_lookup_bytes_=0;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
    if(!observation.keys.empty()){
      sparse=BeamzBuildNativeMovingRowObservation(base_inputs_.data(),observation.keys);
      BeamzRemapNativeMovingRows(base_inputs_.data(),base_inputs_[monitor_offset_],static_cast<const int32_t*>(host_[monitor_offset_].data),observation,sparse,index_rows,host_counts,attributes_.monitor_count,moving_index_width_);
      moving_lookup_bytes_=sparse.device_bytes;
    }
#endif
    moving_monitor_index_bytes_=observation.remapped_indices.size()*sizeof(int32_t);
    moving_monitor_index_offsets_bytes_=index_rows.size()*sizeof(int32_t);
    moving_cut_=cut;
    if(rolling_band_){
      auto retained = cut;
      if (retained[0] == 0) {
        // A raw DFT gather can explicitly observe an unowned overlap cell.
        // That entire side must keep the original evolution across exchanges.
        auto first = observation.keys.begin();
        for (int component = 0; component < 6; ++component) {
          const auto& field = base_inputs_[component];
          const int64_t plane = field.dims[1] * field.dims[2];
          const int64_t base = observation.field_base[component];
          const auto end = std::lower_bound(first, observation.keys.end(), base + field.dims[0] * plane);
          if (first != end) {
            if (retained[1] >= 0 && *first < base + retained[1] * plane) retained[1] = -1;
            if (retained[2] >= 0 && end[-1] >= base + retained[2] * plane) retained[2] = -1;
          }
          first = end;
        }
      }
      rolling_plan_=std::make_unique<beamz::cuda::rolling::Plan>(beamz::cuda::rolling::MakePlan(
          {minimum[0],minimum[1],minimum[2]},rolling_band_,retained,macros*depth,source_z));
      moving_workspace_.rolling=rolling_plan_.get();moving_workspace_.rolling_band=rolling_band_;
      moving_macro_slab_={};moving_macro_slab_.macro_count=macros;moving_macro_slab_.overlap_slots=raw_slots;
      moving_macro_slab_.observation_slots=1;moving_macro_slab_.observation_end=observation.keys.empty()?0:1;
      moving_macro_slab_.level_count=1;moving_macro_slab_.macro_level_stride=1;moving_macro_slab_.max_frontier=moving_plan_.max_frontier;
    }else moving_macro_slab_=BeamzBuildNativeMovingMacroSlab(moving_plan_,macros,raw_slots,true,observation.keys.empty()?0:scheduling_sms,cut);
    if(!rolling_band_&&!tensor_enabled_){
      moving_plan_.metadata_bytes+=beamz::cuda::moving::FaceLines(moving_plan_.counts)*sizeof(int32_t);
      BeamzColorMovingFaces(moving_plan_.slots,moving_plan_.counts,0,-1,moving_plan_.max_frontier);
      BeamzColorMovingFaces(moving_macro_slab_.slots,moving_plan_.counts,moving_macro_slab_.macro_level_stride,cut[0],moving_macro_slab_.max_frontier);
    }
    moving_macro_slab_.scheduling_sms=scheduling_sms;
    const int observation_slots=moving_macro_slab_.observation_slots;
    moving_workspace_.raw_slots=observation_slots;
    moving_observation_bytes_=moving_monitor_index_bytes_+moving_monitor_index_offsets_bytes_+size_t(depth)*observation_slots*observation.keys.size()*sizeof(float)+moving_lookup_bytes_;
    const auto& frequencies=base_inputs_[monitor_offset_+kMonitorFrequenciesInput];
    const size_t temporal_phase_bytes=size_t(depth)*frequencies.dims[0]*frequencies.dims[1]*12;
    moving_observation_bytes_+=temporal_phase_bytes;
    moving_plan_.metadata_bytes+=size_t(moving_plan_.epochs)*(moving_macro_slab_.max_frontier-moving_plan_.max_frontier)*sizeof(uint64_t);
    moving_plan_.trail_floats=beamz::cuda::moving::TrailOwners(moving_macro_slab_.max_frontier,tensor_enabled_)*((tensor_enabled_?9:6)*depth-3)*moving_plan_.surface_points;
    if(rolling_band_){
      moving_plan_.metadata_bytes=moving_plan_.tile_flags.size()+sizeof(int32_t)+rolling_plan_->mask_bytes;
      moving_plan_.trail_floats=0;
    }
    moving_extra_bytes_=moving_plan_.metadata_bytes+moving_plan_.trail_floats*sizeof(float)+moving_observation_bytes_+moving_macro_slab_.device_bytes()+moving_macro_slab_.slots.size()*4+moving_scatters.size()*sizeof(beamz::cuda::moving::Scatter)+recorder_bytes;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
    if(!BEAMZ_CUDA_NATIVE_MOVING_POLICIES)throw std::invalid_argument("parallel moving requires policy partition");
    while(moving_policy_streams_.size()<6)moving_policy_streams_.push_back(std::make_unique<Stream>());
    while(moving_policy_done_.size()<6)moving_policy_done_.push_back(std::make_unique<MovingDependencyEvent>());
    if(!moving_policy_fork_)moving_policy_fork_=std::make_unique<MovingDependencyEvent>();
    for(int policy=0;policy<6;++policy){moving_workspace_.policy_streams[policy]=moving_policy_streams_[policy]->value;moving_workspace_.policy_done_events[policy]=moving_policy_done_[policy]->value;}
    moving_workspace_.policy_fork_event=moving_policy_fork_->value;moving_workspace_.parallel_policies=true;
#endif
    size_t free,total;Check(cudaMemGetInfo(&free,&total),"query moving admission");
    if(moving_extra_bytes_+(256ULL<<20)>free)throw std::invalid_argument("moving workspace plus256MiB reserve exceeds actual free GPU memory");
    auto upload=[&](const void*data,size_t bytes,bool zero=false){auto memory=std::make_unique<DeviceMemory>(bytes);
      if(bytes){if(zero)Check(cudaMemsetAsync(memory->data,0,bytes,stream_.value),"initialize moving metadata");else if(data)Check(cudaMemcpyAsync(memory->data,data,bytes,cudaMemcpyHostToDevice,stream_.value),"upload moving metadata");}
      void*pointer=memory->data;moving_allocations_.push_back(std::move(memory));return pointer;
    };
    if(rolling_band_)moving_workspace_.rolling_masks=static_cast<uint8_t*>(upload(nullptr,rolling_plan_->mask_bytes));
    moving_temporal_dft_phases_=static_cast<float*>(upload(nullptr,temporal_phase_bytes));
    if(!moving_scatters.empty())moving_workspace_.scatters=static_cast<const beamz::cuda::moving::Scatter*>(upload(moving_scatters.data(),moving_scatters.size()*sizeof(beamz::cuda::moving::Scatter)));
    if(!volumes.empty())moving_workspace_.volumes=static_cast<const beamz::cuda::moving::VolumeRecording*>(upload(volumes.data(),volumes.size()*sizeof(beamz::cuda::moving::VolumeRecording)));
    std::vector<std::vector<int32_t>> recorder_indices(moving_recordings.size());
    moving_recorder_indices_.assign(moving_recordings.size(),{});
    for(size_t i=0;i<moving_recordings.size();++i){
      const auto& rec=moving_recordings[i];
      if(!rec.host_indices.bytes)continue;
      const int component=rec.plan.component;
      const int64_t base=observation.field_base[component];
      const auto* input=static_cast<const int32_t*>(rec.host_indices.data);
      auto& output=recorder_indices[i];output.reserve(rec.host_indices.bytes/sizeof(int32_t));
      for(size_t j=0;j<rec.host_indices.bytes/sizeof(int32_t);++j){
        const auto key=base+input[j];
        const auto& field=base_inputs_[component];
        const int64_t nx=field.dims[2],ny=field.dims[1],index=input[j];
        const int raw=BeamzMovingRawOffset(sparse,component,int(index/(nx*ny)),int((index/nx)%ny),int(index%nx));
        if(raw<0||size_t(raw)>=observation.keys.size()||observation.keys[raw]!=key)
          throw std::logic_error("moving recorder key was lost");
        output.push_back(raw);
      }
      auto& device=moving_recorder_indices_[i];device=rec.plan.indices;
      device.data=upload(output.data(),rec.host_indices.bytes);
    }
    moving_workspace_.slab_tile_ids_device=static_cast<int32_t*>(upload(moving_macro_slab_.tile_ids.data(),moving_macro_slab_.device_bytes()));
    moving_workspace_.slab_policy_offsets_host=moving_macro_slab_.policy_offsets.data();
    moving_workspace_.slab_noio_counts_host=moving_macro_slab_.uniform_noio_counts.data();moving_workspace_.slab_observer_counts_host=moving_macro_slab_.uniform_observer_counts.data();
    moving_workspace_.slab_level_count=moving_macro_slab_.level_count;
    moving_workspace_.slab_slots_device=static_cast<int32_t*>(upload(moving_macro_slab_.slots.data(),moving_macro_slab_.slots.size()*4));
    moving_workspace_.slab_max_frontier=moving_macro_slab_.max_frontier;moving_workspace_.slab_macro_stride=moving_macro_slab_.macro_level_stride;
    moving_workspace_.tile_ids_device=static_cast<int32_t*>(upload(moving_plan_.tile_ids.data(),rolling_band_?0:moving_plan_.tile_ids.size()*4));
    moving_workspace_.slots_device=static_cast<int32_t*>(upload(moving_plan_.slots.data(),rolling_band_?0:moving_plan_.slots.size()*4));
    moving_workspace_.tile_flags_device=static_cast<uint8_t*>(upload(moving_plan_.tile_flags.data(),moving_plan_.tile_flags.size()));
    moving_workspace_.trails=static_cast<float*>(upload(nullptr,moving_plan_.trail_floats*sizeof(float)));
    moving_workspace_.owner_tags=static_cast<uint64_t*>(upload(nullptr,rolling_band_?0:size_t(moving_plan_.epochs)*moving_macro_slab_.max_frontier*sizeof(uint64_t),true));
    moving_workspace_.epoch=static_cast<uint32_t*>(upload(nullptr,rolling_band_?0:4,true));moving_workspace_.bad=static_cast<int32_t*>(upload(nullptr,4,true));
    moving_workspace_.uniform_observer_counts_host=moving_plan_.uniform_observer_counts.data();
    moving_workspace_.uniform_noio_counts_host=moving_plan_.uniform_noio_counts.data();
    moving_workspace_.policy_offsets_host=moving_plan_.policy_offsets.empty()?nullptr:moving_plan_.policy_offsets.data();
    moving_workspace_.rank_offsets_host=moving_plan_.rank_offsets.data();moving_workspace_.bulk_counts_host=moving_plan_.bulk_counts.data();
    std::copy(moving_plan_.counts.begin(),moving_plan_.counts.end(),moving_workspace_.counts);std::copy(moving_plan_.owned.begin(),moving_plan_.owned.end(),moving_workspace_.owned);
    moving_workspace_.max_frontier=moving_plan_.max_frontier;moving_workspace_.rank_count=moving_plan_.bulk_counts.size();moving_workspace_.tile_count=moving_plan_.tile_ids.size();moving_workspace_.depth=depth;moving_workspace_.epochs=moving_plan_.epochs;moving_workspace_.trail_floats=moving_plan_.trail_floats;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
    moving_workspace_.observation_tiles=static_cast<const int32_t*>(upload(sparse.coarse.data(),sparse.coarse.size()*4));
    moving_workspace_.observation_leaf=static_cast<const int32_t*>(upload(sparse.leaf.data(),sparse.leaf.size()*8));
    std::copy(sparse.origin.begin(),sparse.origin.end(),moving_workspace_.observation_origin);
    std::copy(sparse.grid.begin(),sparse.grid.end(),moving_workspace_.observation_grid);
#endif
    auto&obs=moving_workspace_.observation;
    obs.keys=nullptr;
    obs.publication_words=nullptr;
    moving_remapped_=base_inputs_[monitor_offset_];moving_remapped_.data=upload(observation.remapped_indices.data(),moving_monitor_index_bytes_);
    if(!index_rows.empty())moving_index_offsets_=static_cast<const int32_t*>(upload(index_rows.data(),index_rows.size()*sizeof(int32_t)));
    obs.values=static_cast<float*>(upload(nullptr,size_t(depth)*observation_slots*observation.keys.size()*4));obs.key_count=observation.keys.size();
    std::copy(observation.field_base.begin(),observation.field_base.end(),obs.field_base);obs.mask_nz=observation.mask_shape[0];obs.mask_ny=observation.mask_shape[1];obs.mask_nx=observation.mask_shape[2];
    static std::atomic<uint64_t> next_plan_id{1};
    moving_plan_id_=next_plan_id.fetch_add(1,std::memory_order_relaxed);
    if(rolling_band_)Check(static_cast<cudaError_t>(BeamzPrepareNativeRolling(stream_.value,validate.program,other_.data(),moving_workspace_)),"compile rolling CUDA tile effects");
    Check(cudaStreamSynchronize(stream_.value),"finish moving workspace preparation");
    // Per-rank ranges stay live on the host. The device owns the tile maps.
    std::vector<int32_t>().swap(moving_plan_.tile_ids);
    std::vector<int32_t>().swap(moving_plan_.slots);
    std::vector<uint8_t>().swap(moving_plan_.tile_flags);
    std::vector<int32_t>().swap(moving_macro_slab_.tile_ids);
    std::vector<int32_t>().swap(moving_macro_slab_.slots);
    std::vector<beamz::cuda::moving::SourceCell>().swap(moving_source_cells_);
    for(auto& record:moving_recordings)record.host_indices={};
    resident_bytes_+=moving_extra_bytes_;moving_ready_=true;
    RetireUploadedHostArrays();

  }
  int EnqueueMovingDft(const BeamzProgramLaunch&program,int offset,int raw_step_offset=0){
    const auto&obs=moving_workspace_.observation;
    if(!program.monitors||!program.monitors->monitor_count)return cudaSuccess;
    auto h=program.h_ab,e=program.e_ab;auto monitors=*program.monitors;if(moving_index_width_==4)monitors.indices=moving_remapped_;
    BeamzBuffer field{};field.rank=3;field.element_type=kBeamzF32;field.dims[0]=field.dims[1]=1;field.dims[2]=obs.key_count;field.row_stride=field.plane_stride=obs.key_count;field.data=obs.values;
    for(int c=0;c<3;++c)h.outputs[c]=e.outputs[c]=field;
    return BeamzEnqueueTemporalDft(stream_.value,h,e,monitors,offset,raw_step_offset,moving_workspace_.depth,moving_temporal_dft_phases_,nullptr,moving_index_offsets_,moving_remapped_.data,moving_index_width_,moving_dft_tiles_);
  }

  int EnqueueMovingRecorders(int offset,int raw_step_offset) {
    if(boundary_gathers_)return 0;
    const auto& obs=moving_workspace_.observation;
    for(size_t i=0;i<recordings_.size();++i){
      if(!moving_recorder_indices_[i].data)continue;
      const auto& record=recordings_[i];auto plan=record.plan;
      plan.indices=moving_recorder_indices_[i];
      const int interval=plan.interval;
      const int previous=(step_+offset)/interval;
      const int last=int(std::min(record.capacity,int64_t((step_+offset+moving_workspace_.depth)/interval)));
      for(int frame=previous;frame<last;){
        ++frame;
        const int raw=raw_step_offset+frame*interval-step_-offset-1;
        BeamzBuffer field{};field.rank=3;field.element_type=kBeamzF32;
        field.dims[0]=field.dims[1]=1;field.dims[2]=obs.key_count;
        field.row_stride=field.plane_stride=obs.key_count;
        field.data=obs.values+size_t(raw)*obs.key_count;
        const auto error=BeamzEnqueueNativeRecorderFrame(stream_.value,field,plan,frame-step_/interval-1);
        if(error!=cudaSuccess)return error;
      }
    }
    return cudaSuccess;
  }

#endif

  std::unique_lock<std::mutex> Guard() const {
    std::unique_lock<std::mutex> guard(mutex_, std::defer_lock);
    // Another Python thread must not hold the GIL while waiting for a session
    // whose GPU operation is about to reacquire it to return its result.
    { nb::gil_scoped_release release; guard.lock(); }
    return guard;
  }


  void RestorePreservedState(){
    for(int index:persistent_)if(zero_reset_.count(index)&&host_[index].bytes)
      Check(cudaMemsetAsync(base_inputs_[index].data,0,host_[index].bytes,stream_.value),"restore captured zero state");
    for(const auto& [index,snapshot]:snapshots_)if(host_[index].bytes)
      Check(cudaMemcpyAsync(base_inputs_[index].data,snapshot->data,host_[index].bytes,
                            cudaMemcpyDeviceToDevice,stream_.value),"restore preserved device state");
  }
  void RequireOpen() const {
    if (closed_) throw std::runtime_error("native session is closed");
  }

  void ValidateClock(int index, int type) const {
    const auto& b = base_inputs_[index];
    if (b.rank != 0 || b.element_type != type || host_[index].bytes != 4)
      throw std::invalid_argument("native clocks must be scalar float32/int32");
  }

  void BindInitial() {
    inputs_ = base_inputs_;
    current_.assign(base_inputs_.begin(), base_inputs_.begin() + 6);
    other_.clear();
    if (workspace_count_)
      other_.assign(base_inputs_.begin() + graph_count_,
                     base_inputs_.begin() + graph_count_ + 6);
    psi_current_.clear();
    if (attributes_.cpml_enabled) {
      for (int c = 0; c < 6; ++c) psi_current_.push_back(base_inputs_[31 + c]);
      for (int c = 0; c < 6; ++c) psi_current_.push_back(base_inputs_[62 + c]);
    }
    last_bank_ = 0;
    last_nsteps_ = 0;
    BindOutputs();
  }

  void BindOutputs() {
    outputs_.clear();
    output_aliases_.clear();
    auto append = [&](int index, const BeamzBuffer& value) {
      output_aliases_.push_back(index);
      outputs_.push_back(value);
    };
    for (int c = 0; c < 6; ++c) {
      inputs_[c] = current_[c];
      append(c, current_[c]);
    }
    if (attributes_.cpml_enabled) {
      for (int c = 0; c < 6; ++c) {
        inputs_[31 + c] = psi_current_[c];
        append(31 + c, psi_current_[c]);
      }
      for (int c = 0; c < 6; ++c) {
        inputs_[62 + c] = psi_current_[6 + c];
        append(62 + c, psi_current_[6 + c]);
      }
    }
    for (int c = 0; c < workspace_count_; ++c)
      append(graph_count_ + c, c < 6 ? other_[c] : base_inputs_[graph_count_ + c]);
    for (int c = 0; c < 6; ++c)
      append(monitor_offset_ + 7 + c, base_inputs_[monitor_offset_ + 7 + c]);
  }

  void UpdateClocks(int nsteps) {
    Check(BeamzNativeAdvanceClocks(
        stream_.value, static_cast<int*>(base_inputs_[monitor_offset_ + 14].data),
        static_cast<int*>(base_inputs_[monitor_offset_ + 15].data),
        static_cast<const float*>(base_inputs_[monitor_offset_ + 13].data),
        static_cast<float*>(clock_time_->data), attributes_.dt, nsteps),
        "advance native clocks");
  }


#if BEAMZ_CUDA_NATIVE_MOVING
  size_t moving_uniform_bulk_=0,moving_uniform_noio_=0,moving_uniform_observer_=0,moving_uniform_prefix_bytes_=0;
  bool moving_ready_=false;
  size_t retired_coefficient_host_bytes_=0;
  size_t moving_extra_bytes_=0,moving_observation_bytes_=0,moving_lookup_bytes_=0,moving_monitor_index_bytes_=0,moving_monitor_index_offsets_bytes_=0;
  const int32_t*moving_index_offsets_=nullptr;
  std::array<int,3> moving_cut_{-1,-1,-1};
  BeamzNativeMovingMacroSlab moving_macro_slab_;
  BeamzNativeMovingPlan moving_plan_;
  int neutral_faces_ = 0;
  std::unique_ptr<beamz::cuda::rolling::Plan> rolling_plan_;
  bool moving_direct_submission_=false;
  BeamzNativeMovingWorkspace moving_workspace_{};
  std::vector<beamz::cuda::moving::SourceCell> moving_source_cells_;
  std::vector<BeamzBuffer> moving_recorder_indices_;
  BeamzBuffer moving_remapped_{};
  int moving_index_width_=4;
  uint64_t moving_dft_tiles_=0;
  std::vector<std::unique_ptr<DeviceMemory>> moving_allocations_;
  size_t retired_upload_host_bytes_=0;
  uint64_t moving_plan_id_=0;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
  std::vector<std::unique_ptr<Stream>>moving_policy_streams_;
  std::vector<std::unique_ptr<MovingDependencyEvent>>moving_policy_done_;
  std::unique_ptr<MovingDependencyEvent>moving_policy_fork_;
#endif
#endif
  bool allow_peer_packets_=true;
  int rolling_band_=0, moving_macros_=kSlabMacros;
  BeamzNativeAttributes attributes_;
  mutable std::mutex mutex_;
  float*moving_temporal_dft_phases_=nullptr;
  bool preserve_ = false, mstep_ = true, generated_ = false, shard_layout_ = false, closed_ = false;
  int graph_count_ = 0, state_count_ = 0, workspace_count_ = 0;
  int source_offset_ = 0, monitor_offset_ = 0;
  int initial_step_ = 0, initial_elapsed_ = 0, step_ = 0, elapsed_ = 0;
  int last_bank_ = 0, last_nsteps_ = 0;
  uint64_t runs_ = 0;
  float origin_ = 0;
  size_t resident_bytes_ = 0, preserve_bytes_ = 0;
  std::shared_ptr<std::atomic<size_t>> checkpoint_bytes_ = std::make_shared<std::atomic<size_t>>(0);
  size_t immutable_aliases_=0,immutable_gpu_alias_bytes_=0,immutable_host_alias_bytes_=0;
  nb::module_ numpy_;
  const int device_ = CurrentDevice();
  Stream stream_;
  std::unique_ptr<ShardResources> shard_;
  Event start_, prepared_, physics_, end_;
  std::vector<std::unique_ptr<Event>> chunk_starts_, chunk_ready_, chunk_finished_;
  std::vector<HostBuffer> host_;
  std::set<int>deferred_scratch_;
  std::vector<std::unique_ptr<DeviceMemory>> allocations_;
  std::unique_ptr<DeviceMemory> clock_time_;
  std::array<const double*, 3> diagnostic_interpolation_{};
  std::array<int, 3> diagnostic_ownership_{-1, 0, 0};
  size_t diagnostic_bytes_ = 0;
  double diagnostic_domain_measure_ = 0;
  std::vector<BeamzDiagnosticMaterial> diagnostic_materials_;
  std::vector<std::unique_ptr<DeviceMemory>> diagnostic_allocations_;
  std::unique_ptr<DeviceMemory> diagnostic_partials_;
  struct MonitorDiagnosticRow { size_t offset, previous; int frequencies, points; };
  std::vector<MonitorDiagnosticRow> monitor_diagnostic_rows_;
  std::unique_ptr<DeviceMemory> monitor_previous_, monitor_partials_, monitor_residual_;
  size_t monitor_diagnostic_bytes_ = 0, monitor_previous_values_ = 0;
  int monitor_real_ = 0, monitor_imag_ = 0;
  bool monitor_diagnostic_ready_ = false;
  bool clamp_source_tails_ = false;
  std::vector<std::pair<int, std::unique_ptr<DeviceMemory>>> snapshots_;
  std::set<int> persistent_;
  std::set<int> zero_reset_;
  std::vector<BeamzBuffer> base_inputs_, inputs_, outputs_, current_, other_;
  std::vector<BeamzBuffer> psi_current_, psi_other_;
  std::vector<int> output_aliases_;
};

}  // namespace

nb::dict CompiledDeviceInfo(int device){
 DeviceScope device_scope;
 if(device<0)device=device_scope.previous;
 else Check(cudaSetDevice(device),"select native device for query");
 int runtime=0,driver=0;
 cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,device),"query native properties");
 Check(cudaRuntimeGetVersion(&runtime),"query runtime version");Check(cudaDriverGetVersion(&driver),"query driver version");
 nb::dict r;r["name"]=prop.name;r["ordinal"]=device;r["major"]=prop.major;r["minor"]=prop.minor;
 r["multiprocessors"]=prop.multiProcessorCount;r["total_memory_bytes"]=prop.totalGlobalMem;
 size_t free_bytes=0,total_bytes=0;Check(cudaMemGetInfo(&free_bytes,&total_bytes),"query native memory");
 r["free_memory_bytes"]=free_bytes;r["runtime_version"]=runtime;r["driver_version"]=driver;
 std::string uuid;const char*digits="0123456789abcdef";
 for(unsigned char byte:prop.uuid.bytes){uuid+=digits[byte>>4];uuid+=digits[byte&15];}r["uuid"]=uuid;
 return r;
}

void RegisterNativeCuda(nb::module_& module) {
 module.attr("compiled_electric_coefficient_math")="cuda_fast_divide_f32";
 module.attr("compiled_tensor_diagnostics")=true;
 module.attr("compiled_coefficient_staging_bytes")=32<<20;
 module.attr("compiled_general_ordinary")=true;module.attr("compiled_immutable_aliases")=true;module.attr("compiled_compact_temporal_indices")=true;
 module.attr("compiled_clamped_source_tails") = true;
  module.attr("compiled_compact_dft_components") = true;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
  module.attr("compiled_ragged_dft_gather") = true;
#endif
 module.def("electric_coefficient_table",[](nb::object values,float dt,float eps0){
   auto numpy=nb::module_::import_("numpy");
   auto input=Describe(values,numpy,false);
   if(input.descriptor.rank!=1||input.descriptor.element_type!=kBeamzF32||input.descriptor.dims[0]<1||input.descriptor.dims[0]>INT32_MAX)
     throw std::invalid_argument("A nonempty contiguous FP32 material table is required");
   auto output=numpy.attr("empty")(nb::make_tuple(input.descriptor.dims[0]),numpy.attr("float32"));
   auto destination=Describe(output,numpy,false);
   {
     nb::gil_scoped_release release;
     const size_t size=input.descriptor.dims[0],capacity=std::min(size,size_t(1)<<22);
     Stream stream;DeviceMemory source(capacity*sizeof(float)),target(capacity*sizeof(float));
     for(size_t start=0;start<size;start+=capacity){const size_t count=std::min(capacity,size-start),bytes=count*sizeof(float);
       Check(cudaMemcpyAsync(source.data,static_cast<const float*>(input.data)+start,bytes,cudaMemcpyHostToDevice,stream.value),"upload epsilon table");
       Check(BeamzEnqueueElectricCoefficients(stream.value,static_cast<const float*>(source.data),static_cast<float*>(target.data),int(count),dt,eps0),"lower electric coefficient table");
       Check(cudaMemcpyAsync(static_cast<float*>(const_cast<void*>(destination.data))+start,target.data,bytes,cudaMemcpyDeviceToHost,stream.value),"download electric coefficient table");
       Check(cudaStreamSynchronize(stream.value),"finish electric coefficient tile");
     }
   }
   return output;
 },"epsilon"_a,"dt"_a,"eps0"_a);

 module.attr("compiled_rectilinear")=true;module.attr("compiled_trail_surface_points")=beamz::cuda::moving::kTrailSurfacePoints;
 module.attr("compiled_one_field_bank")=true;
  nb::class_<CompiledZeroBuffer>(module,"ZeroBuffer")
    .def(nb::init<std::vector<int64_t>,std::string>(),"shape"_a,"dtype"_a="float32")
    .def_prop_ro("shape",[](const CompiledZeroBuffer&z){return nb::cast(z.shape);})
    .def_prop_ro("dtype",[](const CompiledZeroBuffer&z){return nb::module_::import_("numpy").attr("dtype")(z.dtype);})
    .def_prop_ro("nbytes",&CompiledZeroBuffer::nbytes)
    .def_prop_ro("size",[](const CompiledZeroBuffer&z){return z.elements;});
  module.attr("compiled_zero_descriptors")=true;

  module.def("device_info",&CompiledDeviceInfo,"ordinal"_a=-1);
  module.def("_device_count", []{
    int count = 0;
    Check(cudaGetDeviceCount(&count), "count native CUDA devices");
    return count;
  });
  module.def("_has_direct_exchange", &HasDirectExchange);
  module.def("_set_device", [](int ordinal){
    const int previous = CurrentDevice();
    if (previous != ordinal) Check(cudaSetDevice(ordinal), "select native CUDA device");
    return previous;
  });
  module.attr("compiled_depths")=nb::make_tuple(4,8,12,16,20,24,28,32);
  module.attr("compiled_program_sha256")=BEAMZ_CUDA_PHYSICS_SHA256;
  module.attr("generated_ordinary")=true;module.attr("temporal_dft")=true;module.attr("compiled_dft_frequency_batch")=8;module.attr("compiled_cta_threads")=1024;module.attr("compiled_shared_bytes")=51812;module.attr("compiled_profile_shared_bytes")=55268;module.attr("compiled_source_shared_bytes")=58744;module.attr("compiled_owned_zyx")=nb::make_tuple(7,15,31);
  module.attr("unified_temporal")=true;module.attr("moving_key_only_observations")=true;module.attr("moving_observation_row_rank")=true;
  module.attr("moving_terminal_e_pruning")=true;
  module.attr("moving_unified_sources")=true;
  module.attr("compiled_rolling_tiles")=true;
  module.attr("compiled_rolling_material_trace")=true;
  module.attr("compiled_rolling_volume_windows")=true;
  module.attr("compiled_rolling_source_bounds")=true;
  module.attr("compiled_boundary_acquisition")=true;
  module.attr("moving_block_uniform_e")=true;
 module.attr("moving_macro_slab_count")=kSlabMacros;module.attr("moving_raw_ring_slots")=3;module.attr("compiled_raw_slots")=nb::make_tuple(2,3,4,6,8,12,16);module.attr("dynamic_macro_pipeline")=true;module.attr("native_lazy_mstep_scratch")=true;
  module.attr("moving_noio_uniform_e")=true;
  module.attr("moving_observer_uniform_1024")=true;
  module.attr("moving_compact_observer_dispatch")=true;
  module.attr("moving_compact_noio_dispatch")=true;
  module.attr("moving_noio_uniform_gated")=false;
  module.attr("moving_noio_axis_bundle")=true;
  module.attr("moving_noio_uniform_1024")=true;
  module.attr("moving_noio_uniform_threads")=1024;
  module.attr("moving_noio_uniform_isotropic_only")=true;
  module.attr("__version__") = kComponentVersion;
  module.attr("__abi_version__") = kAbiVersion;
  module.attr("full_intermediates") = bool(BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES);
  module.attr("moving")=bool(BEAMZ_CUDA_NATIVE_MOVING);
#if BEAMZ_CUDA_NATIVE_MOVING
  module.attr("compiled_temporal_boundary_widths") = true;
  module.attr("temporal_peer_staging_bytes") = beamz::cuda::halo::kEndpointBytes;
  module.attr("compiled_temporal_packet_links") = true;
  module.attr("compiled_transparent_halos") = true;
  module.attr("compiled_shard_resource_cuts") = true;
  module.attr("compiled_temporal_sources") = true;
  module.attr("compiled_temporal_recorders") = true;
  module.attr("compiled_temporal_diagonal_materials") = true;
  module.attr("compiled_temporal_bf16_psi") = true;
  module.attr("compiled_temporal_tensor") = true;
  module.attr("compiled_budgeted_overlap")=true;
  module.attr("compiled_observation_spacing")=true;
  module.def("moving_resource_plan",&MovingResourcePlan,"inputs"_a,"depth"_a=16,"raw_slots"_a=3,"scatters"_a=nb::tuple(),"recorders"_a=nb::tuple(),"tensor"_a=false,"budget"_a=SIZE_MAX,"minimum_overlap"_a=0,"scheduling_sms"_a=0,"macro_slab_count"_a=kSlabMacros,"partition_cut"_a=std::array<int,3>{-1,-1,-1},"boundary_code"_a=0,"rolling_band"_a=0);
#endif
  module.attr("jax_independent") = true;
  module.attr("deep_d4") = false;
  module.attr("deep_admission_kind") = -1;
  module.def("deep_observation_plan", &DeepObservationPlan,
              "field_shapes"_a, "packed_indices"_a);
  module.attr("compiled_shard_rows") = true;
  nb::class_<NativeSession>(module, "NativeSession")
      .def(nb::init<nb::sequence, nb::dict, bool, std::string, bool>(),
           "inputs"_a, "attributes"_a, "preserve_inputs"_a=false,
           "kind"_a="mstep", "initialize_scratch"_a=true)
      .def("reset", &NativeSession::Reset)
      .def("restore_state", &NativeSession::RestoreState,
           "state_buffers"_a, "observation_origin"_a, "current_step"_a, "elapsed_steps"_a)
      .def("run", &NativeSession::Run, "nsteps"_a=-1)
      .def("run_schedule", &NativeSession::RunSchedule, "chunks"_a)
      .def("configure_effects", &NativeSession::ConfigureEffects,
           "tensor"_a, "scatters"_a, "recorders"_a, "chunk_steps"_a=256,"at_boundaries"_a=false)
      .def("rebind_recordings", &NativeSession::RebindRecordings)
      .def("_configure_shard", &NativeSession::ConfigureShard, "geometries"_a,
           "neighbors"_a = std::array<int,2>{-2,-2})
      #if BEAMZ_CUDA_NATIVE_MOVING
      .def_static("_run_temporal_partitions",&NativeSession::RunTemporalPartitions,
                  "owners"_a,"spans"_a,"axis"_a,"steps"_a,"window"_a,"depth"_a,"slots"_a,
                  "asynchronous"_a,"packet_links"_a=std::vector<bool>{})
#endif
      .def_static("_run_shards", &NativeSession::RunSharded)
      .def("configure_diagnostics", &NativeSession::ConfigureDiagnostics,
           "materials"_a, "axes"_a, "domain_measure"_a,
           "owned"_a=std::array<int,3>{-1,0,0}, "interpolation"_a=nb::tuple())
      .def("field_diagnostics", &NativeSession::FieldDiagnostics)
      .def("_configure_monitor_diagnostics", &NativeSession::ConfigureMonitorDiagnostics)
      .def("_monitor_diagnostics", &NativeSession::MonitorDiagnostics)
      .def("_clear_monitor_diagnostics", &NativeSession::ClearMonitorDiagnostics)
      .def("run_moving",&NativeSession::RunMoving,"chunks"_a,"depth"_a=16,"raw_slots"_a=3,"scheduling_sms"_a=0)
      .def("run_deep4", &NativeSession::RunDeep4, "chunks"_a)
      .def("probe", &NativeSession::Probe,"requests"_a)
      .def("download", &NativeSession::Download)
      .def("download_device_state", &NativeSession::DownloadDeviceState, "budget"_a)
      .def("download_into", &NativeSession::DownloadInto,
           "output_index"_a, "destination"_a)
      .def("clocks", &NativeSession::Clocks)
      .def("metadata", &NativeSession::Metadata)
      .def("close", &NativeSession::Close);
}
