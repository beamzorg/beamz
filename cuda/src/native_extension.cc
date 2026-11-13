#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "native_launch.h"
#include "native_deep.h"
#include "native_runtime.h"
#include "graph.h"

#ifndef BEAMZ_CUDA_DEEP_D4
#define BEAMZ_CUDA_DEEP_D4 0
#endif

namespace nb = nanobind;
using namespace nb::literals;
using namespace beamz::cuda::abi;

namespace {

void Check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
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

struct HostBuffer {
  nb::object array;
  nb::object dtype;
  nb::tuple shape;
  BeamzBuffer descriptor{};
  const void* data = nullptr;
  size_t bytes = 0;
};

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

class NativeSession {
 public:
  NativeSession(nb::sequence inputs, nb::dict attributes, bool preserve_inputs,
                std::string kind, bool initialize_scratch)
      : attributes_(Attributes(attributes)), preserve_(preserve_inputs),
        mstep_(kind == "mstep"), initialize_scratch_(initialize_scratch),
        numpy_(nb::module_::import_("numpy")) {
    if (kind != "mstep" && kind != "program")
      throw std::invalid_argument("kind must be mstep or program");
    if (mstep_ && (attributes_.metric_kind != 0 ||
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
    persistent_.insert(monitor_offset_ + 14);
    persistent_.insert(monitor_offset_ + 15);
    host_.reserve(input_count);
    base_inputs_.reserve(input_count);
    allocations_.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
      host_.push_back(Describe(inputs[i], numpy_, persistent_.count(i) != 0 ||
          (BEAMZ_CUDA_DEEP_D4 && i == static_cast<size_t>(monitor_offset_))));
      auto memory = std::make_unique<DeviceMemory>(host_.back().bytes);
      auto descriptor = host_.back().descriptor;
      descriptor.data = memory->data;
      if (host_.back().bytes)
        Check(cudaMemcpyAsync(memory->data, host_.back().data, host_.back().bytes,
                              cudaMemcpyHostToDevice, stream_.value), "upload prepared input");
      resident_bytes_ += memory->bytes;
      allocations_.push_back(std::move(memory));
      base_inputs_.push_back(descriptor);
    }
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
        auto snapshot = std::make_unique<DeviceMemory>(host_[index].bytes);
        if (host_[index].bytes)
          Check(cudaMemcpyAsync(snapshot->data, base_inputs_[index].data,
                                host_[index].bytes, cudaMemcpyDeviceToDevice,
                                stream_.value), "create preserved device state");
        preserve_bytes_ += snapshot->bytes;
        snapshots_.emplace_back(index, std::move(snapshot));
      }
    }
    BindInitial();
    step_ = initial_step_;
    elapsed_ = initial_elapsed_;
    UpdateClocks(0);
    Check(cudaStreamSynchronize(stream_.value), "finish native preparation");
  }

  ~NativeSession() { if (!closed_) cudaStreamSynchronize(stream_.value); }

  void Reset() {
    auto guard = Guard();
    RequireOpen();
    nb::gil_scoped_release release;
    Check(cudaStreamSynchronize(stream_.value), "synchronize before reset");
    for (int index : persistent_)
      if (host_[index].bytes)
        Check(cudaMemcpyAsync(base_inputs_[index].data, host_[index].data,
                              host_[index].bytes, cudaMemcpyHostToDevice,
                              stream_.value), "reset native state");
    BindInitial();
    step_ = initial_step_;
    elapsed_ = initial_elapsed_;
    UpdateClocks(0);
    Check(cudaStreamSynchronize(stream_.value), "finish native reset");
  }

  nb::dict Run(int nsteps) {
    if (nsteps < 0) nsteps = attributes_.nsteps;
    return RunSchedule({nsteps});
  }

  nb::dict RunSchedule(const std::vector<int>& chunks) {
    auto guard = Guard();
    RequireOpen();
    if (chunks.empty()) throw std::invalid_argument("native schedule cannot be empty");
    int64_t total_steps = 0;
    for (int count : chunks) {
      if (count < 1) throw std::invalid_argument("native chunk steps must be positive");
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
      if (coefficients.dims[0] && int64_t(start_step) + total_steps > waveforms.dims[1])
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
        BindInitial();
        for (const auto& snapshot : snapshots_) {
          const int index = snapshot.first;
          if (host_[index].bytes)
            Check(cudaMemcpyAsync(base_inputs_[index].data, snapshot.second->data,
                                  host_[index].bytes, cudaMemcpyDeviceToDevice,
                                  stream_.value), "restore preserved state");
        }
        step_ = initial_step_;
        elapsed_ = initial_elapsed_;
      }
      for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        const int nsteps = chunks[chunk];
        if (chunk)
          Check(cudaEventRecord(chunk_starts_[chunk]->value, stream_.value),
                "record chunk preparation start");
        if (mstep_ && initialize_scratch_)
          for (int c = 0; c < 6; ++c) {
            const int index = graph_count_ + 6 + c;
            if (host_[index].bytes)
              Check(cudaMemsetAsync(base_inputs_[index].data, 0, host_[index].bytes,
                                    stream_.value), "initialize compact workspace");
          }
        Check(cudaEventRecord(chunk_ready_[chunk]->value, stream_.value),
              "record chunk preparation end");
        BindOutputs();
        auto a = attributes_;
        a.nsteps = nsteps;
        BeamzNativeLaunchStorage launch;
        BeamzNativePrepare(a, inputs_.data(), outputs_.data(), source_offset_,
                            monitor_offset_, workspace_count_, mstep_, &launch);
        const int error = mstep_ ? BeamzLaunchMstep(stream_.value, launch.mstep)
                                 : BeamzLaunchProgram(stream_.value, launch.program);
        Check(static_cast<cudaError_t>(error), "execute native program");
        Check(cudaEventRecord(chunk_finished_[chunk]->value, stream_.value),
              "record chunk physics end");
        UpdateClocks(nsteps);
        step_ += nsteps;
        elapsed_ += nsteps;
        last_bank_ = mstep_ ? ((nsteps / 2) & 1) : 0;
        if (last_bank_) std::swap(current_, other_);
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
      last_chunk_count_ = chunks.size();
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
    result["initialize_scratch"] = initialize_scratch_ && mstep_;
    size_t scratch_bytes = 0;
    if (mstep_ && initialize_scratch_)
      for (int c = 0; c < 6; ++c) scratch_bytes += host_[graph_count_ + 6 + c].bytes;
    result["scratch_zero_bytes"] = scratch_bytes * chunks.size();
    result["preserving_copy_bytes"] = preserve_ ? preserve_bytes_ : 0;
    return result;
  }

  nb::list Download() {
    auto guard = Guard();
    RequireOpen();
    nb::list result;
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto& host = host_[output_aliases_[i]];
      nb::object array = numpy_.attr("empty")(host.shape, "dtype"_a=host.dtype);
      const auto destination = Describe(array, numpy_, false);
      if (host.bytes)
        Check(cudaMemcpyAsync(const_cast<void*>(destination.data), outputs_[i].data,
                              host.bytes, cudaMemcpyDeviceToHost, stream_.value),
              "download native output");
      result.append(array);
    }
    Check(cudaStreamSynchronize(stream_.value), "finish native download");
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
    auto* target = static_cast<unsigned char*>(const_cast<void*>(destination.data));
    const auto* source = static_cast<const unsigned char*>(outputs_[output_index].data);
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

  nb::dict RunDeep4(const std::vector<int>& chunks) {
#if !BEAMZ_CUDA_DEEP_D4
    (void) chunks;
    throw std::runtime_error("deep D4 is not compiled into this native component");
#else
    auto guard = Guard();
    RequireOpen();
    if (chunks.empty()) throw std::invalid_argument("deep schedule cannot be empty");
    int64_t total = 0;
    for (int count : chunks) {
      if (count < 1) throw std::invalid_argument("deep chunks must be positive");
      total += count;
      if (total > INT32_MAX) throw std::invalid_argument("deep schedule exceeds int32");
    }
    const int first_step = preserve_ ? initial_step_ : step_;
    const int first_elapsed = preserve_ ? initial_elapsed_ : elapsed_;
    if (int64_t(first_step) + total > INT32_MAX ||
        int64_t(first_elapsed) + total > INT32_MAX)
      throw std::invalid_argument("deep continuation exceeds int32 clocks");
    for (int group = 0; group < 9; ++group)
      if (base_inputs_[source_offset_ + 3 * group].dims[0] &&
          first_step + total > base_inputs_[source_offset_ + 3 * group + 1].dims[1])
        throw std::invalid_argument("deep schedule exceeds prepared source waveforms");
    EnsureDeepResources();
    const auto started = std::chrono::steady_clock::now();
    float cuda_ms = 0, physics_ms = 0, prepare_ms = 0;
    {
      nb::gil_scoped_release release;
      while (chunk_starts_.size() < chunks.size()) {
        chunk_starts_.push_back(std::make_unique<Event>());
        chunk_ready_.push_back(std::make_unique<Event>());
        chunk_finished_.push_back(std::make_unique<Event>());
      }
      Check(cudaEventRecord(start_.value, stream_.value), "record deep start");
      if (preserve_) {
        BindInitial();
        for (const auto& snapshot : snapshots_)
          if (host_[snapshot.first].bytes)
            Check(cudaMemcpyAsync(base_inputs_[snapshot.first].data, snapshot.second->data,
                                  host_[snapshot.first].bytes, cudaMemcpyDeviceToDevice,
                                  stream_.value), "restore deep initial state");
        step_ = initial_step_;
        elapsed_ = initial_elapsed_;
      }
      for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        if (chunk)
          Check(cudaEventRecord(chunk_starts_[chunk]->value, stream_.value),
                "record deep chunk preparation");
        Check(cudaEventRecord(chunk_ready_[chunk]->value, stream_.value),
              "record deep chunk start");
        const int count = chunks[chunk];
        const auto initial_fields = current_, initial_other = other_;
        const auto initial_psi = psi_current_, initial_psi_other = psi_other_;
        BindOutputs();
        auto a = attributes_;
        a.nsteps = count;
        BeamzNativeLaunchStorage identity;
        BeamzNativePrepare(a, inputs_.data(), outputs_.data(), source_offset_,
                            monitor_offset_, workspace_count_, false, &identity);
        std::string key = BeamzGraphKey("native-deep-d4-v1", stream_.value,
                                        identity.program);
        for (const auto& field : other_) BeamzAppendBufferKey(&key, field);
        for (const auto& psi : psi_other_) BeamzAppendBufferKey(&key, psi);
        BeamzAppendBufferKey(&key, deep_remapped_);
        for (const auto* pointer : {deep_keys_->data, deep_words_->data,
                                    deep_values_->data, clock_time_->data})
          key.append(reinterpret_cast<const char*>(&pointer), sizeof(pointer));
        const auto enqueue = [&]() {
          for (int offset = 0; offset + 3 < count; offset += 4) {
            BindOutputs();
            BeamzNativeLaunchStorage launch;
            BeamzNativePrepare(a, inputs_.data(), outputs_.data(), source_offset_,
                                monitor_offset_, workspace_count_, false, &launch);
            for (int c = 0; c < 6; ++c) {
              launch.program.h_ab.outputs[3 + c] = psi_other_[c];
              launch.program.e_ab.outputs[3 + c] = psi_other_[6 + c];
            }
            int error = BeamzEnqueueDeepWarpD4(stream_.value, launch.program,
                                               other_.data(), offset, deep_observation_);
            if (error) return static_cast<cudaError_t>(error);
            error = BeamzEnqueueDeepWarpDft4(stream_.value, launch.program,
                                             offset, deep_observation_, deep_remapped_);
            if (error) return static_cast<cudaError_t>(error);
            std::swap(current_, other_);
            std::swap(psi_current_, psi_other_);
          }
          BindOutputs();
          BeamzNativeLaunchStorage tail;
          BeamzNativePrepare(a, inputs_.data(), outputs_.data(), source_offset_,
                              monitor_offset_, workspace_count_, false, &tail);
          for (int offset = (count / 4) * 4; offset < count; ++offset) {
            const int error = BeamzEnqueueProgramStepAt(stream_.value,
                                                        tail.program, offset);
            if (error) return static_cast<cudaError_t>(error);
          }
          return BeamzNativeAdvanceClocks(
              stream_.value, static_cast<int*>(base_inputs_[monitor_offset_ + 14].data),
              static_cast<int*>(base_inputs_[monitor_offset_ + 15].data),
              static_cast<const float*>(base_inputs_[monitor_offset_ + 13].data),
              static_cast<float*>(clock_time_->data), attributes_.dt, count);
        };
        Check(BeamzLaunchGraph(stream_.value, key,
                               (a.cuda_flags & kBeamzGraphCache) != 0,
                               a.graph_cache_capacity, enqueue), "execute deep graph");
        // Cache hits skip the enqueue closure. Commit the same bank parity
        // independently of whether this call captured or replayed its graph.
        current_ = initial_fields;
        other_ = initial_other;
        psi_current_ = initial_psi;
        psi_other_ = initial_psi_other;
        if ((count / 4) & 1) {
          std::swap(current_, other_);
          std::swap(psi_current_, psi_other_);
        }
        step_ += count;
        elapsed_ += count;
        Check(cudaEventRecord(chunk_finished_[chunk]->value, stream_.value),
              "record deep chunk end");
      }
      BindOutputs();
      last_bank_ = 0;
      last_nsteps_ = static_cast<int>(total);
      last_chunk_count_ = chunks.size();
      ++runs_;
      Check(cudaEventRecord(end_.value, stream_.value), "record deep end");
      Check(cudaEventSynchronize(end_.value), "synchronize deep result");
      Check(cudaEventElapsedTime(&cuda_ms, start_.value, end_.value), "measure deep time");
      for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        float physics = 0, preparation = 0;
        Check(cudaEventElapsedTime(&physics, chunk_ready_[chunk]->value,
                                    chunk_finished_[chunk]->value), "measure deep chunk");
        Check(cudaEventElapsedTime(&preparation,
                                    chunk ? chunk_starts_[chunk]->value : start_.value,
                                    chunk_ready_[chunk]->value), "measure deep preparation");
        physics_ms += physics;
        prepare_ms += preparation;
      }
    }
    nb::dict result;
    result["cuda_ms"] = cuda_ms;
    result["physics_cuda_ms"] = physics_ms;
    result["prepare_cuda_ms"] = prepare_ms;
    result["host_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    result["nsteps"] = static_cast<int>(total);
    result["chunks"] = nb::cast(chunks);
    result["output_bank"] = 0;
    result["physical_field_bank"] = current_[0].data == base_inputs_[0].data ? 0 : 1;
    result["preserve_inputs"] = preserve_;
    result["initialize_scratch"] = false;
    result["scratch_zero_bytes"] = 0;
    result["preserving_copy_bytes"] = preserve_ ? preserve_bytes_ : 0;
    result["algorithm"] = "deep4_experimental";
    result["clock_kernel_inside_physics_interval"] = true;
    return result;
#endif
  }

  nb::dict Clocks() {
    auto guard = Guard();
    RequireOpen();
    int step = 0, elapsed = 0;
    float time = 0;
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
    result["observation_origin"] = origin_;
    return result;
  }

  nb::dict Metadata() const {
    auto guard = Guard();
    nb::dict result;
    result["kind"] = mstep_ ? "mstep" : "program";
    result["input_count"] = host_.size();
    result["output_count"] = output_aliases_.size();
    result["resident_bytes"] = closed_ ? 0 : resident_bytes_ + preserve_bytes_;
    result["allocated_payload_bytes"] = resident_bytes_ + preserve_bytes_;
    result["preserve_snapshot_bytes"] = preserve_bytes_;
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
    result["final_workspace_bytes"] = final_bytes;
    result["phase_workspace_bytes"] = phase_bytes;
    result["preserving_copy_bytes_per_run"] = preserve_ ? persistent_bytes : 0;
    result["scratch_zero_bytes_per_run"] =
        mstep_ && initialize_scratch_ ? scratch_bytes * last_chunk_count_ : 0;
    result["final_phase_zero_bytes_per_run"] = 0;
    result["workspace_policy"] =
        "scratch_zero_per_chunk_final_and_phase_reused_without_fill";
    result["execution_policy"] = preserve_ ? "preserve_initial_state" : "owned_resident";
    result["preserve_inputs"] = preserve_;
    result["initialize_scratch"] = initialize_scratch_ && mstep_;
    result["runs"] = runs_;
    result["last_nsteps"] = last_nsteps_;
    result["output_bank"] = last_bank_;
    result["physical_field_bank"] =
        current_[0].data == base_inputs_[0].data ? 0 : 1;
    result["closed"] = closed_;
    result["deep_d4_compiled"] = bool(BEAMZ_CUDA_DEEP_D4);
#if BEAMZ_CUDA_DEEP_D4
    result["deep_ready"] = deep_ready_;
    result["deep_admission_kind"] = BeamzDeepD4AdmissionKind();
    result["deep_psi_shadow_bytes"] = deep_psi_bytes_;
    result["deep_observation_bytes"] = deep_observation_bytes_;
    result["deep_observation_key_count"] = deep_observation_.key_count;
#endif
    return result;
  }

  void Close() {
    auto guard = Guard();
    if (closed_) return;
    Check(cudaStreamSynchronize(stream_.value), "close native session");
    snapshots_.clear();
#if BEAMZ_CUDA_DEEP_D4
    deep_shadows_.clear();
    deep_keys_.reset(); deep_words_.reset(); deep_indices_.reset(); deep_values_.reset();
    deep_ready_ = false;
#endif
    allocations_.clear();
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
  std::unique_lock<std::mutex> Guard() const {
    std::unique_lock<std::mutex> guard(mutex_, std::defer_lock);
    // Another Python thread must not hold the GIL while waiting for a session
    // whose GPU operation is about to reacquire it to return its result.
    { nb::gil_scoped_release release; guard.lock(); }
    return guard;
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
#if BEAMZ_CUDA_DEEP_D4
    psi_other_ = deep_shadow_descriptors_;
#endif
    last_bank_ = 0;
    last_nsteps_ = 0;
    last_chunk_count_ = 0;
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

#if BEAMZ_CUDA_DEEP_D4
  void EnsureDeepResources() {
    if (deep_ready_) return;
    if (!mstep_ || workspace_count_ != 12 || !attributes_.cpml_enabled ||
        attributes_.metric_kind != 0 || attributes_.boundary_code != (12 << 8))
      throw std::invalid_argument("deep D4 requires a Cartesian FP32 CPML12 mstep packet without PEC");
    const int admission = BeamzDeepD4AdmissionKind();
    if (admission != 0 && admission != 1)
      throw std::invalid_argument("unknown compiled deep D4 admission contract");
    if (admission == 1) {
      int signature = 0;
      for (int c = 0; c < 6; ++c)
        if (base_inputs_[source_offset_ + 3 * (3 + c)].dims[0] > 0)
          signature |= 1 << c;
      if (signature != 20)
        throw std::invalid_argument(
            "typed deep D4 requires source signature 20 (Hz and Ey only); received " +
            std::to_string(signature));
      for (int c = 0; c < 6; ++c)
        if (base_inputs_[6 + c].rank != 0 ||
            base_inputs_[6 + c].element_type != kBeamzF32)
          throw std::invalid_argument("typed deep D4 requires all H coefficients to be scalar float32");
      for (int c = 0; c < 3; ++c) {
        const auto& table = base_inputs_[37 + c];
        const auto& codes = base_inputs_[40 + c];
        if (table.rank != 1 || table.element_type != kBeamzF32 ||
            table.dims[0] < 1 || table.dims[0] > INT32_MAX)
          throw std::invalid_argument("typed deep D4 requires positive rank1 float32 E codebooks with int32 offsets");
        if (codes.rank != 1 || codes.element_type != kBeamzS32)
          throw std::invalid_argument("typed deep D4 requires rank1 byte-packed E codes; compressed rank2 codes are unsupported");
        int64_t count = 1;
        for (int axis = 0; axis < 3; ++axis) {
          const int64_t dimension = base_inputs_[3 + c].dims[axis];
          if (dimension < 1 || dimension > INT32_MAX || count > INT32_MAX / dimension)
            throw std::invalid_argument("typed deep D4 requires positive E field volumes within int32 offsets");
          count *= dimension;
        }
      }
    }
    BindOutputs();
    auto a = attributes_;
    a.nsteps = 4;
    BeamzNativeLaunchStorage validate;
    BeamzNativePrepare(a, inputs_.data(), outputs_.data(), source_offset_,
                        monitor_offset_, workspace_count_, false, &validate);
    Check(static_cast<cudaError_t>(BeamzValidateNativeProgram(validate.program)),
          "validate deep native program");
    for (int c = 0; c < 6; ++c) {
      if (base_inputs_[6 + c].rank != 0)
        throw std::invalid_argument("deep D4 currently requires scalar H coefficients");
      if (base_inputs_[c].element_type != kBeamzF32)
        throw std::invalid_argument("deep D4 currently requires FP32 fields");
      for (int axis = 0; axis < 3; ++axis)
        if (base_inputs_[c].dims[axis] <= 32)
          throw std::invalid_argument("deep D4 currently requires each field extent above32");
    }
    for (int c = 0; c < 3; ++c) {
      const auto& material = validate.program.e_ab.inputs[9 + c];
      if (material.element_type != kBeamzS32 || (material.rank != 1 && material.rank != 2))
        throw std::invalid_argument("deep D4 currently requires lossless packed E coefficients");
      if (validate.groups[c].coefficients.dims[0])
        throw std::invalid_argument("deep D4 does not support pre-E sources");
    }
    for (const auto& psi : psi_current_)
      if (psi.element_type != kBeamzF32)
        throw std::invalid_argument("deep D4 does not support BF16 auxiliary state");
    const auto plan = BeamzBuildNativeDeepPlan(
        base_inputs_.data(), base_inputs_[monitor_offset_],
        static_cast<const int32_t*>(host_[monitor_offset_].data));
    if (attributes_.monitor_count && plan.keys.empty())
      throw std::invalid_argument("deep D4 requires at least one valid observed cell");
    // A failed allocation/upload may leave a partial preparation. Retrying it
    // releases that partial state rather than appending a second shadow bank.
    if (!deep_shadows_.empty() || deep_keys_ || deep_values_) {
      Check(cudaStreamSynchronize(stream_.value), "synchronize partial deep setup");
      deep_shadows_.clear(); deep_shadow_descriptors_.clear();
      deep_keys_.reset(); deep_words_.reset(); deep_indices_.reset(); deep_values_.reset();
      resident_bytes_ -= deep_psi_bytes_ + deep_observation_bytes_;
      deep_psi_bytes_ = deep_observation_bytes_ = 0;
    }
    for (int c = 0; c < 12; ++c) {
      const int index = c < 6 ? 31 + c : 62 + c - 6;
      auto memory = std::make_unique<DeviceMemory>(host_[index].bytes);
      auto descriptor = base_inputs_[index];
      descriptor.data = memory->data;
      if (descriptor.data == psi_current_[c].data)
        throw std::runtime_error("deep psi input and output banks alias");
      resident_bytes_ += memory->bytes;
      deep_psi_bytes_ += memory->bytes;
      deep_shadow_descriptors_.push_back(descriptor);
      deep_shadows_.push_back(std::move(memory));
    }
    psi_other_ = deep_shadow_descriptors_;
    auto upload = [&](const void* data, size_t bytes) {
      auto memory = std::make_unique<DeviceMemory>(bytes);
      if (bytes)
        Check(cudaMemcpyAsync(memory->data, data, bytes, cudaMemcpyHostToDevice,
                              stream_.value), "upload deep observation plan");
      resident_bytes_ += memory->bytes;
      deep_observation_bytes_ += memory->bytes;
      return memory;
    };
    deep_keys_ = upload(plan.keys.data(), plan.keys.size() * sizeof(int64_t));
    deep_words_ = upload(plan.publication_words.data(), plan.publication_words.size() * sizeof(uint32_t));
    deep_indices_ = upload(plan.remapped_indices.data(), plan.remapped_indices.size() * sizeof(int32_t));
    deep_values_ = std::make_unique<DeviceMemory>(4 * plan.keys.size() * sizeof(float));
    resident_bytes_ += deep_values_->bytes;
    deep_observation_bytes_ += deep_values_->bytes;
    deep_remapped_ = base_inputs_[monitor_offset_];
    deep_remapped_.data = deep_indices_->data;
    deep_observation_.keys = static_cast<const int64_t*>(deep_keys_->data);
    deep_observation_.publication_words = static_cast<const uint32_t*>(deep_words_->data);
    deep_observation_.values = static_cast<float*>(deep_values_->data);
    std::copy(plan.field_base.begin(), plan.field_base.end(), deep_observation_.field_base);
    deep_observation_.key_count = static_cast<int>(plan.keys.size());
    deep_observation_.mask_nz = plan.mask_shape[0];
    deep_observation_.mask_ny = plan.mask_shape[1];
    deep_observation_.mask_nx = plan.mask_shape[2];
    Check(cudaStreamSynchronize(stream_.value), "finish deep preparation");
    deep_ready_ = true;
  }
#endif

  BeamzNativeAttributes attributes_;
  mutable std::mutex mutex_;
  bool preserve_ = false, mstep_ = true, initialize_scratch_ = true, closed_ = false;
  int graph_count_ = 0, state_count_ = 0, workspace_count_ = 0;
  int source_offset_ = 0, monitor_offset_ = 0;
  int initial_step_ = 0, initial_elapsed_ = 0, step_ = 0, elapsed_ = 0;
  int last_bank_ = 0, last_nsteps_ = 0;
  uint64_t runs_ = 0;
  size_t last_chunk_count_ = 0;
  float origin_ = 0;
  size_t resident_bytes_ = 0, preserve_bytes_ = 0;
  nb::module_ numpy_;
  Stream stream_;
  Event start_, prepared_, physics_, end_;
  std::vector<std::unique_ptr<Event>> chunk_starts_, chunk_ready_, chunk_finished_;
  std::vector<HostBuffer> host_;
  std::vector<std::unique_ptr<DeviceMemory>> allocations_;
  std::unique_ptr<DeviceMemory> clock_time_;
  std::vector<std::pair<int, std::unique_ptr<DeviceMemory>>> snapshots_;
  std::set<int> persistent_;
  std::vector<BeamzBuffer> base_inputs_, inputs_, outputs_, current_, other_;
  std::vector<BeamzBuffer> psi_current_, psi_other_;
  std::vector<int> output_aliases_;
#if BEAMZ_CUDA_DEEP_D4
  bool deep_ready_ = false;
  size_t deep_psi_bytes_ = 0, deep_observation_bytes_ = 0;
  std::vector<std::unique_ptr<DeviceMemory>> deep_shadows_;
  std::vector<BeamzBuffer> deep_shadow_descriptors_;
  std::unique_ptr<DeviceMemory> deep_keys_, deep_words_, deep_indices_, deep_values_;
  BeamzBuffer deep_remapped_{};
  beamz::cuda::deep::Observation deep_observation_{};
#endif
};

}  // namespace

NB_MODULE(_cuda_native, module) {
  module.attr("__version__") = kComponentVersion;
  module.attr("__abi_version__") = kAbiVersion;
  module.attr("full_intermediates") = bool(BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES);
  module.attr("jax_independent") = true;
  module.attr("deep_d4") = bool(BEAMZ_CUDA_DEEP_D4);
#if BEAMZ_CUDA_DEEP_D4
  module.attr("deep_admission_kind") = BeamzDeepD4AdmissionKind();
#else
  module.attr("deep_admission_kind") = -1;
#endif
  module.def("deep_observation_plan", &DeepObservationPlan,
              "field_shapes"_a, "packed_indices"_a);
  nb::class_<NativeSession>(module, "NativeSession")
      .def(nb::init<nb::sequence, nb::dict, bool, std::string, bool>(),
           "inputs"_a, "attributes"_a, "preserve_inputs"_a=false,
           "kind"_a="mstep", "initialize_scratch"_a=true)
      .def("reset", &NativeSession::Reset)
      .def("run", &NativeSession::Run, "nsteps"_a=-1)
      .def("run_schedule", &NativeSession::RunSchedule, "chunks"_a)
      .def("run_deep4", &NativeSession::RunDeep4, "chunks"_a)
      .def("download", &NativeSession::Download)
      .def("download_into", &NativeSession::DownloadInto,
           "output_index"_a, "destination"_a)
      .def("clocks", &NativeSession::Clocks)
      .def("metadata", &NativeSession::Metadata)
      .def("close", &NativeSession::Close);
}
