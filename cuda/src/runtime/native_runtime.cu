#include "native_runtime.h"

namespace {
__global__ void AdvanceClocks(int* current_step, int* elapsed_steps,
                              const float* origin, float* current_time,
                              float dt, int nsteps) {
  *current_step += nsteps;
  *elapsed_steps += nsteps;
  // Match the existing state-time expression's two rounding steps. In
  // particular a nonzero origin can distinguish this from a contracted FMA.
  const float elapsed_time = __fmul_rn(static_cast<float>(*elapsed_steps), dt);
  *current_time = __fadd_rn(*origin, elapsed_time);
}
}  // namespace

cudaError_t BeamzNativeAdvanceClocks(cudaStream_t stream, int* current_step,
                                   int* elapsed_steps, const float* origin,
                                   float* current_time, float dt, int nsteps) {
  AdvanceClocks<<<1, 1, 0, stream>>>(current_step, elapsed_steps, origin,
                                    current_time, dt, nsteps);
  return cudaGetLastError();
}


#include <cmath>

namespace {

__device__ float MaterialAt(const BeamzDiagnosticMaterial& material, int index) {
  if (material.encoding == 0) return material.values[0];
  if (material.encoding == 1) return material.values[index];
  int first = 0, last = material.count;
  while (first < last) {
    const int middle = first + (last - first) / 2;
    if (material.ends[middle] <= index) first = middle + 1;
    else last = middle;
  }
  return material.values[first];
}

constexpr int kDiagnosticThreads = 256;

struct AddDftNorms {
  __device__ BeamzDftNorms operator()(BeamzDftNorms a, BeamzDftNorms b) const {
    return {a.current + b.current, a.previous + b.previous,
            a.difference + b.difference, a.nonfinite | b.nonfinite};
  }
};

__device__ BeamzDftNorms ReduceDftNorms(BeamzDftNorms local, BeamzDftNorms* shared) {
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = kDiagnosticThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] = AddDftNorms{}(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  return shared[0];
}

__global__ void DftDiagnosticTiles(const float* real, const float* imag,
    float* previous_real, float* previous_imag, int points, int tiles,
    bool compare, BeamzDftNorms* partials) {
  __shared__ BeamzDftNorms storage[kDiagnosticThreads];
  const int frequency = blockIdx.x / tiles;
  const int first = (blockIdx.x % tiles) * kBeamzDftDiagnosticTile;
  const int count = min(points - first, kBeamzDftDiagnosticTile);
  const size_t base = size_t(frequency) * points + first;
  BeamzDftNorms local{};
  for (int point = threadIdx.x; point < count; point += blockDim.x) {
    const size_t i = base + point;
    const double r = real[i], q = imag[i];
    local.nonfinite |= !isfinite(r) || !isfinite(q);
    if (compare) {
      const double prev_r = previous_real[i], prev_q = previous_imag[i];
      const double dr = r - prev_r, dq = q - prev_q;
      local.current += r * r + q * q;
      local.previous += prev_r * prev_r + prev_q * prev_q;
      local.difference += dr * dr + dq * dq;
    }
    previous_real[i] = r;
    previous_imag[i] = q;
  }
  const auto total = ReduceDftNorms(local, storage);
  if (!threadIdx.x) partials[blockIdx.x] = total;
}

__global__ void FinishDftDiagnostics(const BeamzDftNorms* partials, int tiles,
                                     BeamzDftResidual* result) {
  __shared__ BeamzDftNorms storage[kDiagnosticThreads];
  BeamzDftNorms local{};
  for (int tile = threadIdx.x; tile < tiles; tile += blockDim.x)
    local = AddDftNorms{}(local, partials[size_t(blockIdx.x) * tiles + tile]);
  const auto total = ReduceDftNorms(local, storage);
  if (!threadIdx.x) {
    if (total.nonfinite) atomicOr(&result->nonfinite, total.nonfinite);
    const double scale = fmax(total.current, total.previous);
    if (scale > 0.0 && !total.nonfinite)
      atomicMax(&result->maximum_bits,
                static_cast<unsigned long long>(__double_as_longlong(sqrt(total.difference / scale))));
  }
}

__device__ void ReduceDiagnosticBlock(BeamzDiagnosticPartial local,
                                     BeamzDiagnosticPartial* partials) {
  __shared__ BeamzDiagnosticPartial shared[kDiagnosticThreads];
  shared[threadIdx.x] = local;
  __syncthreads();
  for (int stride = kDiagnosticThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x].sum += shared[threadIdx.x + stride].sum;
      shared[threadIdx.x].maximum = fmaxf(shared[threadIdx.x].maximum,
                                         shared[threadIdx.x + stride].maximum);
      shared[threadIdx.x].nonfinite |= shared[threadIdx.x + stride].nonfinite;
    }
    __syncthreads();
  }
  if (!threadIdx.x) partials[blockIdx.x] = shared[0];
}

__global__ void FieldDiagnostics(BeamzBuffer field,
                                 BeamzDiagnosticMaterial material,
                                 double domain_measure,
                                 BeamzDiagnosticPartial* partials) {
  const int nx = int(field.dims[2]), ny = int(field.dims[1]);
  const int count = int(field.dims[0] * field.dims[1] * field.dims[2]);
  BeamzDiagnosticPartial local{0.0, 0.0f, 0};
  for (int64_t logical = int64_t(blockIdx.x) * kDiagnosticThreads + threadIdx.x;
       logical < count; logical += int64_t(gridDim.x) * kDiagnosticThreads) {
    const int index = int(logical), zy = index / nx;
    if (!material.widths[0][zy / ny] || !material.widths[1][zy % ny] ||
        !material.widths[2][index % nx]) continue;
    const float value = static_cast<const float*>(field.data)[
        BeamzPhysicalIndex(field, index)];
    local.nonfinite |= isnan(value) ? 1u : isinf(value) ? 2u : 0u;
    local.maximum = fmaxf(local.maximum, fabsf(value));
    // Match the reference's float64 volume construction, FP32 conversion,
    // and FP32 division by the domain measure. Accumulating products in double
    // avoids a volume-dependent serial FP32 accumulation error.
    const float volume = float(material.widths[0][zy / ny] *
        material.widths[1][zy % ny] * material.widths[2][index % nx]);
    const float measure = __fdiv_rn(volume, float(domain_measure));
    const float density = __fmul_rn(MaterialAt(material, index),
                                   __fmul_rn(value, value));
    local.sum += double(__fmul_rn(density, measure));
  }
  ReduceDiagnosticBlock(local, partials);
}

// An electric component is centered along exactly one physical axis. Its
// node value uses the volume-weighted adjacent samples, with one-sided bounds.
__device__ float ElectricAtNode(BeamzBuffer field, int axis, const int* point,
                                const double* widths) {
  int q[] = {point[0], point[1], point[2]};
  const int coordinate = q[axis];
  const int count = int(field.dims[axis]);
  const auto* electric = static_cast<const float*>(field.data);
  if (coordinate == 0 || coordinate == count) {
    q[axis] = min(coordinate, count - 1);
    return electric[BeamzOffset3D(field, q[0], q[1], q[2])];
  }
  const float right = electric[BeamzOffset3D(field, q[0], q[1], q[2])];
  --q[axis];
  const float left = electric[BeamzOffset3D(field, q[0], q[1], q[2])];
  const float left_width = float(widths[coordinate - 1]), right_width = float(widths[coordinate]);
  const float weight = __fdiv_rn(left_width, __fadd_rn(left_width, right_width));
  return __fadd_rn(__fmul_rn(weight, left), __fmul_rn(__fadd_rn(1, -weight), right));
}

__global__ void TensorDiagnostics(
    BeamzBuffer left, int left_axis, BeamzBuffer right, int right_axis,
    BeamzBuffer nodes, BeamzDiagnosticMaterial material, const double* left_widths,
    const double* right_widths, double domain_measure,
    BeamzDiagnosticPartial* partials) {
  const int nx = int(nodes.dims[2]), ny = int(nodes.dims[1]);
  const int count = int(nodes.dims[0] * nodes.dims[1] * nodes.dims[2]);
  BeamzDiagnosticPartial local{0.0, 0.0f, 0};
  for (int64_t logical = int64_t(blockIdx.x) * kDiagnosticThreads + threadIdx.x;
       logical < count; logical += int64_t(gridDim.x) * kDiagnosticThreads) {
    const int index = int(logical), zy = index / nx;
    const float cross = MaterialAt(material, index);
    if (cross == 0) continue;
    const int point[] = {zy / ny, zy % ny, index % nx};
    const double weight = material.widths[0][point[0]] *
        material.widths[1][point[1]] * material.widths[2][point[2]];
    if (weight == 0.) continue;
    const float a = ElectricAtNode(left, left_axis, point, left_widths);
    const float b = ElectricAtNode(right, right_axis, point, right_widths);
    const float volume = float(weight);
    const float measure = __fdiv_rn(volume, float(domain_measure));
    local.sum += double(__fmul_rn(__fmul_rn(cross, __fmul_rn(a, b)), measure));
  }
  ReduceDiagnosticBlock(local, partials);
}

}  // namespace

cudaError_t BeamzEnqueueDftDiagnostics(cudaStream_t stream, const float* real,
    const float* imag, float* previous_real, float* previous_imag,
    int frequencies, int points, bool compare, BeamzDftNorms* partials,
    BeamzDftResidual* residual) {
  const int tiles = 1 + (points - 1) / kBeamzDftDiagnosticTile;
  DftDiagnosticTiles<<<frequencies * tiles, kDiagnosticThreads, 0, stream>>>(
      real, imag, previous_real, previous_imag, points, tiles, compare, partials);
  if (const auto status = cudaPeekAtLastError(); status != cudaSuccess) return status;
  FinishDftDiagnostics<<<frequencies, kDiagnosticThreads, 0, stream>>>(partials, tiles, residual);
  return cudaPeekAtLastError();
}

cudaError_t BeamzEnqueueFieldDiagnostics(
    cudaStream_t stream, const BeamzBuffer& field,
    const BeamzDiagnosticMaterial& material, double domain_measure,
    BeamzDiagnosticPartial* partials) {
  FieldDiagnostics<<<kBeamzDiagnosticBlocks, kDiagnosticThreads, 0, stream>>>(
      field, material, domain_measure, partials);
  return cudaGetLastError();
}

cudaError_t BeamzEnqueueTensorDiagnostics(
    cudaStream_t stream, const BeamzBuffer& left, int left_axis,
    const BeamzBuffer& right, int right_axis, const BeamzBuffer& nodes,
    const BeamzDiagnosticMaterial& material, const double* left_widths,
    const double* right_widths, double domain_measure,
    BeamzDiagnosticPartial* partials) {
  TensorDiagnostics<<<kBeamzDiagnosticBlocks, kDiagnosticThreads, 0, stream>>>(
      left, left_axis, right, right_axis, nodes, material, left_widths,
      right_widths, domain_measure, partials);
  return cudaGetLastError();
}

namespace {
__global__ void RecordField(BeamzBuffer field, BeamzNativeRecorder recorder,
                            int frame) {
  const int64_t point = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t points = recorder.frames.dims[1];
  if (point >= points) return;
  const auto* values = static_cast<const float*>(field.data);
  float value = 0;
  if (recorder.indices.data) {
    const auto* indices = static_cast<const int32_t*>(recorder.indices.data);
    const auto* weights = static_cast<const float*>(recorder.weights.data);
    const int neighbors = int(recorder.indices.dims[1]);
    const int64_t start = point * neighbors;
    for (int k = 0; k < neighbors; ++k) {
      const float sample = values[BeamzPhysicalIndex(field, indices[start + k])];
      value = __fadd_rn(value, __fmul_rn(sample, weights[start + k]));
    }
  } else {
    value = values[BeamzPhysicalIndex(field, int(point))];
  }
  static_cast<float*>(recorder.frames.data)[int64_t(frame) * points + point] =
      value * recorder.sign;
}
}  // namespace

cudaError_t BeamzEnqueueNativeRecorder(cudaStream_t stream,
                                       const BeamzProgramLaunch& program,
                                       const BeamzNativeRecorder& recorder,
                                       int frame) {
  const auto& field = recorder.component < 3
      ? program.h_ab.outputs[recorder.component]
      : program.e_ab.outputs[recorder.component - 3];
  return BeamzEnqueueNativeRecorderFrame(stream,field,recorder,frame);
}

cudaError_t BeamzEnqueueNativeRecorderFrame(cudaStream_t stream,const BeamzBuffer& field,
                                            const BeamzNativeRecorder& recorder,int frame) {
  constexpr int threads = 256;
  const int blocks = int((recorder.frames.dims[1] + threads - 1) / threads);
  if (!blocks) return cudaSuccess;
  RecordField<<<blocks, threads, 0, stream>>>(field, recorder, frame);
  return cudaGetLastError();
}
