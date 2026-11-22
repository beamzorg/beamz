// Included after UpdateComponent: reuse exactly the streamed curl and CPML
// arithmetic, then apply the reciprocal shared-node constitutive correction.
template <bool Cpml, int MetricKind, bool Packed = false>
__global__ void TensorElectricUpdate(BeamzLaunch launch, BeamzTensorLaunch tensor) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  UpdateComponent<1, 0, Cpml, MetricKind, true, false, -1, Packed, false, true>(
      launch, z, y, x, &tensor);
  UpdateComponent<1, 1, Cpml, MetricKind, true, false, -1, Packed, false, true>(
      launch, z, y, x, &tensor);
  UpdateComponent<1, 2, Cpml, MetricKind, true, false, -1, Packed, false, true>(
      launch, z, y, x, &tensor);
}

template <int Component>
__device__ __forceinline__ float TensorNodeCurl(
    const BeamzTensorLaunch& tensor, int z, int y, int x) {
  constexpr int axis = 2 - Component;
  const auto& curl = tensor.curls[Component];
  int lo[3] = {z, y, x}, hi[3] = {z, y, x};
  const int node = lo[axis], count = static_cast<int>(curl.dims[axis]);
  lo[axis] = max(0, node - 1);
  hi[axis] = min(count - 1, node);
  const float a = Read3D(curl, lo[0], lo[1], lo[2]);
  const float b = Read3D(curl, hi[0], hi[1], hi[2]);
  if (node == 0 || node == count) return a;
  const auto& widths = tensor.widths[axis];
  if (widths.rank == 0) return 0.5f * (a + b);
  const int global_node =
      static_cast<const int32_t*>(tensor.origin.data)[axis] + node;
  const auto* h = static_cast<const float*>(widths.data);
  const float weight = h[global_node - 1] / (h[global_node - 1] + h[global_node]);
  return weight * a + (1.0f - weight) * b;
}

template <int Row, int Column>
__device__ __forceinline__ float TensorCrossTerm(
    const BeamzTensorLaunch& tensor, int z, int y, int x) {
  constexpr int pair = Row + Column - 1;  // xy, xz, yz
  const int offset =
      ((z * static_cast<int>(tensor.cross.dims[1]) + y) *
           static_cast<int>(tensor.cross.dims[2]) + x) * 3 + pair;
  const float coefficient = static_cast<const float*>(tensor.cross.data)[offset];
  return coefficient * TensorNodeCurl<Column>(tensor, z, y, x);
}

template <int Row>
__device__ __forceinline__ float TensorNodeCorrection(
    const BeamzTensorLaunch& tensor, int z, int y, int x) {
  if constexpr (Row == 0)
    return TensorCrossTerm<0, 1>(tensor, z, y, x) +
           TensorCrossTerm<0, 2>(tensor, z, y, x);
  if constexpr (Row == 1)
    return TensorCrossTerm<1, 0>(tensor, z, y, x) +
           TensorCrossTerm<1, 2>(tensor, z, y, x);
  if constexpr (Row == 2)
    return TensorCrossTerm<2, 0>(tensor, z, y, x) +
           TensorCrossTerm<2, 1>(tensor, z, y, x);
}

template <int Component>
__device__ __forceinline__ void AddTensorComponent(
    const BeamzLaunch& launch, const BeamzTensorLaunch& tensor, int z, int y, int x) {
  const auto& scratch = tensor.curls[Component];
  if (z >= scratch.dims[0] || y >= scratch.dims[1] || x >= scratch.dims[2]) return;
  const auto* origin = static_cast<const int32_t*>(tensor.origin.data);
  const int gz = z + origin[0], gy = y + origin[1], gx = x + origin[2];
  const auto& output = launch.outputs[Component];
  if (gz < 0 || gy < 0 || gx < 0 || gz >= output.dims[0] ||
      gy >= output.dims[1] || gx >= output.dims[2] ||
      beamz::cuda::yee::PecConstrained(
          output, 1, Component, launch.metallic_edges, gz, gy, gx)) return;
  int high[3] = {z, y, x};
  ++high[2 - Component];
  const float cross = 0.5f * (
      TensorNodeCorrection<Component>(tensor, z, y, x) +
      TensorNodeCorrection<Component>(tensor, high[0], high[1], high[2]));
  static_cast<float*>(output.data)[BeamzOffset3D(output, gz, gy, gx)] +=
      launch.dt_over_eps * cross;
}

__global__ void AddTensorCorrection(BeamzLaunch launch, BeamzTensorLaunch tensor) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  AddTensorComponent<0>(launch, tensor, z, y, x);
  AddTensorComponent<1>(launch, tensor, z, y, x);
  AddTensorComponent<2>(launch, tensor, z, y, x);
}

template <int MetricKind>
void EnqueueTensorElectric(cudaStream_t stream, const BeamzLaunch& launch,
                          dim3 blocks, dim3 threads) {
  const bool packed = launch.inputs[9].element_type == kBeamzS32;
  if (launch.nterms && packed)
    TensorElectricUpdate<true, MetricKind, true><<<blocks, threads, 0, stream>>>(
        launch, *launch.tensor);
  else if (packed)
    TensorElectricUpdate<false, MetricKind, true><<<blocks, threads, 0, stream>>>(
        launch, *launch.tensor);
  else if (launch.nterms)
    TensorElectricUpdate<true, MetricKind><<<blocks, threads, 0, stream>>>(
        launch, *launch.tensor);
  else
    TensorElectricUpdate<false, MetricKind><<<blocks, threads, 0, stream>>>(
        launch, *launch.tensor);
  const auto& cross = launch.tensor->cross;
  const dim3 correction_blocks(
      (cross.dims[2] + threads.x - 1) / threads.x,
      (cross.dims[1] + threads.y - 1) / threads.y,
      (cross.dims[0] + threads.z - 1) / threads.z);
  AddTensorCorrection<<<correction_blocks, threads, 0, stream>>>(
      launch, *launch.tensor);
}
