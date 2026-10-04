// Standalone shared-memory sanitizer exercise; numerical parity lives in
// pytest.
// nvcc -std=c++17 -arch=sm_86 -Icuda/src cuda/tests/cpml_core_sanitizer.cu \
//   cuda/src/update.cu -o cpml_core_sanitizer
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "kernels.h"

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

struct Buffers {
  std::vector<void *> allocations;
  ~Buffers() {
    for (void *data : allocations)
      Check(cudaFree(data));
  }
  BeamzBuffer Field(std::initializer_list<int64_t> dims, bool padded) {
    auto it = dims.begin();
    const int64_t z = it[0], y = it[1], x = it[2];
    if (!padded)
      return Make(dims);
    auto value = Make({z, ((y + 7) / 8) * 8, ((x + 63) / 64) * 64});
    value.row_stride = value.dims[2];
    value.plane_stride = value.dims[1] * value.dims[2];
    value.dims[1] = y;
    value.dims[2] = x;
    return value;
  }
  BeamzBuffer Make(std::initializer_list<int64_t> dims,
                   BeamzElementType type = kBeamzF32) {
    BeamzBuffer result{};
    result.rank = dims.size();
    result.element_type = type;
    size_t count = 1;
    int axis = 0;
    for (int64_t dim : dims) {
      result.dims[axis++] = dim;
      count *= dim;
    }
    if (result.rank == 3) {
      result.row_stride = result.dims[2];
      result.plane_stride = result.dims[1] * result.dims[2];
    }
    const size_t bytes=count*(type==kBeamzBF16?2:4);
    Check(cudaMalloc(&result.data, bytes));
    Check(cudaMemset(result.data, 0, bytes));
    allocations.push_back(result.data);
    if (type == kBeamzF32) {
      std::vector<float> values(count, 0.01f);
      Check(cudaMemcpy(result.data, values.data(), count * 4,
                       cudaMemcpyHostToDevice));
    }
    if(type==kBeamzBF16) {
      std::vector<__nv_bfloat16> values(count,__float2bfloat16_rn(.01f));
      Check(cudaMemcpy(result.data,values.data(),bytes,cudaMemcpyHostToDevice));
    }
    return result;
  }
};

void Exercise(int nz, int ny, int nx, bool packed, bool padded) {
  Buffers buffers;
  BeamzLaunch h{}, e{};
  h.phase = 0;
  e.phase = 1;
  h.uniform_cpml_thickness = e.uniform_cpml_thickness = 12;
  h.inv_resolution = e.inv_resolution = 1.f;
  for (int c = 0; c < 3; ++c) {
    int64_t z = nz + (c == 2), y = ny + (c == 1), x = nx + (c == 0);
    h.inputs[c] = buffers.Field({z, y, x}, padded);
    h.outputs[c] = buffers.Field({z, y, x}, padded);
    e.inputs[3 + c] = h.outputs[c];
    z = nz + (c != 2);
    y = ny + (c != 1);
    x = nx + (c != 0);
    e.inputs[c] = buffers.Field({z, y, x}, padded);
    e.outputs[c] = buffers.Field({z, y, x}, padded);
    h.inputs[3 + c] = e.inputs[c];
    h.inputs[6 + c] =
        buffers.Make({nz + (c == 2), ny + (c == 1), nx + (c == 0)});
    h.inputs[9 + c] = buffers.Make({});
    e.inputs[6 + c] = packed ? buffers.Make({1}) : buffers.Make({z, y, x});
    e.inputs[9 + c] = packed ? buffers.Make({(z * y * x + 3) / 4}, kBeamzS32)
                             : buffers.Make({z, y, x});
  }
  BeamzSourceGroupLaunch sources[9]{};
  for (int c = 0; c < 3; ++c) {
    auto &group = sources[3 + c];
    group.coefficients = buffers.Make({2, 7, 8, 12});
    group.waveforms = buffers.Make({2, 3});
    group.starts = buffers.Make({2, 3}, kBeamzS32);
    group.current_step = buffers.Make({}, kBeamzS32);
    const int starts[] = {13, 13, 13, 14, 14, 14};
    Check(cudaMemcpy(group.starts.data, starts, sizeof(starts),
                     cudaMemcpyHostToDevice));
  }
  for (int tile = 0; tile < 3; ++tile) {
    for (int step = 0; step < 3; ++step) {
      Check(BeamzEnqueueCpmlCore(nullptr, h, e, sources, step, tile));
      Check(cudaDeviceSynchronize());
    }
  }
  BeamzBuffer final[6];
  for (int c = 0; c < 6; ++c) {
    const auto &original = c < 3 ? h.outputs[c] : e.outputs[c - 3];
    final[c] = buffers.Field(
        {original.dims[0], original.dims[1], original.dims[2]}, padded);
    if (c < 3)
      sources[6 + c] = sources[3 + c];
  }
  auto publication =
      buffers.Make({nz + 1, (ny + 8) / 8, (nx + 16) / 16}, kBeamzS32);
  Check(cudaMemset(publication.data, 1,
                   publication.dims[0] * publication.dims[1] *
                       publication.dims[2] * sizeof(int)));
  for (int tile = 0; tile < 3; ++tile) {
    Check(BeamzEnqueueCpmlPairCore(nullptr, h, e, final, sources, 0, {}, tile));
    Check(BeamzEnqueueCpmlPairCore(nullptr, h, e, final, sources, 0,
                                   publication, tile));
    Check(cudaDeviceSynchronize());
  }
  h.nterms = e.nterms = 6;
  constexpr int axes[] = {1,0,0,2,2,1};
  for (const auto psi_type : {kBeamzF32,kBeamzBF16}) {
  for (auto *phase : {&h, &e}) {
    for (int t=0;t<6;++t) {
      const int axis=axes[t];
      int64_t profile[3]={1,1,1}; profile[axis]=24;
      for (int q=0;q<3;++q)
        phase->inputs[13+3*t+q]=buffers.Make({profile[0],profile[1],profile[2]});
      const auto &out=phase->outputs[t/2];
      int64_t psi[3]={out.dims[0],out.dims[1],out.dims[2]}; psi[axis]=24;
      phase->inputs[31+t]=buffers.Make({psi[0],psi[1],psi[2]},psi_type);
      phase->outputs[3+t]=buffers.Make({psi[0],psi[1],psi[2]},psi_type);
    }
  }
  for(int tile=0;tile<3;++tile) {
    Check(BeamzEnqueueTemporalCpml(nullptr,h,e,final,sources,0,publication,tile));
    Check(cudaDeviceSynchronize());
  }
  Check(BeamzEnqueueSpatialCpml(nullptr, h, e, sources, 0));
  Check(cudaDeviceSynchronize());
  }
  Check(BeamzEnqueueCpmlPairBand(nullptr, h, e, sources, 0));
  Check(cudaDeviceSynchronize());
}

int main(int argc, char **argv) {
  const bool small = argc > 1;
  for (bool packed : {false, true}) {
    for (bool padded : {false, true}) {
      if (small) {
        Exercise(37, 41, 61, packed, padded);
      } else {
        Exercise(48, 56, 80, packed, padded);
        Exercise(61, 73, 97, packed, padded);
      }
    }
  }
  std::puts("CPML core sanitizer exercise completed");
}
