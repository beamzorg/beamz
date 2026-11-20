// Private moving consumer; no cross-TU device calls.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "abi_layout.h"
#include "kernels.h"
#include "launch.h"
#include "yee_primitives.cuh"
#include "mstep_index.h"
#if BEAMZ_CUDA_NATIVE_MOVING
#include "native_moving.h"
#include "temporal/stencils/compact_full_edge.cuh"
#include "cell.cuh"
#include "temporal/kernel.cuh"
#endif
