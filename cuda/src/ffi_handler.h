#ifndef BEAMZ_CUDA_FFI_HANDLER_H_
#define BEAMZ_CUDA_FFI_HANDLER_H_

#include "xla/ffi/api/c_api.h"

extern "C" XLA_FFI_Error* beamz_cuda_streamed(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* beamz_cuda_sharded(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* beamz_cuda_program(XLA_FFI_CallFrame* call_frame);

#endif  // BEAMZ_CUDA_FFI_HANDLER_H_
