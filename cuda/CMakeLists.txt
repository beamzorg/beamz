cmake_minimum_required(VERSION 3.24)
project(beamz_cuda LANGUAGES CXX CUDA)

option(
  BEAMZ_CUDA_FAST_MATH
  "Enable CUDA approximate math intrinsics after hardware parity validation"
  OFF
)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jax; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE JAX_FFI_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

nanobind_add_module(
  _cuda
  NB_STATIC
  src/extension.cc
  src/ffi_handler.cc
  src/graph.cu
  src/io.cu
  src/program.cu
  src/update.cu
  src/sharded.cu
  src/hopper.cu
)
target_compile_features(_cuda PRIVATE cxx_std_17)
target_include_directories(_cuda PRIVATE "${JAX_FFI_INCLUDE}" src)
set_target_properties(
  _cuda
  PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(
  _cuda
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
)
if(BEAMZ_CUDA_FAST_MATH)
  target_compile_options(_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
install(TARGETS _cuda LIBRARY DESTINATION beamz)
