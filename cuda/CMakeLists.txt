cmake_minimum_required(VERSION 3.24)
project(beamz_cuda LANGUAGES CXX)

# One source tree, two GPU platforms. CUDA spelling is the source dialect;
# src/gpu_runtime.h and src/gpu_device.h map it onto HIP.
set(BEAMZ_GPU_PLATFORM "CUDA" CACHE STRING "Platform to build for: CUDA, HIP or CPU")
set_property(CACHE BEAMZ_GPU_PLATFORM PROPERTY STRINGS CUDA HIP CPU)

option(
  BEAMZ_CUDA_FAST_MATH
  "Enable CUDA approximate math intrinsics after hardware parity validation"
  OFF
)
set(BEAMZ_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
    "CUDA architectures to compile (use 90 for an H100-only build)")
set(BEAMZ_HIP_ARCHITECTURES "gfx1100" CACHE STRING
    "HIP architectures to compile")
option(BEAMZ_HIP_ALLOW_WAVE64
    "Allow 64-wide wave targets (gfx9, -mwavefrontsize64), unvalidated" OFF)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jax; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE JAX_FFI_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

set(BEAMZ_GPU_SOURCES
  src/graph.cu
  src/io.cu
  src/program.cu
  src/update.cu
  src/adjoint.cu
  src/sharded.cu
)

if(BEAMZ_GPU_PLATFORM STREQUAL "CUDA")
  enable_language(CUDA)
  set(BEAMZ_EXTENSION_NAME _cuda)
elseif(BEAMZ_GPU_PLATFORM STREQUAL "HIP")
  # Kernels are wave-width agnostic, but only wave32 (RDNA) has run on
  # hardware. gfx9 (GCN/CDNA) and -mwavefrontsize64 build 64-wide waves.
  if(NOT BEAMZ_HIP_ALLOW_WAVE64)
    foreach(arch IN LISTS BEAMZ_HIP_ARCHITECTURES)
      if(arch MATCHES "^gfx9")
        message(FATAL_ERROR "${arch} runs 64-wide waves, which BeamZ has not "
          "validated on hardware; set BEAMZ_HIP_ALLOW_WAVE64=ON to build anyway")
      endif()
    endforeach()
    if(CMAKE_HIP_FLAGS MATCHES "wavefrontsize64")
      message(FATAL_ERROR "-mwavefrontsize64 builds 64-wide waves, which BeamZ "
        "has not validated on hardware; set BEAMZ_HIP_ALLOW_WAVE64=ON to build anyway")
    endif()
  endif()
  set(CMAKE_HIP_ARCHITECTURES "${BEAMZ_HIP_ARCHITECTURES}")
  enable_language(HIP)
  find_package(hip CONFIG REQUIRED)
  set_source_files_properties(${BEAMZ_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
  set(BEAMZ_EXTENSION_NAME _hip)
elseif(BEAMZ_GPU_PLATFORM STREQUAL "CPU")
  # Kernels compile as C++ and run through the OpenMP block scheduler in
  # src/cpu_launch.h.
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  set_source_files_properties(${BEAMZ_GPU_SOURCES} PROPERTIES LANGUAGE CXX)
  set(BEAMZ_EXTENSION_NAME _cpu)
else()
  message(FATAL_ERROR
    "BEAMZ_GPU_PLATFORM must be CUDA, HIP or CPU, not ${BEAMZ_GPU_PLATFORM}")
endif()

nanobind_add_module(
  ${BEAMZ_EXTENSION_NAME}
  NB_STATIC
  src/extension.cc
  src/ffi_handler.cc
  ${BEAMZ_GPU_SOURCES}
)
target_compile_features(${BEAMZ_EXTENSION_NAME} PRIVATE cxx_std_17)
target_include_directories(${BEAMZ_EXTENSION_NAME} PRIVATE "${JAX_FFI_INCLUDE}" src)
target_compile_definitions(
  ${BEAMZ_EXTENSION_NAME} PRIVATE BEAMZ_EXTENSION_NAME=${BEAMZ_EXTENSION_NAME}
)

if(BEAMZ_GPU_PLATFORM STREQUAL "CUDA")
  set_target_properties(
    ${BEAMZ_EXTENSION_NAME}
    PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${BEAMZ_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_options(
    ${BEAMZ_EXTENSION_NAME}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
  )
  if(BEAMZ_CUDA_FAST_MATH)
    target_compile_options(
      ${BEAMZ_EXTENSION_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
elseif(BEAMZ_GPU_PLATFORM STREQUAL "CPU")
  target_compile_definitions(${BEAMZ_EXTENSION_NAME} PRIVATE BEAMZ_PLATFORM_CPU)
  set(BEAMZ_CPU_FLAGS "-O3;-march=native" CACHE STRING
      "Optimisation and target flags for the CPU build")
  # Explicit rounding in the kernels must not be undone by contraction.
  target_compile_options(
    ${BEAMZ_EXTENSION_NAME}
    PRIVATE ${BEAMZ_CPU_FLAGS} -ffp-contract=off
  )
  set_source_files_properties(
    ${BEAMZ_GPU_SOURCES} PROPERTIES COMPILE_OPTIONS "-xc++"
  )
  target_link_libraries(${BEAMZ_EXTENSION_NAME} PRIVATE OpenMP::OpenMP_CXX)
else()
  set_target_properties(
    ${BEAMZ_EXTENSION_NAME}
    PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
  )
  target_compile_definitions(${BEAMZ_EXTENSION_NAME} PRIVATE BEAMZ_PLATFORM_HIP)
  # No -fgpu-rdc, unlike CUDA_SEPARABLE_COMPILATION above: every device symbol
  # is defined in the translation unit that uses it (headers hold inline device
  # code; there is no extern __device__/__constant__), so whole-program device
  # compilation suffices. A cross-unit device reference fails to build here.
  # hip-clang contracts a*b+c by default; the kernels pin rounding explicitly
  # and must agree with the CUDA build and JAX.
  target_compile_options(
    ${BEAMZ_EXTENSION_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>
  )
  target_link_libraries(${BEAMZ_EXTENSION_NAME} PRIVATE hip::host)
endif()

install(TARGETS ${BEAMZ_EXTENSION_NAME} LIBRARY DESTINATION beamz)
