cmake_minimum_required(VERSION 3.24)
project(beamz_cuda LANGUAGES CXX CUDA)

option(BEAMZ_CUDA_BUILD_FFI "Build the JAX FFI component" ON)
option(BEAMZ_CUDA_BUILD_NATIVE "Build the independent NumPy/CUDA runner" OFF)
option(BEAMZ_CUDA_DEEP_D4 "Research candidate: generated four-step native core" OFF)

option(
  BEAMZ_CUDA_FAST_MATH
  "Enable CUDA approximate math intrinsics after hardware parity validation"
  OFF
)
option(BEAMZ_CUDA_MSTEP_SKIP_FINAL_BARRIER
       "Remove the redundant packed-mstep terminal stage barrier" ON)
option(BEAMZ_CUDA_MSTEP_EDGE_SPECIALIZE
       "Research candidate: specialize uniform-CPML mstep edges" OFF)
option(BEAMZ_CUDA_MSTEP_EXACT_INDEX
       "Use exact reciprocal division for mstep edge indices" ON)
option(BEAMZ_CUDA_MSTEP_EARLY_PREFETCH_WAIT
       "Research candidate: merge packed-mstep prefetch and publication barriers" OFF)
option(BEAMZ_CUDA_MSTEP_TRACED_CELL
       "Use generated Python-traced Yee cell arithmetic" ON)
option(BEAMZ_CUDA_MSTEP_REGISTER_CORE
       "Research candidate: keep packed-mstep history in registers" OFF)
option(BEAMZ_CUDA_MSTEP_REGISTER_FOUR_CTAS
       "Research candidate: target four resident register-core 8x16 blocks" OFF)
option(BEAMZ_CUDA_MSTEP_LAYOUT_SPECIALIZE
       "Specialize homogeneous packed-material layouts" ON)
option(BEAMZ_CUDA_MSTEP_CPML_AXIS_MASK
       "Research candidate: specialize uniform-CPML edge active axes" OFF)
option(BEAMZ_CUDA_MSTEP_INPLACE_EDGES
       "Research candidate: publish core intermediates and update full edge fields in place" OFF)
option(BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES
       "Recompute the first-step edge envelope in full field banks" ON)
option(BEAMZ_CUDA_MSTEP_FULL_EDGE_TYPED
       "Specialize proven full-edge CPML precision and wall handling" ON)
option(BEAMZ_CUDA_MSTEP_FULL_EDGE_COEFFICIENTS
       "Research candidate: specialize full-edge scalar and packed coefficients" OFF)
set(BEAMZ_CUDA_MSTEP_RESEARCH_DEFINITIONS "")
foreach(variant
    SKIP_FINAL_BARRIER EDGE_SPECIALIZE EXACT_INDEX EARLY_PREFETCH_WAIT
    TRACED_CELL REGISTER_CORE REGISTER_FOUR_CTAS LAYOUT_SPECIALIZE
    CPML_AXIS_MASK INPLACE_EDGES FULL_INTERMEDIATES FULL_EDGE_TYPED
    FULL_EDGE_COEFFICIENTS)
  list(APPEND BEAMZ_CUDA_MSTEP_RESEARCH_DEFINITIONS
       "BEAMZ_CUDA_MSTEP_${variant}=$<BOOL:${BEAMZ_CUDA_MSTEP_${variant}}>")
endforeach()
set_source_files_properties(src/update.cu PROPERTIES COMPILE_DEFINITIONS
  "${BEAMZ_CUDA_MSTEP_RESEARCH_DEFINITIONS};BEAMZ_CUDA_DEEP_D4=$<BOOL:${BEAMZ_CUDA_DEEP_D4}>")
set_source_files_properties(src/program.cu PROPERTIES COMPILE_DEFINITIONS
  "BEAMZ_CUDA_MSTEP_INPLACE_EDGES=$<BOOL:${BEAMZ_CUDA_MSTEP_INPLACE_EDGES}>;BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES=$<BOOL:${BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES}>")
set_source_files_properties(src/extension.cc src/native_extension.cc PROPERTIES
  COMPILE_DEFINITIONS
  "BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES=$<BOOL:${BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES}>")
set_property(SOURCE src/native_extension.cc APPEND PROPERTY COMPILE_DEFINITIONS
  "BEAMZ_CUDA_DEEP_D4=$<BOOL:${BEAMZ_CUDA_DEEP_D4}>")
set(BEAMZ_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
    "CUDA architectures to compile (use 90 for an H100-only build)")

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

if(BEAMZ_CUDA_BUILD_FFI OR BEAMZ_CUDA_BUILD_NATIVE)
  # Both frontends embed these exact compiled objects. There is no new shared
  # runtime to install, and the arithmetic/compiler options cannot drift
  # between the direct native and FFI entry points in one build tree.
  add_library(beamz_cuda_core OBJECT
    src/graph.cu src/io.cu src/program.cu src/update.cu)
  target_include_directories(beamz_cuda_core PRIVATE src)
  target_compile_features(beamz_cuda_core PRIVATE cxx_std_17)
  set_target_properties(beamz_cuda_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${BEAMZ_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(beamz_cuda_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo;-Xcompiler=-fvisibility=hidden>)
  if(BEAMZ_CUDA_FAST_MATH)
    target_compile_options(beamz_cuda_core PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
endif()

if(BEAMZ_CUDA_BUILD_FFI)
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
  src/adjoint.cu
  src/sharded.cu
)
target_compile_features(_cuda PRIVATE cxx_std_17)
target_include_directories(_cuda PRIVATE "${JAX_FFI_INCLUDE}" src)
target_link_libraries(_cuda PRIVATE beamz_cuda_core)
set_target_properties(
  _cuda
  PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${BEAMZ_CUDA_ARCHITECTURES}"
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
endif()

if(BEAMZ_CUDA_BUILD_NATIVE)
  nanobind_add_module(
    _cuda_native
    NB_STATIC
    src/native_extension.cc
    src/native_launch.cc
    src/native_deep.cc
    src/native_runtime.cu
  )
  target_compile_features(_cuda_native PRIVATE cxx_std_17)
  target_include_directories(_cuda_native PRIVATE src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(_cuda_native PRIVATE beamz_cuda_core)
  set_target_properties(_cuda_native PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES "${BEAMZ_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(_cuda_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>)
  if(BEAMZ_CUDA_FAST_MATH)
    target_compile_options(_cuda_native PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()
  # A top-level import must not execute beamz/__init__.py, which belongs to the
  # existing JAX frontend. The optional native component imports independently.
  install(TARGETS _cuda_native LIBRARY DESTINATION .)
endif()
