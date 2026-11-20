cmake_minimum_required(VERSION 3.24)
project(beamz_cuda LANGUAGES CXX CUDA)

option(BEAMZ_CUDA_BUILD_FFI "Build the JAX FFI component" ON)
option(BEAMZ_CUDA_BUILD_NATIVE "Build the independent NumPy/CUDA runner" OFF)

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
set_source_files_properties(src/ordinary/update.cu PROPERTIES COMPILE_DEFINITIONS
  "${BEAMZ_CUDA_MSTEP_RESEARCH_DEFINITIONS}")
set_source_files_properties(src/ordinary/program.cu PROPERTIES COMPILE_DEFINITIONS
  "BEAMZ_CUDA_MSTEP_INPLACE_EDGES=$<BOOL:${BEAMZ_CUDA_MSTEP_INPLACE_EDGES}>;BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES=$<BOOL:${BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES}>")
set_source_files_properties(src/bindings/extension.cc src/bindings/native_extension.cc PROPERTIES
  COMPILE_DEFINITIONS
  "BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES=$<BOOL:${BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES}>")
set(beamz_default_architectures 80 86 89 90)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND beamz_default_architectures 120)
endif()
set(BEAMZ_CUDA_ARCHITECTURES "${beamz_default_architectures}" CACHE STRING
    "CUDA architectures to compile (use 90 for an H100-only build)")

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
# Release moving kernels omit bulky debug payloads; profiling builds retain them.
include(CheckCompilerFlag)
check_compiler_flag(CUDA "-lineinfo -Xptxas=--suppress-debug-info"
                    BEAMZ_CUDA_SUPPRESS_DEBUG_INFO_SUPPORTED)
if(BEAMZ_CUDA_SUPPRESS_DEBUG_INFO_SUPPORTED)
  set_property(SOURCE src/temporal/native_moving_kernel.cu APPEND PROPERTY
               COMPILE_OPTIONS "$<$<CONFIG:Release>:-Xptxas=--suppress-debug-info>")
endif()
set(BEAMZ_CUDA_PHYSICS "${CMAKE_CURRENT_SOURCE_DIR}/compiler/physics.py")
set(BEAMZ_CUDA_CELL "${CMAKE_CURRENT_SOURCE_DIR}/src/physics/cell.cuh")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${BEAMZ_CUDA_PHYSICS}" "${BEAMZ_CUDA_CELL}" "${CMAKE_CURRENT_SOURCE_DIR}/src/physics/yee_primitives.cuh")
execute_process(COMMAND "${Python_EXECUTABLE}" "${BEAMZ_CUDA_PHYSICS}" --fingerprint
  OUTPUT_VARIABLE BEAMZ_CUDA_PHYSICS_SHA OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set_property(SOURCE src/bindings/native_extension.cc APPEND PROPERTY COMPILE_DEFINITIONS
  BEAMZ_CUDA_PHYSICS_SHA256="${BEAMZ_CUDA_PHYSICS_SHA}")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/cell.cuh" "${CMAKE_CURRENT_BINARY_DIR}/generated/curl_terms.inc" "${CMAKE_CURRENT_BINARY_DIR}/generated/yee_cell_generated.cuh"
  COMMAND "${Python_EXECUTABLE}" "${BEAMZ_CUDA_PHYSICS}" --output "${CMAKE_CURRENT_BINARY_DIR}/generated"
  DEPENDS "${BEAMZ_CUDA_PHYSICS}" "${BEAMZ_CUDA_CELL}" VERBATIM)
add_custom_target(beamz_cuda_physics
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${CMAKE_CURRENT_BINARY_DIR}/generated/cell.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/src/physics/cell.cuh"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${CMAKE_CURRENT_BINARY_DIR}/generated/curl_terms.inc" "${CMAKE_CURRENT_SOURCE_DIR}/src/physics/curl_terms.inc"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${CMAKE_CURRENT_BINARY_DIR}/generated/yee_cell_generated.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/src/physics/yee_cell_generated.cuh"
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/generated/cell.cuh" "${CMAKE_CURRENT_BINARY_DIR}/generated/curl_terms.inc" "${CMAKE_CURRENT_BINARY_DIR}/generated/yee_cell_generated.cuh")
set(BEAMZ_CUDA_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/generated" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/src/common" "${CMAKE_CURRENT_SOURCE_DIR}/src/physics" "${CMAKE_CURRENT_SOURCE_DIR}/src/ordinary" "${CMAKE_CURRENT_SOURCE_DIR}/src/ordinary/mstep" "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings" "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime" "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/deep" "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/sharded" "${CMAKE_CURRENT_SOURCE_DIR}/src/temporal" "${CMAKE_CURRENT_SOURCE_DIR}/src/temporal/host" "${CMAKE_CURRENT_SOURCE_DIR}/src/temporal/host/observation")

if(BEAMZ_CUDA_BUILD_FFI OR BEAMZ_CUDA_BUILD_NATIVE)
  # Both frontends embed these exact compiled objects. There is no new shared
  # runtime to install, and the arithmetic/compiler options cannot drift
  # between the direct native and FFI entry points in one build tree.
  add_library(beamz_cuda_core OBJECT
    src/common/graph.cu src/common/io.cu src/ordinary/program.cu
    src/ordinary/update.cu src/runtime/sharded/sharded.cu)
  add_dependencies(beamz_cuda_core beamz_cuda_physics)
  target_include_directories(beamz_cuda_core PRIVATE ${BEAMZ_CUDA_INCLUDE_DIRS})
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

# One native runtime and two schedules, compiled directly from maintained sources.
add_library(beamz_cuda_native OBJECT
  src/bindings/native_launch.cc src/runtime/native_deep.cc src/temporal/host/native_moving.cc
  src/runtime/native_runtime.cu src/temporal/native_moving_kernel.cu
  src/ordinary/compiled_ordinary.cu src/temporal/compiled_temporal_dft.cu src/physics/compiled_electric_coefficients.cu)
add_dependencies(beamz_cuda_native beamz_cuda_physics)
target_include_directories(beamz_cuda_native PRIVATE ${BEAMZ_CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(beamz_cuda_native PRIVATE cxx_std_17)
target_compile_definitions(beamz_cuda_native PRIVATE
  BEAMZ_CUDA_NATIVE_MOVING=1 BEAMZ_CUDA_NATIVE_MOVING_POLICIES=1
  BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES=1 BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION=1
  BEAMZ_CUDA_MSTEP_FULL_INTERMEDIATES=1 BEAMZ_CUDA_MSTEP_EXACT_INDEX=1
  BEAMZ_CUDA_MSTEP_FULL_EDGE_TYPED=1 BEAMZ_CUDA_MSTEP_LAYOUT_SPECIALIZE=1
  BEAMZ_CUDA_MSTEP_SKIP_FINAL_BARRIER=1 BEAMZ_CUDA_MSTEP_TRACED_CELL=1)
set_target_properties(beamz_cuda_native PROPERTIES POSITION_INDEPENDENT_CODE ON
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${BEAMZ_CUDA_ARCHITECTURES}")
target_compile_options(beamz_cuda_native PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-lineinfo;-Xcompiler=-fvisibility=hidden>)
set_property(SOURCE src/bindings/native_extension.cc APPEND PROPERTY COMPILE_DEFINITIONS
  BEAMZ_CUDA_NATIVE_MOVING=1 BEAMZ_CUDA_NATIVE_MOVING_POLICIES=1
  BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES=1 BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION=1)

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
  src/bindings/extension.cc
  src/bindings/ffi_handler.cc
  src/bindings/native_extension.cc
  src/runtime/sharded/adjoint.cu
)
target_compile_features(_cuda PRIVATE cxx_std_17)
target_include_directories(_cuda PRIVATE "${JAX_FFI_INCLUDE}" ${BEAMZ_CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(_cuda PRIVATE beamz_cuda_core beamz_cuda_native)
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
    src/bindings/native_extension.cc
    tests/native_standalone.cc
  )
  target_compile_features(_cuda_native PRIVATE cxx_std_17)
  target_include_directories(_cuda_native PRIVATE ${BEAMZ_CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(_cuda_native PRIVATE beamz_cuda_core beamz_cuda_native)
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
