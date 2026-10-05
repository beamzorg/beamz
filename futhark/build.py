#!/usr/bin/env python3
"""Build BeamZ's optional Futhark backend into a JAX FFI shared library.

The steps are:

1. ``futhark cuda --library`` (or ``futhark hip --library``) compiles
   ``fdtd.fut`` to C plus a JSON manifest.
2. A typed XLA FFI handler is generated from the manifest: every array input is
   wrapped with ``futhark_new_raw_*`` (no copy), every scalar input becomes an
   FFI attribute of the same name, and the result record is copied into the
   XLA-owned outputs.
3. Everything is linked into ``beamz/simulation/futhark/_native/`` as
   ``libbeamz_futhark_cuda.so`` or ``libbeamz_futhark_hip.so``.

The compiler comes from the Futhark checkout next to this repository
(``../futhark``, or ``FUTHARK_SRC``), built with ``cabal`` from ghcup. That
checkout carries fixes BeamZ needs that are not yet released: the CUDA primary
context option (shared with XLA, so device pointers alias), results of
intra-block kernels written straight to global memory, failing a call on out of
device memory, and the multi-dimensional histogram and ISPC fixes (see
FUTHARK_ISSUES.md). Set ``FUTHARK`` to use another compiler binary,
``CUDA_HOME`` if the toolkit is not in ``/opt/cuda`` or ``/usr/local/cuda``, and
``ROCM_PATH`` if ROCm is not in ``/opt/rocm``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = ROOT / "beamz" / "simulation" / "futhark" / "_native"
ENTRY = "program"
SYMBOL = "BeamzFutharkProgram"

_ELEMENT = {
    "f32": ("float", "F32"),
    "i32": ("int32_t", "S32"),
    "i64": ("int64_t", "S64"),
}


def _parse_array(type_name: str) -> tuple[int, str] | None:
    rank = type_name.count("[]")
    if not rank:
        return None
    return rank, type_name.replace("[]", "")


@dataclass(frozen=True)
class Platform:
    """GPU runtime spellings that differ between the CUDA and HIP builds."""

    name: str
    pointer: str  # Futhark's raw device-pointer type
    headers: tuple[str, ...]
    # Disable FMA contraction so that the explicit f32.fma calls in fdtd.fut
    # reproduce the CUDA backend's separately rounded multiply/FMA sequence.
    rtc_option: tuple[str, str]
    # Needed by the handler, which includes the runtime headers before fdtd.h
    # (whose own definitions cover fdtd.c).
    handler_defines: tuple[str, ...]
    libraries: tuple[str, ...]
    library_dir: str
    toolkits: tuple[str, ...]

    @property
    def gpu(self) -> bool:
        return self.name in ("cuda", "hip")

    @property
    def library(self) -> str:
        return f"libbeamz_futhark_{self.name}.so"

    def toolkit(self) -> Path | None:
        if not self.gpu:
            return None
        for candidate in self.toolkits:
            if candidate and (Path(candidate) / "include" / self.headers[0]).exists():
                return Path(candidate)
        raise SystemExit(f"{self.name} toolkit not found; set CUDA_HOME or ROCM_PATH")


PLATFORMS = {
    "cuda": Platform(
        name="cuda",
        pointer="CUdeviceptr",
        headers=("cuda.h", "cuda_runtime.h", "nvrtc.h"),
        rtc_option=("futhark_context_config_add_nvrtc_option", "--fmad=false"),
        handler_defines=(),
        libraries=("cuda", "nvrtc", "cudart"),
        library_dir="lib64",
        toolkits=(os.environ.get("CUDA_HOME", ""), "/opt/cuda", "/usr/local/cuda"),
    ),
    "hip": Platform(
        name="hip",
        pointer="hipDeviceptr_t",
        headers=("hip/hip_runtime.h", "hip/hiprtc.h"),
        rtc_option=("futhark_context_config_add_build_option", "-ffp-contract=off"),
        handler_defines=("__HIP_PLATFORM_AMD__",),
        libraries=("amdhip64", "hiprtc"),
        library_dir="lib",
        toolkits=(os.environ.get("ROCM_PATH", ""), "/opt/rocm"),
    ),
    # Host-memory builds for XLA's CPU client: no streams, no runtime compiler.
    **{
        name: Platform(
            name=name,
            pointer="unsigned char*",
            headers=(),
            rtc_option=("", ""),
            handler_defines=(),
            libraries=("pthread", "m"),
            library_dir="",
            toolkits=(),
        )
        for name in ("c", "multicore", "ispc")
    },
}


def generate_handler(
    manifest: dict,
    platform: Platform = PLATFORMS["cuda"],
    rtc_includes: tuple[str, ...] = (),
) -> str:
    """Return C++ for one typed XLA FFI handler around the Futhark entry."""
    rt = platform.name  # runtime API prefix: cudaMemcpyAsync / hipMemcpyAsync
    entry = manifest["entry_points"][ENTRY]
    record = manifest["types"][entry["output"]["type"]]
    fields = record["record"]["fields"]
    attrs, arrays, call_args = [], [], []
    for item in entry["inputs"]:
        array = _parse_array(item["type"])
        name = item["name"]
        if array is None:
            ctype, _ = _ELEMENT[item["type"]]
            attrs.append((name, ctype))
            call_args.append(name)
        else:
            rank, element = array
            if item["consumed"]:
                raise SystemExit(
                    f"Entry input {name!r} is consumed; XLA-aliased inputs must not be"
                )
            arrays.append((name, rank, element))
            call_args.append(f"in_{name}")

    lines = [
        f"// Generated by futhark/build.py from the {ENTRY!r} manifest; do not edit.",
        "#include <cstdint>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <cstring>",
        "#include <mutex>",
        "#include <string>",
        "",
        # Include the runtime headers outside extern "C"; fdtd.h repeats them.
        *[f"#include <{name}>" for name in platform.headers],
        "",
        '#include "xla/ffi/api/ffi.h"',
        "",
        'extern "C" {',
        '#include "fdtd.h"',
        "}",
        "",
        "namespace ffi = xla::ffi;",
        "",
        "namespace {",
        "",
        "struct Runtime {",
        "  futhark_context_config* config = nullptr;",
        "  futhark_context* context = nullptr;",
        "  int device = -1;",
        "  std::mutex lock;",
        "};",
        "",
        "Runtime& GetRuntime() {",
        "  static Runtime runtime;",
        "  return runtime;",
        "}",
        "",
        "std::string TakeError(futhark_context* ctx) {",
        "  char* message = futhark_context_get_error(ctx);",
        '  std::string result = message ? message : "unknown Futhark error";',
        "  std::free(message);",
        "  return result;",
        "}",
        "",
        "// One context per process, created on XLA's device the first time a",
        "// program runs. Kernels compile at runtime once and may be cached on disk.",
        "ffi::Error EnsureContext(Runtime& runtime) {",
        "  int device = 0;",
        *(
            [
                f"  if ({rt}GetDevice(&device) != {rt}Success) {{",
                f'    return ffi::Error::Internal("{rt}GetDevice failed");',
                "  }",
            ]
            if platform.gpu
            else []
        ),
        "  if (runtime.context != nullptr) {",
        "    if (runtime.device != device) {",
        "      return ffi::Error(ffi::ErrorCode::kUnimplemented,",
        '                        "Futhark backend supports one GPU per process");',
        "    }",
        "    return ffi::Error::Success();",
        "  }",
        "  runtime.config = futhark_context_config_new();",
        *(
            [
                '  const std::string selector = "#" + std::to_string(device);',
                "  futhark_context_config_set_device(runtime.config, selector.c_str());",
                "  // Match the CUDA backend's explicitly rounded multiply/FMA sequence.",
                f'  {platform.rtc_option[0]}(runtime.config, "{platform.rtc_option[1]}");',
                # HIPRTC does not search the ROCm headers that Futhark kernels include.
                *[
                    f'  {platform.rtc_option[0]}(runtime.config, "-I{path}");'
                    for path in rtc_includes
                ],
                "  // Experiments: one extra kernel compiler option, e.g. a register cap.",
                '  if (const char* option = std::getenv("BEAMZ_FUTHARK_RTC_OPTION")) {',
                f"    {platform.rtc_option[0]}(runtime.config, option);",
                "  }",
                *(
                    [
                        "  // Share XLA's primary context so raw device pointers alias.",
                        "  futhark_context_config_set_use_primary_context(runtime.config, 1);",
                        "  // Diagnostics: the PTX NVRTC built, for `ptxas -v` register counts.",
                        '  if (const char* ptx = std::getenv("BEAMZ_FUTHARK_DUMP_PTX")) {',
                        "    futhark_context_config_dump_ptx_to(runtime.config, ptx);",
                        "  }",
                        "  // Fail on exhaustion instead of paging managed memory, which slows",
                        "  // kernels about tenfold (BEAMZ_FUTHARK_UNIFIED=1 restores paging).",
                        "  futhark_context_config_set_unified_memory(",
                        '      runtime.config, std::getenv("BEAMZ_FUTHARK_UNIFIED") ? 1 : 0);',
                    ]
                    if platform.name == "cuda"
                    else []
                ),
            ]
            if platform.gpu
            else [
                '  if (const char* threads = std::getenv("BEAMZ_FUTHARK_THREADS")) {',
                "    futhark_context_config_set_num_threads(runtime.config, std::atoi(threads));",
                "  }",
            ]
            if platform.name != "c"
            else []
        ),
        "  // Opt-in per-kernel timing, printed to stderr after every program.",
        '  if (std::getenv("BEAMZ_FUTHARK_PROFILE")) {',
        "    futhark_context_config_set_profiling(runtime.config, 1);",
        "  }",
        "  // Diagnostics: Futhark's event log (allocations, kernels) on stderr.",
        '  if (std::getenv("BEAMZ_FUTHARK_LOG")) {',
        "    futhark_context_config_set_logging(runtime.config, 1);",
        "  }",
        "  // Diagnostics: synchronise and check for errors after every kernel.",
        '  if (std::getenv("BEAMZ_FUTHARK_DEBUG")) {',
        "    futhark_context_config_set_debugging(runtime.config, 1);",
        "  }",
        '  if (const char* cache = std::getenv("BEAMZ_FUTHARK_CACHE_FILE")) {',
        "    futhark_context_config_set_cache_file(runtime.config, cache);",
        "  }",
        "  futhark_context* context = futhark_context_new(runtime.config);",
        "  char* error = context ? futhark_context_get_error(context) : nullptr;",
        "  if (context == nullptr || error != nullptr) {",
        '    std::string message = error ? error : "futhark_context_new failed";',
        "    std::free(error);",
        "    // Leave no half-built context behind so that the next call retries.",
        "    if (context != nullptr) futhark_context_free(context);",
        "    futhark_context_config_free(runtime.config);",
        "    runtime.config = nullptr;",
        "    return ffi::Error::Internal(message);",
        "  }",
        "  runtime.context = context;",
        "  runtime.device = device;",
        "  return ffi::Error::Success();",
        "}",
        "",
        "ffi::Error Mismatch(const char* name) {",
        '  return ffi::Error::InvalidArgument(std::string("unexpected dtype, rank or shape for ") + name);',
        "}",
        "",
    ]
    attr_params = "".join(f", {ctype} {name}" for name, ctype in attrs)
    stream_param = f"{rt}Stream_t stream, " if platform.gpu else ""
    lines += [
        f"ffi::Error ProgramImpl({stream_param}ffi::RemainingArgs args,",
        f"                       ffi::RemainingRets rets{attr_params}) {{",
        f"  if (args.size() != {len(arrays)} || rets.size() != {len(fields)}) {{",
        '    return ffi::Error::InvalidArgument("Futhark program arity mismatch");',
        "  }",
        "  Runtime& runtime = GetRuntime();",
        "  std::lock_guard<std::mutex> guard(runtime.lock);",
        "  if (auto error = EnsureContext(runtime); error.failure()) return error;",
        "  futhark_context* ctx = runtime.context;",
        *(
            [
                "  // Producers run on XLA's stream, Futhark on its own.",
                f"  if ({rt}StreamSynchronize(stream) != {rt}Success) {{",
                '    return ffi::Error::Internal("XLA stream synchronization failed");',
                "  }",
            ]
            if platform.gpu
            else []
        ),
    ]
    for index, (name, rank, element) in enumerate(arrays):
        _, dtype = _ELEMENT[element]
        dims = ", ".join(f"d_{name}[{axis}]" for axis in range(rank))
        lines += [
            f"  auto a_{name} = args.get<ffi::AnyBuffer>({index});",
            f"  if (a_{name}.has_error()) return a_{name}.error();",
            f"  if (a_{name}->element_type() != ffi::DataType::{dtype} ||",
            f'      a_{name}->dimensions().size() != {rank}) return Mismatch("{name}");',
            f"  auto d_{name} = a_{name}->dimensions();",
            f"  futhark_{element}_{rank}d* in_{name} = futhark_new_raw_{element}_{rank}d(",
            f"      ctx, reinterpret_cast<{platform.pointer}>(a_{name}->untyped_data()), {dims});",
        ]
    release_inputs = [
        f"    futhark_free_{element}_{rank}d(ctx, in_{name});"
        for name, rank, element in arrays
    ]
    opaque = record["ctype"].replace(" *", "")
    lines += [
        f"  {opaque}* result = nullptr;",
        f"  int status = {entry['cfun']}(ctx, &result, {', '.join(call_args)});",
        "  if (status == 0) status = futhark_context_sync(ctx);",
        "  std::string failure = status == 0 ? std::string() : TakeError(ctx);",
        "  auto release_inputs = [&]() {",
        *release_inputs,
        "  };",
        "  if (status != 0) {",
        "    release_inputs();",
        "    return ffi::Error::Internal(failure);",
        "  }",
        "  ffi::Error outcome = ffi::Error::Success();",
    ]
    copy = (
        [
            f"      if ({rt}MemcpyAsync(target, source, bytes, {rt}MemcpyDeviceToDevice,",
            f"                         stream) != {rt}Success) {{",
            '        outcome = ffi::Error::Internal("output copy failed");',
            "      }",
        ]
        if platform.gpu
        else ["      std::memcpy(target, source, bytes);"]
    )
    outputs = []
    for index, field in enumerate(fields):
        rank, element = _parse_array(field["type"])
        cname, dtype = _ELEMENT[element]
        outputs.append(f"futhark_free_{element}_{rank}d(ctx, out_{index});")
        lines += [
            f"  futhark_{element}_{rank}d* out_{index} = nullptr;",
            "  {",
            f"    {field['project']}(ctx, &out_{index}, result);",
            f"    auto r = rets.get<ffi::AnyBuffer>({index});",
            f"    const int64_t* shape = futhark_shape_{element}_{rank}d(ctx, out_{index});",
            "    size_t elements = 1;",
            "    bool same = r.has_value() && (*r)->dimensions().size() == "
            f"{rank} && (*r)->element_type() == ffi::DataType::{dtype};",
            f"    for (int axis = 0; same && axis < {rank}; ++axis) {{",
            "      same = (*r)->dimensions()[axis] == shape[axis];",
            "      elements *= static_cast<size_t>(shape[axis]);",
            "    }",
            "    void* target = same ? (*r)->untyped_data() : nullptr;",
            "    void* source = reinterpret_cast<void*>(",
            f"        futhark_values_raw_{element}_{rank}d(ctx, out_{index}));",
            f"    size_t bytes = elements * sizeof({cname});",
            "    if (!same) {",
            f'      outcome = Mismatch("output {index}");',
            "    } else if (bytes != 0 && source != target) {",
            *copy,
            "    }",
            "  }",
        ]
    if platform.gpu:
        lines += [
            "  // Keep Futhark memory alive until the copies have drained.",
            f"  if ({rt}StreamSynchronize(stream) != {rt}Success) {{",
            '    outcome = ffi::Error::Internal("output copy failed");',
            "  }",
        ]
    lines += [f"  {line}" for line in outputs]
    free_record = record["ops"]["free"]
    lines += [
        f"  {free_record}(ctx, result);",
        '  if (std::getenv("BEAMZ_FUTHARK_PROFILE")) {',
        "    char* report = futhark_context_report(ctx);",
        "    std::fputs(report, stderr);",
        "    std::free(report);",
        "  }",
        "  release_inputs();",
        "  return outcome;",
        "}",
        "",
        "}  // namespace",
        "",
        f"XLA_FFI_DEFINE_HANDLER_SYMBOL({SYMBOL}, ProgramImpl,",
        "    ffi::Ffi::Bind()",
        *(
            [f"        .Ctx<ffi::PlatformStream<{rt}Stream_t>>()"]
            if platform.gpu
            else []
        ),
        "        .RemainingArgs()",
        "        .RemainingRets()",
        *[f'        .Attr<{ctype}>("{name}")' for name, ctype in attrs],
        ");",
        "",
    ]
    return "\n".join(lines)


def find_futhark() -> str | None:
    """Return ``FUTHARK``, else the sibling checkout's compiler (built first),
    else ``futhark`` on ``PATH``."""
    if compiler := os.environ.get("FUTHARK"):
        return compiler
    checkout = Path(os.environ.get("FUTHARK_SRC", ROOT.parent / "futhark"))
    if (checkout / "futhark.cabal").exists():
        # Equivalent of sourcing ~/.ghcup/env.
        env = dict(os.environ)
        extra = [Path.home() / ".ghcup" / "bin", Path.home() / ".cabal" / "bin"]
        env["PATH"] = os.pathsep.join([*map(str, extra), env.get("PATH", "")])
        cabal = shutil.which("cabal", path=env["PATH"])
        if cabal is None:
            raise SystemExit(
                f"cabal not found; needed to build the compiler in {checkout}"
            )
        subprocess.run(
            [cabal, "-v0", "build", "exe:futhark"], cwd=checkout, env=env, check=True
        )
        return subprocess.run(
            [cabal, "-v0", "list-bin", "exe:futhark"],
            cwd=checkout,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return shutil.which("futhark")


def build(futhark: str, work: Path, output: Path, platform: Platform) -> Path:
    import jax.ffi

    toolkit = platform.toolkit()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    stem = work / "fdtd"
    subprocess.run(
        [futhark, platform.name, "--library", "-o", str(stem), str(HERE / "fdtd.fut")],
        check=True,
    )
    manifest = json.loads(stem.with_suffix(".json").read_text())
    handler = work / "ffi_handler.cc"
    rtc_includes = (str(toolkit / "include"),) if platform.name == "hip" else ()
    handler.write_text(generate_handler(manifest, platform, rtc_includes))
    flags = ["-O3", "-fPIC", f"-I{work}"]
    if toolkit is not None:
        flags.append(f"-I{toolkit / 'include'}")
    else:
        # Host code is the numerical program here; keep FMAs explicit.
        flags += ["-march=native", "-ffp-contract=off"]
    source, handler_object = stem.with_suffix(".c"), work / "ffi_handler.o"
    subprocess.run(
        ["cc", *flags, "-std=c11", "-c", str(source), "-o", str(work / "fdtd.o")],
        check=True,
    )
    objects = [work / "fdtd.o"]
    if platform.name == "ispc":
        kernels = work / "fdtd.kernels.o"
        subprocess.run(
            [os.environ.get("ISPC", "ispc"), "-O3", "--pic", "--addressing=64"]
            + ["--woff", str(stem.with_suffix(".kernels.ispc")), "-o", str(kernels)],
            check=True,
        )
        objects.append(kernels)
    subprocess.run(
        ["c++", *flags, "-std=c++17", f"-I{jax.ffi.include_dir()}"]
        + [f"-D{define}" for define in platform.handler_defines]
        + ["-c", str(handler), "-o", str(handler_object)],
        check=True,
    )
    library = output / platform.library
    search = []
    if toolkit is not None:
        library_dir = toolkit / platform.library_dir
        search = [f"-L{library_dir}", f"-Wl,-rpath,{library_dir}"]
    subprocess.run(
        ["c++", "-shared", "-o", str(library), *map(str, objects)]
        + [str(handler_object), *search]
        + [f"-l{name}" for name in platform.libraries],
        check=True,
    )
    (output / f"manifest_{platform.name}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--futhark", help="default: see find_futhark")
    parser.add_argument("--platform", choices=sorted(PLATFORMS), default="cuda")
    parser.add_argument("--work", type=Path, help="default: futhark/build/<platform>")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.futhark = args.futhark or find_futhark()
    if not args.futhark:
        parser.error("futhark compiler not found; pass --futhark or set FUTHARK")
    platform = PLATFORMS[args.platform]
    work = args.work or HERE / "build" / platform.name
    print(build(args.futhark, work, args.output, platform))


if __name__ == "__main__":
    main()
