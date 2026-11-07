"""The native component builds for CUDA and HIP from one source tree.

Platform specifics belong in the two prelude headers. Elsewhere, guards such as
``__CUDA_ARCH__`` are undefined under hip-clang and would silently select host
code paths in device kernels.
"""

from __future__ import annotations

import re
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[2] / "cuda" / "src"
PRELUDES = {"gpu_runtime.h", "gpu_device.h", "cpu_launch.h"}
FORBIDDEN = {
    "__CUDA_ARCH__": re.compile(r"__CUDA_ARCH__"),
    "__CUDACC__": re.compile(r"__CUDACC__"),
    "<cuda_runtime_api.h>": re.compile(r"<cuda_runtime_api\.h>"),
    "<cuda_bf16.h>": re.compile(r"<cuda_bf16\.h>"),
    "<cuda_pipeline.h>": re.compile(r"<cuda_pipeline\.h>"),
    "__pipeline_*": re.compile(r"__pipeline_\w+"),
    "__nv_bfloat16": re.compile(r"__nv_bfloat16"),
    # The CPU platform has no <<<>>> syntax and no dynamic shared memory.
    "<<<launch>>> (use BEAMZ_LAUNCH)": re.compile(r"<<<"),
    "extern __shared__ (use BEAMZ_DYNAMIC_SHARED)": re.compile(r"extern\s+__shared__"),
    # Struct kernel parameters bind by reference on the CPU platform.
    "__grid_constant__ (use BEAMZ_GRID_ARG)": re.compile(r"__grid_constant__"),
}


def _launch_bounds_violations(text: str) -> list[tuple[int, str]]:
    """Launch-bound uses that break on one of the platforms.

    HIP reads a second ``__launch_bounds__`` argument as minimum waves per
    execution unit, not blocks per multiprocessor, so two-argument bounds go
    through ``BEAMZ_LAUNCH_BOUNDS(threads, min_blocks)``. Both spellings are
    preprocessor macros on HIP, which split arguments on commas inside
    template argument lists; such arguments need parentheses. Declarations
    may span lines, so this scans the whole file.
    """
    violations = []
    for match in re.finditer(r"\b(__launch_bounds__|BEAMZ_LAUNCH_BOUNDS)\s*\(", text):
        if text[: match.start()].rstrip().endswith("#define"):
            continue
        depth, index = 1, match.end()
        while depth and index < len(text):
            depth += text[index] == "("
            depth -= text[index] == ")"
            index += 1
        body = text[match.end() : index - 1]
        while re.search(r"\([^()]*\)", body):
            body = re.sub(r"\([^()]*\)", "", body)
        arguments = body.count(",") + 1
        expected = 1 if match.group(1) == "__launch_bounds__" else 2
        if arguments != expected:
            violations.append(
                (match.start(), f"{match.group(1)} with {arguments} macro arguments")
            )
    return violations


def test_cuda_sources_use_platform_preludes():
    sources = [
        path
        for pattern in ("*.cu", "*.cuh", "*.h", "*.cc")
        for path in SOURCE_DIR.glob(pattern)
        if path.name not in PRELUDES
    ]
    assert sources
    violations = []
    for path in sorted(sources):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            for name, pattern in FORBIDDEN.items():
                if pattern.search(line):
                    violations.append(f"{path.name}:{number}: {name}")
        text = path.read_text()
        for offset, message in _launch_bounds_violations(text):
            number = text.count("\n", 0, offset) + 1
            violations.append(f"{path.name}:{number}: {message}")
    assert not violations, "\n".join(violations)


def test_cuda_runtime_names_have_hip_aliases():
    """Every CUDA runtime name in the sources is aliased for HIP builds.

    A missing alias compiles on CUDA and breaks only the HIP build.
    """
    runtime = (SOURCE_DIR / "gpu_runtime.h").read_text()
    hip_branch = runtime.split("#if defined(BEAMZ_PLATFORM_HIP)", 1)[1]
    hip_branch = hip_branch.split("#else", 1)[0]
    aliased = set(re.findall(r"^#define (cuda\w+)", hip_branch, re.MULTILINE))
    used = {}
    for pattern in ("*.cu", "*.cuh", "*.h", "*.cc"):
        for path in SOURCE_DIR.glob(pattern):
            if path.name in PRELUDES:
                continue
            for name in re.findall(r"\bcuda[A-Z]\w*", path.read_text()):
                used.setdefault(name, path.name)
    missing = sorted(f"{used[name]}: {name}" for name in set(used) - aliased)
    assert not missing, "\n".join(missing)


def test_platform_preludes_exist():
    for name in PRELUDES:
        assert (SOURCE_DIR / name).is_file(), name
