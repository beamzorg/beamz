"""Execution-backend selection with a lazy optional CUDA extension boundary."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import jax

from beamz.simulation._cuda_abi import (
    CUDA_ABI_VERSION,
    CUDA_ADJOINT_TARGET,
    CUDA_BF16_PSI,
    CUDA_CPML_PAIR,
    CUDA_DEFAULT_FLAGS,
    CUDA_FIELD_PAD32,
    CUDA_FIELD_PAD64,
    CUDA_FIELD_PAD_Y8,
    CUDA_GRAPH_CACHE,
    CUDA_MATERIAL_ADJOINT_TARGET,
    CUDA_SHARDED_TARGET,
    CUDA_SHELL32X4,
    CUDA_SHELL32X8,
    CUDA_STREAMED_TARGET,
    CUDA_STREAMED_TARGETS,
    CUDA_TEMPORAL_PAIR,
)

ExecutionBackend = Literal[
    "auto",
    "jax",
    "cuda",
    "cuda_streamed",
    "cuda_mstep",
]
ResolvedBackend = Literal["jax", "cuda_streamed", "cuda_mstep"]

_EXTENSION_MODULE = "beamz._cuda"
# The same native sources build for AMD GPUs through HIP as beamz._hip.
_HIP_EXTENSION_MODULE = "beamz._hip"
# ...and for JAX's CPU client as beamz._cpu (an OpenMP block scheduler).
_CPU_EXTENSION_MODULE = "beamz._cpu"
_EXTENSION_MODULES = {
    "cuda": _EXTENSION_MODULE,
    "rocm": _HIP_EXTENSION_MODULE,
    "cpu": _CPU_EXTENSION_MODULE,
}
# XLA platform names for FFI registration, keyed by the extension's
# ``__gpu_platform__``. Extensions predating the attribute are CUDA builds.
_XLA_PLATFORMS = {"cuda": "CUDA", "rocm": "ROCM", "cpu": "cpu"}
_REGISTERED_MODULE: ModuleType | None = None
_DEFAULT_CUDA_GRAPH_CACHE_CAPACITY = 32
_MAX_CUDA_GRAPH_CACHE_CAPACITY = 4096


class CudaBackendUnavailable(RuntimeError):
    """An explicitly requested CUDA backend cannot run in this process."""


def _env_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def cuda_flags_from_env() -> int:
    """Snapshot CUDA execution choices into one cacheable bitset."""
    flags = CUDA_DEFAULT_FLAGS
    if _env_enabled("BEAMZ_CUDA_DISABLE_GRAPH_CACHE", default=False):
        flags &= ~CUDA_GRAPH_CACHE
    precision = os.environ.get("BEAMZ_CUDA_CPML_PSI_PRECISION", "fp32").lower()
    if precision in {"bf16", "bfloat16"}:
        flags |= CUDA_BF16_PSI
    elif precision not in {"fp32", "float32", ""}:
        raise ValueError("BEAMZ_CUDA_CPML_PSI_PRECISION must be 'fp32' or 'bf16'")
    padding = os.environ.get("BEAMZ_CUDA_FIELD_PADDING", "none").lower()
    layouts = {
        "none": 0,
        "32": CUDA_FIELD_PAD32,
        "64": CUDA_FIELD_PAD64,
        "32x8": CUDA_FIELD_PAD32 | CUDA_FIELD_PAD_Y8,
        "64x8": CUDA_FIELD_PAD64 | CUDA_FIELD_PAD_Y8,
    }
    if padding not in layouts:
        raise ValueError("BEAMZ_CUDA_FIELD_PADDING must be none, 32, 64, 32x8, or 64x8")
    flags |= layouts[padding]
    temporal_steps = os.environ.get("BEAMZ_CUDA_TEMPORAL_STEPS", "1")
    if temporal_steps not in {"1", "2"}:
        raise ValueError("BEAMZ_CUDA_TEMPORAL_STEPS must be 1 or 2")
    if temporal_steps == "2":
        flags |= CUDA_TEMPORAL_PAIR
    if _env_enabled("BEAMZ_CUDA_CPML_TEMPORAL", default=False):
        flags |= CUDA_TEMPORAL_PAIR | CUDA_CPML_PAIR
    shell_tile = os.environ.get("BEAMZ_CUDA_CPML_SHELL_TILE", "64x4")
    shell_tiles = {"64x4": 0, "32x8": CUDA_SHELL32X8, "32x4": CUDA_SHELL32X4}
    if shell_tile not in shell_tiles:
        raise ValueError("BEAMZ_CUDA_CPML_SHELL_TILE must be 64x4, 32x8, or 32x4")
    flags |= shell_tiles[shell_tile]
    from .cuda.mstep_config import variant_flags_from_env

    flags |= variant_flags_from_env()
    return flags


def platform_cuda_flags(flags: int) -> int:
    """Drop schedule requests the selected native build cannot run.

    The CPU build runs barrier kernels only as one-thread blocks. The interior
    pair tiles are written that way; the CPML pair is not yet, so it falls
    back to the interior pair, as unsupported tuning choices do on GPUs.
    """
    if _gpu_platform() == "cpu":
        flags &= ~CUDA_CPML_PAIR
    return flags


def cuda_storage_axes_from_env(*, sharded: bool = False) -> tuple[int, int, int]:
    """Snapshot an experimental, right-handed CUDA storage-axis order."""
    value = os.environ.get(
        "BEAMZ_CUDA_STORAGE_AXES", "201" if sharded else "012"
    ).strip()
    orders = {"012": (0, 1, 2), "120": (1, 2, 0), "201": (2, 0, 1)}
    if value not in orders:
        raise ValueError("BEAMZ_CUDA_STORAGE_AXES must be 012, 120, or 201")
    return orders[value]


def cuda_graph_cache_capacity_from_env() -> int:
    """Snapshot the bounded native graph-cache target for a compiled program.

    A capacity of zero keeps graph capture available but disables persistent
    executables.  Native eviction may briefly exceed a positive target while
    every candidate is still running on its CUDA stream.
    """
    value = os.environ.get("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY")
    if value is None or not value.strip():
        return _DEFAULT_CUDA_GRAPH_CACHE_CAPACITY
    try:
        capacity = int(value)
    except ValueError as exc:
        raise ValueError(
            "BEAMZ_CUDA_GRAPH_CACHE_CAPACITY must be an integer from 0 to "
            f"{_MAX_CUDA_GRAPH_CACHE_CAPACITY}"
        ) from exc
    if not 0 <= capacity <= _MAX_CUDA_GRAPH_CACHE_CAPACITY:
        raise ValueError(
            "BEAMZ_CUDA_GRAPH_CACHE_CAPACITY must be an integer from 0 to "
            f"{_MAX_CUDA_GRAPH_CACHE_CAPACITY}"
        )
    return capacity


@dataclass(frozen=True, slots=True)
class CudaBackendStatus:
    """Diagnostic snapshot used by backend selection and bug reports."""

    available: bool
    extension_version: str | None
    abi_version: int | None
    targets: tuple[str, ...]
    gpu_devices: tuple[str, ...]
    compute_capabilities: tuple[int, ...]
    reason: str | None = None
    gpu_platform: str = "cuda"

    def as_dict(self) -> dict[str, object]:
        """Return stable diagnostics suitable for logs and bug reports."""
        return {
            "available": self.available,
            "extension_version": self.extension_version,
            "abi_version": self.abi_version,
            "targets": self.targets,
            "gpu_devices": self.gpu_devices,
            "compute_capabilities": self.compute_capabilities,
            "reason": self.reason,
            "gpu_platform": self.gpu_platform,
        }


def normalize_backend(backend: str | None) -> ExecutionBackend:
    value = (
        os.environ.get("BEAMZ_EXECUTION_BACKEND", "auto")
        if backend is None
        else str(backend)
    )
    aliases = {
        "auto": "auto",
        "jax": "jax",
        "xla": "jax",
        "cuda": "cuda",
        "streamed": "cuda_streamed",
        "cuda_streamed": "cuda_streamed",
        "cuda-streamed": "cuda_streamed",
        "cuda_mstep": "cuda_mstep",
        "cuda-mstep": "cuda_mstep",
    }
    try:
        return aliases[value.strip().lower()]  # type: ignore[return-value]
    except KeyError as exc:
        choices = "auto, jax, cuda, cuda_streamed, cuda_mstep"
        raise ValueError(
            f"Unknown execution backend {value!r}; use one of: {choices}."
        ) from exc


def _gpu_platform() -> str:
    """``"rocm"`` or ``"cuda"`` for JAX's GPU client, ``"cpu"`` without one."""
    devices = _gpu_devices()
    if not devices:
        return "cpu"
    for device in devices:
        client = getattr(device, "client", None)
        # e.g. "PJRT C API\nrocm 70200" or "PJRT C API\ncuda 13000".
        version = str(getattr(client, "platform_version", "")).lower()
        if any(line.strip().startswith("rocm") for line in version.splitlines()):
            return "rocm"
        break
    return "cuda"


def _extension_module_name() -> str:
    return _EXTENSION_MODULES[_gpu_platform()]


def _load_extension() -> ModuleType:
    return importlib.import_module(_extension_module_name())


def _extension_platform(extension: ModuleType) -> str:
    platform = str(getattr(extension, "__gpu_platform__", "cuda"))
    if platform not in _XLA_PLATFORMS:
        raise CudaBackendUnavailable(f"unknown native GPU platform {platform!r}")
    return platform


def _gpu_devices():
    try:
        return tuple(jax.devices("gpu"))
    except Exception:
        return ()


def _compute_capability(device) -> int:
    capability = getattr(device, "compute_capability", None)
    if capability is None:
        stats = getattr(device, "device_stats", lambda: {})() or {}
        capability = stats.get("compute_capability")
    if isinstance(capability, tuple):
        return int(capability[0]) * 10 + int(capability[1])
    if isinstance(capability, str) and capability.startswith("gfx"):
        # ROCm reports the AMD ISA name; NVIDIA capability gates never apply.
        return 0
    if isinstance(capability, str):
        parts = capability.replace("sm_", "").split(".")
        return int(parts[0]) * 10 + int(parts[1]) if len(parts) == 2 else int(parts[0])
    if capability is None:
        # The model name is enough for hardware diagnostics when older jaxlib builds
        # do not expose compute capability directly.
        return 90 if "H100" in str(getattr(device, "device_kind", "")).upper() else 0
    return int(capability)


def _required_targets(platform: str) -> frozenset[str]:
    """Targets a native build must provide; the CPU build is single-device."""
    if platform == "cpu":
        return CUDA_STREAMED_TARGETS - {CUDA_SHARDED_TARGET}
    return CUDA_STREAMED_TARGETS


def _validated_registrations(extension: ModuleType) -> dict[str, object]:
    module = getattr(extension, "__name__", "native extension")
    abi_version = getattr(extension, "__abi_version__", None)
    if abi_version != CUDA_ABI_VERSION:
        raise CudaBackendUnavailable(
            f"{module} ABI mismatch: "
            f"runtime requires v{CUDA_ABI_VERSION}, extension provides "
            f"{abi_version!r}"
        )
    registrations = extension.registrations()
    if not isinstance(registrations, dict) or not registrations:
        raise CudaBackendUnavailable(f"{module} exposes no FFI registrations")
    targets = {str(name) for name in registrations}
    required = _required_targets(str(getattr(extension, "__gpu_platform__", "cuda")))
    missing = sorted(required - targets)
    if missing:
        raise CudaBackendUnavailable(
            f"{module} is missing required streamed FFI targets: " + ", ".join(missing)
        )
    from ._cuda_abi import CUDA_MSTEP_TARGET

    known = required | ({CUDA_MSTEP_TARGET} & targets)
    return {name: registrations[name] for name in sorted(known)}


def register_cuda_ffi_targets(module: ModuleType | None = None) -> tuple[str, ...]:
    """Register extension-provided typed FFI handlers exactly once."""
    global _REGISTERED_MODULE
    extension = _load_extension() if module is None else module
    if _REGISTERED_MODULE is extension:
        return tuple(sorted(_validated_registrations(extension)))
    registrations = _validated_registrations(extension)
    platform = _XLA_PLATFORMS[_extension_platform(extension)]
    # Older JAXLIB releases lack this optional enum. Keep its lookup inside
    # CUDA registration so portable imports do not require CUDA-era APIs.
    from jaxlib import xla_client

    target_traits = getattr(xla_client, "CustomCallTargetTraits", None)
    supports_registration_traits = target_traits is not None
    for name, capsule in registrations.items():
        options = {"platform": platform, "api_version": 1}
        if target_traits is not None and supports_registration_traits:
            options["traits"] = (
                target_traits.COMMAND_BUFFER_COMPATIBLE
                if str(name)
                in {
                    CUDA_STREAMED_TARGET,
                    CUDA_ADJOINT_TARGET,
                    CUDA_MATERIAL_ADJOINT_TARGET,
                }
                else target_traits.DEFAULT
            )
        try:
            jax.ffi.register_ffi_target(str(name), capsule, **options)
        except RuntimeError as error:
            if "plugin does not support custom call traits" not in str(error):
                raise
            # Older CUDA PJRT plugins require traits in the native FFI metadata
            # rather than the Python registration API. Both declare the same
            # capture contract; retain compatibility without hiding other errors.
            supports_registration_traits = False
            jax.ffi.register_ffi_target(
                str(name), capsule, platform=platform, api_version=1
            )
    _REGISTERED_MODULE = extension
    return tuple(sorted(str(name) for name in registrations))


def _cpu_extension_installed() -> bool:
    """Whether a GPU-less process can use the native CPU (OpenMP) build."""
    try:
        importlib.import_module(_CPU_EXTENSION_MODULE)
    except ImportError:
        return False
    return True


def cuda_backend_status(*, register: bool = True) -> CudaBackendStatus:
    devices = _gpu_devices()
    device_names = tuple(
        str(getattr(device, "device_kind", device)) for device in devices
    )
    capabilities = tuple(_compute_capability(device) for device in devices)
    if not devices and not _cpu_extension_installed():
        return CudaBackendStatus(
            False,
            None,
            None,
            (),
            device_names,
            capabilities,
            "JAX has no visible GPU devices and the native CPU build "
            f"({_CPU_EXTENSION_MODULE}) is not installed",
        )
    try:
        extension = _load_extension()
        registrations = _validated_registrations(extension)
        gpu_platform = _extension_platform(extension)
        targets = (
            register_cuda_ffi_targets(extension)
            if register
            else tuple(sorted(registrations))
        )
    except (ImportError, AttributeError, CudaBackendUnavailable) as exc:
        return CudaBackendStatus(
            False,
            None,
            None,
            (),
            device_names,
            capabilities,
            f"optional {_extension_module_name()} extension is unavailable: {exc}",
        )
    version = str(getattr(extension, "__version__", "unknown"))
    return CudaBackendStatus(
        True,
        version,
        CUDA_ABI_VERSION,
        targets,
        device_names,
        capabilities,
        gpu_platform=gpu_platform,
    )


def resolve_backend(backend: str | None) -> ResolvedBackend:
    """Resolve public backend policy without silently weakening explicit requests."""
    requested = normalize_backend(backend)
    if requested == "jax":
        return "jax"
    status = cuda_backend_status()
    if requested == "auto" and not status.available:
        return "jax"
    if requested == "auto" and status.gpu_platform != "cuda":
        # The HIP build is opt-in until it is tuned and validated on AMD.
        return "jax"
    if not status.available:
        raise CudaBackendUnavailable(
            f"backend={requested!r} was requested but is unavailable: "
            f"{status.reason}. The cuda_* backends run BeamZ's optional native "
            "component (beamz._cuda on NVIDIA, beamz._hip on AMD); build it, "
            "or use backend='jax'."
        )
    has_streamed = _required_targets(status.gpu_platform).issubset(status.targets)
    if requested == "cuda_mstep":
        from ._cuda_abi import CUDA_MSTEP_TARGET

        if not has_streamed or CUDA_MSTEP_TARGET not in status.targets:
            raise CudaBackendUnavailable(
                f"cuda_mstep target is missing from {_extension_module_name()}"
            )
        return "cuda_mstep"
    if requested == "cuda_streamed":
        if not has_streamed:
            raise CudaBackendUnavailable(
                f"cuda_streamed target is missing from {_extension_module_name()}"
            )
        return "cuda_streamed"
    if has_streamed:
        return "cuda_streamed"
    if requested == "auto":
        return "jax"
    raise CudaBackendUnavailable(
        f"{_extension_module_name()} has no compatible execution target"
    )
