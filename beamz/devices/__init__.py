"""Immutable sources, monitors, ports, and boundary specifications."""

import sys as _sys
from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import Any as _Any


class _LegacyModule(_ModuleType):
    def __init__(self, name: str = "", target: str = "") -> None:
        super().__init__(name)
        self.__dict__["_target"] = target

    def __getattr__(self, name: str = "") -> _Any:
        if name == "__all__":
            target = _import_module(self.__dict__["_target"])
            return getattr(
                target, "__all__", [n for n in dir(target) if not n.startswith("_")]
            )
        if name in {"__file__", "__cached__"}:
            return getattr(_import_module(self.__dict__["_target"]), name)
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(_import_module(self.__dict__["_target"]), name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("__") or name == "_target":
            super().__setattr__(name, value)
        else:
            setattr(_import_module(self.__dict__["_target"]), name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("__") or name == "_target":
            super().__delattr__(name)
        else:
            delattr(_import_module(self.__dict__["_target"]), name)

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__()) | set(dir(_import_module(self.__dict__["_target"])))
        )


def _install_legacy_modules(
    package: str = "", targets: tuple[tuple[str, str], ...] = ()
) -> None:
    namespace = _sys.modules[package].__dict__
    for name, target in targets:
        full_name = f"{package}.{name}"
        namespace[name] = _sys.modules.setdefault(
            full_name, _LegacyModule(full_name, target)
        )


_install_legacy_modules(
    __name__,
    (
        ("_boundary_compile", "beamz.devices.common._boundary_compile"),
        ("_immutable", "beamz.devices.common._immutable"),
        ("_placement", "beamz.devices.common._placement"),
        ("visualization", "beamz.devices.common.visualization"),
    ),
)

from beamz.devices.boundaries import PEC, PML, Absorber, Bloch, Periodic  # noqa: E402

__all__ = ["Absorber", "PEC", "PML", "Bloch", "Periodic"]
