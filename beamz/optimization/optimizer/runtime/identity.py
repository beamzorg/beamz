"""Stable identity for objective callables."""

import inspect
import types

from beamz._cache_tokens import cache_token


def _function_token(function, seen=None):
    """Include closure/configuration values, not merely a callable's name."""
    seen = set() if seen is None else seen
    if id(function) in seen:
        return (function.__module__, function.__qualname__)
    if not isinstance(function, types.FunctionType):
        raise TypeError("For callable objects, supply objective_id.")
    seen.add(id(function))

    def token(value):
        if isinstance(value, types.ModuleType):
            return ("module", value.__name__)
        if isinstance(value, types.FunctionType):
            return (
                _function_token(value, seen)
                if value.__module__ == function.__module__
                else cache_token(value)
            )
        if isinstance(value, types.CodeType):
            return (
                value.co_code.hex(),
                tuple(token(v) for v in value.co_consts),
                value.co_names,
            )
        return cache_token(value)

    closure = inspect.getclosurevars(function)
    return (
        token(function.__code__),
        token(function.__defaults__),
        token(function.__kwdefaults__),
        tuple(
            sorted(
                (name, token(value))
                for name, value in {**closure.globals, **closure.nonlocals}.items()
            )
        ),
    )


# Preserve checkpoint fingerprints created before optimization modules were split.
_LEGACY_MODULES = {
    "beamz.optimization.design.spec": "beamz.optimization.design",
    "beamz.optimization.design.initialization": "beamz.optimization.region",
    "beamz.optimization.design.region": "beamz.optimization.region",
    "beamz.optimization.design.masked": "beamz.optimization.region",
    "beamz.optimization.design.validation": "beamz.optimization._validation",
    "beamz.optimization.topology.penalty": "beamz.optimization.region",
    "beamz.optimization.design.geometry.contour": "beamz.optimization.polygonize",
    "beamz.optimization.design.geometry.pixels": "beamz.optimization.polygonize",
    "beamz.optimization.design.geometry.density": "beamz.optimization.polygonize",
    "beamz.optimization.design.geometry.polygons": "beamz.optimization.polygonize",
    "beamz.optimization.design.geometry.export": "beamz.optimization.polygonize",
    "beamz.optimization.objectives.base": "beamz.optimization.objectives",
    "beamz.optimization.objectives.mode": "beamz.optimization.objectives",
    "beamz.optimization.objectives.composite": "beamz.optimization.objectives",
    "beamz.optimization.objectives.data": "beamz.optimization.data",
    "beamz.optimization.objectives.projection": "beamz.optimization.modes",
    "beamz.optimization.optimizer.optimizer": "beamz.optimization.optimizer",
    "beamz.optimization.optimizer.problem": "beamz.optimization.problem",
    "beamz.optimization.optimizer.result": "beamz.optimization.result",
    "beamz.optimization.optimizer.schedules": "beamz.optimization.schedules",
    "beamz.optimization.optimizer.checkpoint": "beamz.optimization._checkpoint",
    "beamz.optimization.optimizer.runtime.loop": "beamz.optimization.optimizer",
    "beamz.optimization.optimizer.runtime.identity": "beamz.optimization.optimizer",
    "beamz.optimization.optimizer.runtime.load": "beamz.optimization.result",
    "beamz.optimization.topology.projections": "beamz.optimization.projections",
    "beamz.optimization.topology.gradient": "beamz.optimization.topology",
    "beamz.optimization.topology.filtering": "beamz.optimization.autodiff",
    "beamz.optimization.topology.morphology": "beamz.optimization.autodiff",
    "beamz.optimization.topology.transforms.conic": "beamz.optimization.autodiff",
    "beamz.optimization.topology.transforms.density": "beamz.optimization.autodiff",
    "beamz.optimization.topology.transforms.differentiation": "beamz.optimization.autodiff",
    "beamz.optimization.topology.transforms.mask": "beamz.optimization.topology",
    "beamz.optimization.topology.transforms.padding": "beamz.optimization.topology",
}


def _legacy_checkpoint_token(value: object = "") -> object:
    """Translate only module and class identity tokens used in old checkpoints."""
    if not isinstance(value, tuple):
        return value
    parts = list(value)
    if len(parts) == 3 and parts[0] == "callable" and isinstance(parts[1], str):
        if (
            parts[1] == "beamz.optimization.objectives.projection"
            and parts[2] == "bind_modal_data"
        ):
            parts[1] = "beamz.optimization.data"
        else:
            parts[1] = _LEGACY_MODULES.get(parts[1], parts[1])
    elif len(parts) == 2 and parts[0] == "module" and isinstance(parts[1], str):
        parts[1] = _LEGACY_MODULES.get(parts[1], parts[1])
    elif len(parts) >= 2 and isinstance(parts[0], str):
        if isinstance(parts[1], tuple) or parts[1] in ("cache_spec", "canonical_spec"):
            module, separator, name = parts[0].rpartition(".")
            parts[0] = _LEGACY_MODULES.get(module, module) + separator + name
        elif isinstance(parts[1], str) and parts[0] in _LEGACY_MODULES:
            parts[0] = _LEGACY_MODULES[parts[0]]
    return tuple(_legacy_checkpoint_token(part) for part in parts)
