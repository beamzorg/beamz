"""Versioned, non-pickle archives for detached simulation results.

Only explicitly registered immutable value classes can be reconstructed. Array
payloads retain their dtype, including complex DFTs and packed Yee tensors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

FORMAT = "beamz.simulation.results/2"


def _classes():
    from beamz.design.grid import RectilinearGrid
    from beamz.devices._placement import SnappedInterval, SnappedRegion
    from beamz.devices.modes.specs import ModeSpec
    from beamz.devices.monitors.monitors import FieldMonitor, FluxMonitor, ModeMonitor
    from beamz.devices.sources.specs import (
        CustomSource,
        GaussianBeamSource,
        GaussianSource,
        ModeSource,
        PlaneWaveSource,
    )
    from beamz.devices.sources.time import (
        GaussianBandPulse,
        GaussianPulse,
        SampledSignal,
    )

    from .results import (
        FieldMetadata,
        MaterialRegion,
        MonitorResults,
        RunTermination,
        SimulationMetadata,
        SimulationPerformance,
        SimulationResults,
    )

    types = (
        RectilinearGrid,
        SnappedInterval,
        SnappedRegion,
        ModeSpec,
        FieldMonitor,
        FluxMonitor,
        ModeMonitor,
        CustomSource,
        GaussianBeamSource,
        GaussianSource,
        ModeSource,
        PlaneWaveSource,
        GaussianBandPulse,
        GaussianPulse,
        SampledSignal,
        FieldMetadata,
        MaterialRegion,
        MonitorResults,
        RunTermination,
        SimulationMetadata,
        SimulationPerformance,
        SimulationResults,
    )
    return {cls.__name__: cls for cls in types}


def save_results(result, path):
    from .results import SimulationResults

    if not isinstance(result, SimulationResults):
        raise TypeError("Expected SimulationResults")
    registry = _classes()
    arrays = {}

    def encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Result archives cannot contain object arrays")
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if isinstance(value, np.generic):
            return encode(value.item())
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if np.isfinite(value) else {"float": str(value)}
        if isinstance(value, complex):
            return {"complex": [encode(value.real), encode(value.imag)]}
        if isinstance(value, slice):
            return {"slice": [encode(v) for v in (value.start, value.stop, value.step)]}
        if isinstance(value, Mapping):
            return {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"tuple": [encode(v) for v in value]}
        if isinstance(value, frozenset):
            return {"frozenset": [encode(v) for v in value]}
        if is_dataclass(value) and registry.get(type(value).__name__) is type(value):
            return {
                "class": type(value).__name__,
                "fields": {
                    f.name: encode(getattr(value, f.name))
                    for f in fields(value)
                    if f.init
                },
            }
        raise TypeError(f"Unsupported result archive value: {type(value).__name__}")

    manifest = {"format": FORMAT, "result": encode(result)}
    # Opening explicitly prevents numpy from silently appending another suffix.
    with open(path, "wb") as stream:
        np.savez_compressed(
            stream, manifest=np.asarray(json.dumps(manifest, allow_nan=False)), **arrays
        )


def load_results(path):
    from .results import SimulationResults

    registry = _classes()
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest.get("format") not in {"beamz.simulation.results/1", FORMAT}:
            raise ValueError("Unsupported BeamZ result archive version")

        def decode(value: Any) -> Any:
            if not isinstance(value, dict):
                return value
            if "array" in value:
                return np.array(archive[value["array"]], copy=True)
            if "float" in value:
                return float(value["float"])
            if "complex" in value:
                return complex(*(decode(v) for v in value["complex"]))
            if "slice" in value:
                return slice(*(decode(v) for v in value["slice"]))
            if "mapping" in value:
                return {decode(k): decode(v) for k, v in value["mapping"]}
            if "tuple" in value:
                return tuple(decode(v) for v in value["tuple"])
            if "frozenset" in value:
                return frozenset(decode(v) for v in value["frozenset"])
            if "class" in value:
                name = value["class"]
                if name not in registry:
                    raise ValueError(f"Unsupported result archive class: {name}")
                return registry[name](
                    **{k: decode(v) for k, v in value["fields"].items()}
                )
            raise ValueError("Invalid result archive record")

        result = decode(manifest["result"])
    if not isinstance(result, SimulationResults):
        raise ValueError("Archive does not contain SimulationResults")
    return result
