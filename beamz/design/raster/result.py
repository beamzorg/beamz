"""Immutable solver-facing raster result."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from beamz.design._material_layout import CELL_TENSORS, material_fields, support_shapes

from .schema import _CACHE_SCHEMA_VERSION, _ENGINE_VERSION, Grid

_TENSORS = CELL_TENSORS
_METADATA = {
    "cache_schema",
    "engine_version",
    "scene_hash",
    "diagnostics_json",
    "grid_x_edges",
    "grid_y_edges",
    "grid_z_edges",
    *(f"tensor_{name}" for name in _TENSORS),
}


@dataclass(frozen=True, slots=True, init=False)
class RasterResult:
    grid: Grid
    smoothing: str
    yee_tensors: Mapping[str, np.ndarray]
    tensors: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, Any]
    scene_hash: str
    cache_hit: bool

    def __init__(
        self,
        native: Any,
        *,
        grid: Grid,
        smoothing: str,
    ):
        payload = dict(native.take_arrays())
        diagnostics = json.loads(str(payload.pop("diagnostics_json")))
        scene_hash = str(payload.pop("scene_hash"))
        tensors = {name: payload.pop(f"tensor_{name}") for name in _TENSORS}
        self._assign(
            grid,
            smoothing,
            payload,
            tensors,
            diagnostics,
            scene_hash,
            cache_hit=False,
        )

    def _assign(
        self,
        grid: Grid,
        smoothing: str,
        yee_tensors: Mapping[str, Any],
        tensors: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        scene_hash: str,
        *,
        cache_hit: bool,
    ) -> None:
        assign = object.__setattr__
        cell_shape = grid.shape_zyx
        assign(self, "grid", grid)
        assign(self, "smoothing", str(smoothing))
        for name, values, shapes in (
            ("yee_tensors", yee_tensors, support_shapes(cell_shape, raster=True)),
            (
                "tensors",
                {name: tensors[name] for name in _TENSORS},
                dict.fromkeys(_TENSORS, cell_shape),
            ),
        ):
            assign(
                self,
                name,
                material_fields(
                    values,
                    shapes,
                    compact=True,
                    copy=False,
                    label="native raster output",
                ),
            )
        assign(self, "diagnostics", MappingProxyType(dict(diagnostics)))
        assign(self, "scene_hash", str(scene_hash))
        assign(self, "cache_hit", bool(cache_hit))

    @classmethod
    def _from_arrays(
        cls,
        *,
        grid_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
        smoothing: str,
        yee_tensors: Mapping[str, Any],
        tensors: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        scene_hash: str,
    ) -> RasterResult:
        """Validate and freeze fields assembled outside the native raster call."""
        result = cls.__new__(cls)
        result._assign(
            Grid(*grid_edges),
            smoothing,
            yee_tensors,
            tensors,
            diagnostics,
            scene_hash,
            cache_hit=False,
        )
        return result

    @property
    def grid_edges(self):
        return self.grid.edges

    @property
    def is_uniform(self) -> bool:
        return all(
            np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-12, atol=0.0)
            for edges in self.grid_edges
        )

    def _cache_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cache_schema": np.asarray(_CACHE_SCHEMA_VERSION),
            "engine_version": np.asarray(_ENGINE_VERSION),
            "scene_hash": np.asarray(self.scene_hash),
            "diagnostics_json": np.asarray(json.dumps(dict(self.diagnostics))),
        }
        payload.update(
            {f"tensor_{name}": values for name, values in self.tensors.items()}
        )
        payload.update(
            {
                f"grid_{axis}_edges": edges
                for axis, edges in zip(("x", "y", "z"), self.grid_edges, strict=True)
            }
        )
        payload.update(
            {f"yee__{name}": values for name, values in self.yee_tensors.items()}
        )
        return payload

    @classmethod
    def _from_cache(
        cls,
        path: Path,
        *,
        scene_hash: str,
        grid_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
        smoothing: str,
    ) -> RasterResult:
        with np.load(path, allow_pickle=False) as values:
            if (
                int(values["cache_schema"]) != _CACHE_SCHEMA_VERSION
                or str(values["engine_version"]) != _ENGINE_VERSION
                or str(values["scene_hash"]) != scene_hash
            ):
                raise ValueError("Raster cache identity does not match.")
            cached_edges = tuple(
                values[f"grid_{axis}_edges"].copy() for axis in ("x", "y", "z")
            )
            if any(
                not np.array_equal(actual, expected)
                for actual, expected in zip(cached_edges, grid_edges, strict=True)
            ):
                raise ValueError("Raster cache grid does not match.")
            unknown = {
                name
                for name in values.files
                if name not in _METADATA and not name.startswith("yee__")
            }
            if unknown:
                raise ValueError(f"Raster cache has unknown fields: {sorted(unknown)}")
            result = cls.__new__(cls)
            result._assign(
                Grid(*cached_edges),
                smoothing,
                {
                    name.removeprefix("yee__"): values[name].copy()
                    for name in values.files
                    if name.startswith("yee__")
                },
                {name: values[f"tensor_{name}"].copy() for name in _TENSORS},
                json.loads(str(values["diagnostics_json"])),
                scene_hash,
                cache_hit=True,
            )
            return result
