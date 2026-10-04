"""Compilation, rasterization, and persistent result caching."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _native  # type: ignore[attr-defined]
from .result import RasterResult
from .schema import _CACHE_SCHEMA_VERSION, _ENGINE_VERSION, Grid, Scene


@dataclass(frozen=True, slots=True)
class RasterOptions:
    """Dielectric smoothing and a separate metal-interface policy.

    Metals have a principal Re(epsilon_r) < 1 at ``reference_frequency`` (Hz).
    Simulation uses the mean positive source center frequency when unspecified;
    standalone rasterization uses each dispersive material's fit-band midpoint.
    ``metal_smoothing="inherit"`` applies ``smoothing`` to every material.
    ``periodic_axes`` joins material fractions across the specified domain faces
    before smoothing. Simulation derives these axes from its boundaries.
    """

    quality: str = "balanced"
    smoothing: str = "farjadpour_diagonal"
    components: str = "all"
    metal_smoothing: str = "staircase"
    reference_frequency: float | None = None
    periodic_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        choices = {
            "metal_smoothing": {"staircase", "inherit"},
            "quality": {"fast", "balanced", "reference"},
            "smoothing": {"volume", "farjadpour_diagonal", "farjadpour_full"},
            "components": {"all", "two_dimensional_tm", "two_dimensional_te"},
        }
        for name, allowed in choices.items():
            value = str(getattr(self, name)).strip().lower()
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}.")
            object.__setattr__(self, name, value)

        axes = tuple(sorted(set(self.periodic_axes)))
        if set(axes) - {"x", "y", "z"}:
            raise ValueError("periodic_axes must contain only x, y, z.")
        object.__setattr__(self, "periodic_axes", axes)

        if self.reference_frequency is not None:
            frequency = float(self.reference_frequency)
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError("reference_frequency must be finite positive Hz.")
            object.__setattr__(self, "reference_frequency", frequency)


def staircase_materials(scene: Scene, options: RasterOptions) -> tuple[int, ...]:
    """Resolve optical metal identity before replacing media by indicator fields."""
    from beamz.design.dispersion import PoleResidue

    if options.metal_smoothing == "inherit":
        return ()
    metals = []
    for index, material in enumerate(scene.materials):
        if isinstance(material, PoleResidue):
            frequency = options.reference_frequency
            if frequency is None:
                frequency = sum(material.frequency_range) / 2
            minimum = float(np.real(material.eps_model(frequency)))
        else:
            xx, yy, zz, xy, xz, yz = material.epsilon_r
            minimum = np.linalg.eigvalsh(((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)))[0]
        if minimum < 1:
            metals.append(index)
    return tuple(metals)


class CompiledScene:
    """A validated scene that can be rasterized repeatedly."""

    def __init__(self, scene: Scene):
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene.")
        self._scene = scene
        self._native = _native.compile_scene(scene.to_json())
        self.hash = str(self._native.scene_hash)

    def rasterize(
        self,
        grid: Grid,
        *,
        options: RasterOptions | None = None,
        cache_directory: str | Path | None = None,
        _staircase_materials: tuple[int, ...] | None = None,
    ) -> RasterResult:
        if not isinstance(grid, Grid):
            raise TypeError("grid must be a Grid.")
        options = RasterOptions() if options is None else options
        if not isinstance(options, RasterOptions):
            raise TypeError("options must be RasterOptions.")
        metals = (
            staircase_materials(self._scene, options)
            if _staircase_materials is None
            else _staircase_materials
        )
        cache_path = self._cache_path(grid, options, cache_directory, metals)
        if cache_path is not None and cache_path.exists():
            try:
                return RasterResult._from_cache(
                    cache_path,
                    scene_hash=self.hash,
                    grid_edges=grid.edges,
                    smoothing=options.smoothing,
                )
            except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile):
                cache_path.unlink(missing_ok=True)

        native = self._native.rasterize(
            tuple(edges.tolist() for edges in grid.edges),
            options.quality,
            options.smoothing,
            options.components,
            metals,
            tuple(axis in options.periodic_axes for axis in "xyz"),
        )
        result = RasterResult(
            native,
            grid_edges=grid.edges,
            smoothing=options.smoothing,
        )
        if cache_path is not None:
            temporary = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.npz"
            )
            try:
                np.savez_compressed(temporary, **result._cache_payload())
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return result

    def _cache_path(
        self,
        grid: Grid,
        options: RasterOptions,
        directory: str | Path | None,
        metals: tuple[int, ...],
    ) -> Path | None:
        if directory is None:
            return None
        digest = hashlib.blake2b(digest_size=20)
        digest.update(self.hash.encode())
        for edges in grid.edges:
            digest.update(edges.tobytes())
        digest.update(
            json.dumps(
                {
                    "schema": _CACHE_SCHEMA_VERSION,
                    "engine": _ENGINE_VERSION,
                    "quality": options.quality,
                    "smoothing": options.smoothing,
                    "components": options.components,
                    "metal_smoothing": options.metal_smoothing,
                    "reference_frequency": options.reference_frequency,
                    "staircase_materials": metals,
                    "periodic_axes": options.periodic_axes,
                },
                sort_keys=True,
            ).encode()
        )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest.hexdigest()}.npz"


def compile_scene(scene: Scene) -> CompiledScene:
    return CompiledScene(scene)


def rasterize(
    scene: Scene,
    grid: Grid,
    *,
    options: RasterOptions | None = None,
    cache_directory: str | Path | None = None,
    _staircase_materials: tuple[int, ...] | None = None,
) -> RasterResult:
    return compile_scene(scene).rasterize(
        grid,
        options=options,
        cache_directory=cache_directory,
        _staircase_materials=_staircase_materials,
    )
