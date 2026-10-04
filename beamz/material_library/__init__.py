"""Bundled dispersive media, indexed by material and named fit variant.

Example: ``material_library["SiO2"]["Malitson1965"]`` is a BeamZ PoleResidue.
Data and license notices ship with BeamZ; no external solver is required.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from io import StringIO
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from beamz.design.dispersion import PoleResidue


@dataclass(frozen=True)
class MaterialVariant:
    """A fitted medium and its provenance.

    ``medium.frequency_range`` gives the fit's validity interval in Hz.
    ``nk_data`` contains source samples or synthetic design targets.
    """

    medium: PoleResidue
    description: str
    source: str
    license: str
    _nk_file: str | None = None
    fit: Mapping[str, object] | None = None
    references: str = ""
    conditions: str = ""

    @property
    def nk_data(self) -> NDArray[np.float64] | None:
        """Fresh (N, 3) array of wavelength [m], n, k, or None if unavailable.

        Mutating this array does not modify the library. Physical material
        samples and synthetic filter targets are loaded only on access.
        """
        if self._nk_file is None:
            return None
        text = files(__name__).joinpath("data").joinpath(self._nk_file).read_text()
        samples = np.loadtxt(StringIO(text), delimiter=",")
        return samples


@dataclass(frozen=True)
class MaterialItem:
    """Named material with read-only variants and an explicit default fit."""

    name: str
    variants: Mapping[str, MaterialVariant]
    default: str

    def __post_init__(self) -> None:
        if self.default not in self.variants:
            raise ValueError(f"Unknown default variant: {self.default!r}")
        object.__setattr__(self, "variants", MappingProxyType(dict(self.variants)))

    def __getitem__(self, variant: str) -> PoleResidue:
        return self.variants[variant].medium

    @property
    def medium(self) -> PoleResidue:
        """Default fit, usable directly in BeamZ structures and simulations."""
        return self[self.default]


def _load_library() -> Mapping[str, MaterialItem]:
    catalog = json.loads(
        files(__name__).joinpath("data").joinpath("catalog.json").read_text()
    )
    return MappingProxyType(
        {
            key: MaterialItem(
                name=item["name"],
                default=item["default"],
                variants={
                    name: MaterialVariant(
                        medium=PoleResidue.from_spec(variant["medium"]),
                        description=variant["description"],
                        source=variant["source"],
                        license=variant["license"],
                        _nk_file=variant.get("nk_file"),
                        fit=MappingProxyType(variant.get("fit", {})),
                        references=variant.get("references", ""),
                        conditions=variant.get("conditions", ""),
                    )
                    for name, variant in item["variants"].items()
                },
            )
            for key, item in catalog.items()
        }
    )


material_library = _load_library()

__all__ = ["MaterialItem", "MaterialVariant", "material_library"]
