"""
Save/load the FULL workspace — setup (structures/sources/monitors/
region) AND monitor results — as a single HDF5 file.

HDF5 is the established format for exactly this in scientific/HPC
computing: hierarchical structure + large numeric arrays + attached
metadata, all in one portable, language-agnostic, compressible file.
It's also what beamz's own `SimulationResults.to_xarray()` output
naturally maps onto (an `xarray.Dataset` writes to NetCDF, which IS HDF5
under the hood) — this standardizes on a format the underlying data
already has a natural path into, rather than inventing a bespoke one.

Distinct from serialization.py's export_script()/import_script(), which
write/read a plain, human-readable, re-runnable .py file — genuinely
useful as a readable/git-diffable artifact and a "this is the exact
beamz code your GUI represents" teaching aid, but setup-only, no results.
Both are offered as separate File-menu actions: this module is "save my
whole session", that one is "show me the code".

Honest caveat: the results-saving path (`_save_results`/the results half
of `load_workspace`) was written without being able to run beamz
interactively to confirm `SimulationResults.to_xarray()`'s exact output
shape. It's wrapped defensively — a mismatch there raises separately
rather than losing an otherwise-successful setup save — but should be
verified against a real run before being trusted for anything valuable.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Optional

import beamz as bz
import numpy as np

BEAMZ_GUI_FORMAT_VERSION = 1


def _is_beamz_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type) and hasattr(bz, type(value).__name__)


def _to_jsonable(value: Any) -> Any:
    """Recursively converts a recipe value into something json.dumps can
    handle:
      - numpy arrays -> a tagged dict (so _from_jsonable can convert back
        to an ndarray rather than leaving it as a plain list).
      - nested beamz dataclass instances (Material, GaussianPulse, ...)
        -> {"__beamz_class__": ..., "kwargs": {...}}, recursing into
        THEIR fields the same way.
      - tuples -> a tagged list (JSON has no tuple type; leaving this
        untagged would silently turn e.g. `position` into a list on
        reload, and beamz's own validation is stricter about tuples in
        some paths).
    Material is special-cased before the generic dataclass branch for the
    same reason serialization.py's _pyrepr special-cases it: its actual
    dataclass fields are private, internal storage already expanded into
    full tensor form — NOT its public constructor parameter names.
    Recursing into the raw fields would produce kwargs Material's own
    constructor doesn't accept.
    """
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist()}
    if isinstance(value, bz.Material):
        return {
            "__beamz_class__": "Material",
            "kwargs": {
                "permittivity": _to_jsonable(value.permittivity),
                "permeability": _to_jsonable(value.permeability),
                "conductivity": _to_jsonable(value.conductivity),
            },
        }
    if _is_beamz_instance(value):
        kwargs = {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
        return {"__beamz_class__": type(value).__name__, "kwargs": kwargs}
    if isinstance(value, tuple):
        return {"__tuple__": [_to_jsonable(v) for v in value]}
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return np.array(value["__ndarray__"])
        if "__beamz_class__" in value:
            cls = getattr(bz, value["__beamz_class__"])
            kwargs = {k: _from_jsonable(v) for k, v in value["kwargs"].items()}
            return cls(**kwargs)
        if "__tuple__" in value:
            return tuple(_from_jsonable(v) for v in value["__tuple__"])
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


def _export_setup(connector) -> dict[str, Any]:
    entries = [{**entry, "recipe": _to_jsonable(entry["recipe"])} for entry in connector.snapshot()]
    return {
        "format_version": BEAMZ_GUI_FORMAT_VERSION,
        "design_width": connector.design_width,
        "design_height": connector.design_height,
        "design_depth": connector.design_depth,
        "objects": entries,
    }


def _import_setup(connector, setup: dict[str, Any]) -> None:
    connector.design_width = setup.get("design_width", connector.design_width)
    connector.design_height = setup.get("design_height", connector.design_height)
    connector.design_depth = setup.get("design_depth", connector.design_depth)
    decoded = [{**entry, "recipe": _from_jsonable(entry["recipe"])} for entry in setup["objects"]]
    connector.restore(decoded)


def save_workspace(connector, path: str, results: Optional[Any] = None) -> None:
    """Writes `path` as an HDF5 file containing the current setup, and —
    if a completed run's `results` (a beamz.SimulationResults) is passed
    — its data too (best-effort; see module docstring).
    """
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["beamz_gui_format_version"] = BEAMZ_GUI_FORMAT_VERSION
        f.attrs["setup_json"] = json.dumps(_export_setup(connector))
        if results is not None:
            _save_results(f, results)


def _save_results(h5file, results: Any) -> None:
    """Walks whatever `results.to_xarray()` returns and writes each data
    variable / coordinate as its own HDF5 dataset under /results.
    Wrapped so a shape mismatch here (see module docstring's caveat)
    doesn't take down the setup save that, by this point, already
    succeeded — the error is recorded as an attribute instead of raised.
    """
    try:
        dataset = results.to_xarray()
        group = h5file.create_group("results")
        for name, coord in dataset.coords.items():
            group.create_dataset(f"coords/{name}", data=coord.values)
        for name, var in dataset.data_vars.items():
            ds = group.create_dataset(f"data/{name}", data=var.values)
            ds.attrs["dims"] = json.dumps(list(var.dims))
    except Exception as exc:  # noqa: BLE001 — best-effort, see docstring
        h5file.attrs["results_save_error"] = str(exc)


def load_workspace(connector, path: str) -> Optional[Any]:
    """Replaces the connector's scene with what's saved in `path`.
    Returns an xarray.Dataset of the saved results if present, else None.
    Raw HDF5 data, not reconstructed into a beamz.SimulationResults —
    beamz has no public constructor for building one from saved data, so
    the caller gets the underlying arrays to work with directly instead
    of a fake SimulationResults pretending to be a fresh run's output.
    """
    import h5py

    with h5py.File(path, "r") as f:
        setup = json.loads(f.attrs["setup_json"])
        _import_setup(connector, setup)

        if "results" not in f:
            return None
        try:
            import xarray as xr

            group = f["results"]
            data_vars = {}
            for name in group.get("data", {}):
                ds = group["data"][name]
                dims = json.loads(ds.attrs.get("dims", "[]")) or [f"dim_{i}" for i in range(ds.ndim)]
                data_vars[name] = (dims, ds[()])
            coords = {name: group["coords"][name][()] for name in group.get("coords", {})}
            return xr.Dataset(data_vars=data_vars, coords=coords)
        except Exception:
            return None