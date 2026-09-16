"""
Save/load the scene as a plain .py script — the same `bz.ClassName(...)`
lines already shown live in the Script Console (see console.py's
docstring: "nothing more than an interface to beamz"), just written to a
file instead of echoed to a log. Loading it back is genuinely just
running that Python against the connector's shared namespace and
resyncing — the same mechanism the console already uses for typed
commands, not a separate file format to keep in sync with the rest of
the app.

The one non-beamz piece — the SimulationRegion, a GUI-only concept with
no real beamz class (see model.REGION_KIND) — is saved as a plain dict
literal assigned to a recognizable variable name, `BEAMZ_GUI_REGION`.
It's still valid, inspectable Python; the loader just looks for that one
variable name specially since there's no `bz.SimulationRegion()` call to
round-trip through.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import beamz as bz
import numpy as np

from .model import BeamzConnector

REGION_VAR_NAME = "BEAMZ_GUI_REGION"


def _pyrepr(value: Any) -> str:
    """repr() with two adjustments needed for the result to be valid,
    standalone Python (only `import beamz as bz` is assumed):
      - numpy arrays shown as a plain list literal instead of
        `array([...])`.
      - nested beamz objects (e.g. a source's `source_time=GaussianPulse
        (...)`, or a structure's `material=Material(...)`) prefixed with
        `bz.` — plain repr() gives exactly the right
        "ClassName(field=value, ...)" shape already, it just doesn't know
        to qualify the class name. Recurses into the nested object's OWN
        fields the same way, in case of further nesting or an array
        field inside it.
    """
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, bz.Material):
        # Special-cased BEFORE the generic dataclass-recursion branch
        # below: Material's actual dataclass fields are private, internal
        # storage (`_epsilon_r` etc.), already expanded into full 6-value
        # tensor form — NOT its public constructor parameter names
        # (permittivity/permeability/conductivity). Recursing into the
        # raw fields produced `bz.Material(_epsilon_r=(12.11, 12.11,
        # 12.11, 0.0, 0.0, 0.0), ...)`, which isn't even valid — Material
        # doesn't accept an `_epsilon_r` keyword at all. Its own PUBLIC
        # properties round-trip as the original scalar cleanly instead
        # (confirmed directly), so those are what get serialized.
        return f"bz.Material(permittivity={_pyrepr(value.permittivity)}, permeability={_pyrepr(value.permeability)}, conductivity={_pyrepr(value.conductivity)})"
    if dataclasses.is_dataclass(value) and not isinstance(value, type) and hasattr(bz, type(value).__name__):
        inner = ", ".join(
            f"{f.name}={_pyrepr(getattr(value, f.name))}" for f in dataclasses.fields(value) if f.init
        )
        return f"bz.{type(value).__name__}({inner})"
    return repr(value)


def export_script(connector: BeamzConnector) -> str:
    """The current scene as a standalone, re-runnable .py file."""
    lines = ["import beamz as bz", ""]

    regions = connector.by_category("region")
    if regions:
        lines.append(f"{REGION_VAR_NAME} = {regions[0].recipe!r}")
        lines.append("")

    for so in connector.all_structures():
        if so.category == "region":
            continue
        kwargs = ", ".join(f"{k}={_pyrepr(v)}" for k, v in so.recipe.items())
        lines.append(f"{so.name} = bz.{so.class_name}({kwargs})")

    return "\n".join(lines) + "\n"


def import_script(connector: BeamzConnector, code: str) -> list[str]:
    """Replaces the current scene with whatever `code` defines. Clears
    every existing structure/source/monitor/region first (Load is a
    full scene swap, not a merge), then executes `code` into the SAME
    shared namespace the console uses and reconciles it the same way a
    typed console command would — see model.py's sync_from_script.
    Returns the sids that ended up in the scene.
    """
    for so in list(connector.all_structures()):
        connector.remove_structure(so.id)

    exec(compile(code, "<loaded_script>", "exec"), connector.namespace)

    region_dict = connector.namespace.pop(REGION_VAR_NAME, None)
    if isinstance(region_dict, dict):
        connector.add_region(**region_dict)

    return connector.sync_from_script()