import h5py
import json
import datetime
import dataclasses
from typing import Any

import beamz as bz
import numpy as np
import xarray as xr


def _is_beamz_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type) and hasattr(bz, type(value).__name__)


def _to_jsonable(value: Any) -> Any:
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
        return {
            "__beamz_class__": type(value).__name__,
            "kwargs": {
                f.name: _to_jsonable(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if f.init
            },
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_to_jsonable(item) for item in value]}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return np.array(value["__ndarray__"])
        if "__beamz_class__" in value:
            cls = getattr(bz, value["__beamz_class__"])
            return cls(**{key: _from_jsonable(item) for key, item in value["kwargs"].items()})
        if "__tuple__" in value:
            return tuple(_from_jsonable(item) for item in value["__tuple__"])
        return {key: _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def _serialize_model(connector) -> dict[str, Any]:
    return {
        "design_width": connector.design_width,
        "design_height": connector.design_height,
        "design_depth": connector.design_depth,
        "objects": [
            {**entry, "recipe": _to_jsonable(entry["recipe"])}
            for entry in connector.snapshot()
        ],
    }


def _deserialize_model(connector, state: dict[str, Any]) -> None:
    connector.design_width = state.get("design_width", connector.design_width)
    connector.design_height = state.get("design_height", connector.design_height)
    connector.design_depth = state.get("design_depth", connector.design_depth)
    connector.restore([
        {**entry, "recipe": _from_jsonable(entry["recipe"])}
        for entry in state.get("objects", [])
    ])

CLIPBOARD_MARKER = "__BEAMZ_GUI_CLIPBOARD_V1__"


def entries_to_clipboard_text(connector, sids: list[str]) -> str:
    """JSON text for Copy — same `_to_jsonable` pass `_serialize_model()`
    uses for full-workspace save, just over `connector.export_entries()`
    instead of the whole scene. Prefixed with a marker so Paste can tell
    actual BEAMZ-copied objects apart from arbitrary clipboard text
    (which the system clipboard may hold anything at all) before trying
    to `json.loads()` it — and, since this is the OS clipboard, this is
    also what makes copy/paste work BETWEEN two separate running windows
    of the app, not just within one.
    """
    entries = [{**entry, "recipe": _to_jsonable(entry["recipe"])} for entry in connector.export_entries(sids)]
    return CLIPBOARD_MARKER + json.dumps(entries)


def entries_from_clipboard_text(text: str) -> list[dict[str, Any]]:
    """Inverse of `entries_to_clipboard_text()`. Raises ValueError if
    `text` doesn't carry the marker (i.e. it's not something Copy put
    there) rather than attempting — and likely failing confusingly deep
    inside `_from_jsonable` — to parse arbitrary clipboard text as if it
    were beamz JSON.
    """
    if not text.startswith(CLIPBOARD_MARKER):
        raise ValueError("Clipboard does not contain a copied BEAMZ GUI object.")
    entries = json.loads(text[len(CLIPBOARD_MARKER):])
    return [{**entry, "recipe": _from_jsonable(entry["recipe"])} for entry in entries]


class WorkspaceManager:
    def __init__(self, connector):
        self.connector = connector
        self.current_filepath = None

    def save_workspace(self, filepath: str, results: dict = None):
        """
        Saves the complete workspace (setup + results + metadata) to an HDF5 file.
        results: dict mapping monitor_name -> xarray.Dataset
        """
        # 1. Get the JSON setup state
        setup_dict = _serialize_model(self.connector)
        setup_json = json.dumps(setup_dict)

        # 2. Write Setup and Metadata via h5py
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['beamz_gui_version'] = '0.1.0'
            f.attrs['beamz_version'] = 'latest' # Pull from beamz.__version__ if available
            f.attrs['saved_at'] = datetime.datetime.now().isoformat()

            # Setup Dataset (variable length UTF-8 string)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('/setup', data=setup_json, dtype=dt)

        # 3. Append Results via xarray (h5netcdf engine writes directly to HDF5)
        if results:
            for mon_name, dataset in results.items():
                dataset.to_netcdf(
                    filepath, 
                    group=f"/results/{mon_name}", 
                    engine="h5netcdf", 
                    mode="a"
                )
        
        self.current_filepath = filepath

    def load_workspace(self, filepath: str) -> dict:
        """
        Loads the setup into the connector and returns the results dict.
        """
        loaded_results = {}
        
        with h5py.File(filepath, 'r') as f:
            # 1. Restore Setup
            if '/setup' in f:
                setup_json = f['/setup'][()].decode('utf-8')
                setup_dict = json.loads(setup_json)
                _deserialize_model(self.connector, setup_dict)
                # design_changed — not "state_changed", which
                # BeamzConnector has never actually had; this raised
                # AttributeError on every single load, right after
                # restore() (inside _deserialize_model above) had already
                # correctly rebuilt the scene, aborting before the
                # results section below ever ran and surfacing a
                # confusing "Load Error: no attribute 'state_changed'"
                # over what was otherwise a working load.
                self.connector.design_changed.emit()

            # 2. Load Results
            if '/results' in f:
                for mon_name in f['/results'].keys():
                    # Load lazily or fully into memory depending on size
                    ds = xr.open_dataset(
                        filepath, 
                        group=f"/results/{mon_name}", 
                        engine="h5netcdf"
                    ).load() # .load() pulls it into RAM so we can close the file safely
                    loaded_results[mon_name] = ds
                    
        self.current_filepath = filepath
        return loaded_results