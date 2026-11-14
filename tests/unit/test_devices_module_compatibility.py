"""Compatibility and structural checks for the devices package layout."""

import importlib
import inspect
import pickle
from pathlib import Path

from beamz.devices.modes.spatial import discrete, plane
from beamz.devices.sources.launch import mode_launch
from beamz.optimization.optimizer.runtime.identity import _legacy_checkpoint_token


def test_legacy_modules_preserve_symbols_and_monkeypatches(monkeypatch):
    legacy = importlib.import_module("beamz.devices.modes.plane")
    assert legacy.__name__ == "beamz.devices.modes.plane"
    assert legacy.solve_modes is plane.solve_modes
    assert inspect.getsource(legacy) == inspect.getsource(plane)
    replacement = object()
    monkeypatch.setattr(legacy, "solve_modes", replacement)
    assert plane.solve_modes is replacement


def test_old_pickle_module_paths_resolve_to_moved_types():
    assert (
        pickle.loads(b"cbeamz.devices.modes.discrete\nModePlaneSpec\n.")
        is discrete.ModePlaneSpec
    )
    assert (
        pickle.loads(b"cbeamz.devices.sources.mode_launch\nModePlanEntry\n.")
        is mode_launch.ModePlanEntry
    )


def test_moved_modules_keep_checkpoint_tokens():
    assert _legacy_checkpoint_token(
        ("callable", "beamz.devices.modes.spatial.plane", "solve_modes")
    ) == ("callable", "beamz.devices.modes.plane", "solve_modes")
    assert _legacy_checkpoint_token(
        ("beamz.devices.modes.spatial.discrete", "ModePlaneSpec")
    ) == ("beamz.devices.modes.discrete", "ModePlaneSpec")


def test_devices_directory_limits():
    root = Path(__file__).resolve().parents[2] / "beamz" / "devices"
    source = [path for path in root.rglob("*.py") if path.name != "__init__.py"]
    assert len(source) in (35, 36)
    directories = {root, *(path.parent for path in source)}
    for directory in directories:
        files = sum(path.parent == directory for path in source)
        children = {path for path in directories if path.parent == directory}
        assert len(children) <= 5
        assert files + len(children) <= 6
