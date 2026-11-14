"""The package split preserves prior specification and checkpoint identities."""

import inspect
import pickle

import beamz.optimization.autodiff as legacy_autodiff
import numpy as np
from beamz.optimization.autodiff import transform_density as legacy_transform
from beamz.optimization.data import bind_modal_data as legacy_bind
from beamz.optimization.region import FilterProject as LegacyFilterProject
from beamz.optimization.result import load_result as legacy_load_result

from beamz._cache_tokens import cache_token
from beamz.optimization import (
    FilterProject,
    LinearSchedule,
    ModePower,
    UniformInitializationSpec,
)
from beamz.optimization.objectives.projection import bind_modal_data
from beamz.optimization.optimizer import _optimize as legacy_optimize
from beamz.optimization.optimizer.runtime.identity import (
    _function_token,
    _legacy_checkpoint_token,
)
from beamz.optimization.optimizer.runtime.load import load_result
from beamz.optimization.optimizer.runtime.loop import _optimize
from beamz.optimization.topology.transforms.density import transform_density


def test_legacy_specifications_preserve_identity_and_pickle_round_trip():
    values = (
        FilterProject(radius=1.0),
        LinearSchedule(1.0, 2.0, 3),
        ModePower("out"),
        UniformInitializationSpec(),
    )
    legacy_modules = (
        "beamz.optimization.region",
        "beamz.optimization.schedules",
        "beamz.optimization.objectives",
        "beamz.optimization.region",
    )
    assert LegacyFilterProject is FilterProject
    for value, legacy_module in zip(values, legacy_modules, strict=True):
        cls = type(value)
        encoded = pickle.dumps(value, protocol=0)
        legacy = encoded.replace(cls.__module__.encode(), legacy_module.encode())
        assert legacy != encoded
        assert cache_token(pickle.loads(legacy)) == cache_token(value)
        token = _legacy_checkpoint_token(cache_token(value))
        if token[0] == "callable":
            assert token[1] == legacy_module
        else:
            assert token[0] == f"{legacy_module}.{cls.__name__}"
        assert inspect.getsource(cls)


def test_legacy_density_transform_uses_the_new_implementation():
    assert legacy_transform is transform_density
    values = np.full((2, 2), 0.5, dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    np.testing.assert_allclose(
        legacy_transform(values, mask, 1.0, 0.5, 0),
        transform_density(values, mask, 1.0, 0.5, 0),
    )


def test_legacy_module_helpers_resolve_to_moved_implementations():
    assert legacy_bind is bind_modal_data
    assert legacy_load_result is load_result
    assert legacy_optimize is _optimize


def test_legacy_module_keeps_its_identity_for_callback_fingerprints():
    def objective(value):
        return legacy_autodiff.smooth_max(value)

    assert legacy_autodiff.__name__ == "beamz.optimization.autodiff"
    assert "beamz.optimization.autodiff" in repr(_function_token(objective))
