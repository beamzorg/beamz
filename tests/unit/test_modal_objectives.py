"""Composition semantics and immutable objective specifications."""

import numpy as np
import pytest

from beamz.optimization import ModePower, WeightedObjective


def test_objective_arithmetic_keeps_signs_and_spectral_identity():
    transmission = ModePower("output", frequencies=[2e14, 2.1e14])
    reflection = ModePower("input", direction="-")
    objective = 0.5 * (transmission + transmission) - 0.1 * reflection
    assert objective.terms == (transmission, transmission, reflection)
    assert objective.weights == (0.5, 0.5, -0.1)
    assert sum([transmission, reflection]).weights == (1.0, 1.0)
    assert hash(transmission)
    assert transmission.frequencies == (2e14, 2.1e14)


@pytest.mark.parametrize(
    "frequencies", [[], [0], [np.nan], [np.inf], [1e14, 1e14], [[1e14]], 1e14]
)
def test_rejects_invalid_frequency_selection(frequencies):
    with pytest.raises(ValueError, match="frequencies"):
        ModePower("output", frequencies=frequencies)


def test_rejects_invalid_combinations():
    term = ModePower("output")
    with pytest.raises(ValueError, match="finite"):
        term * np.nan
    with pytest.raises(ValueError, match="match"):
        WeightedObjective([term], [])
    with pytest.raises(ValueError, match="nonempty"):
        WeightedObjective([], [])
    with pytest.raises(TypeError):
        term + 1
    with pytest.raises(ValueError, match="reference_mode_index"):
        ModePower("output", reference_mode_index=-1)
