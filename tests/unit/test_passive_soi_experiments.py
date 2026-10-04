"""Guard the experiment controls and independent reference interpretation."""

import numpy as np
import pytest

from beamz import LIGHT_SPEED
from tests.differential.passive_soi.common import load_passive_soi_case
from tests.differential.passive_soi.experiments import (
    ExperimentOptions,
    FieldDecayFailure,
    check_known_failure,
    power_comparison,
)


def test_coarse_grid_is_not_evidence_of_converged_reference_agreement():
    case = load_passive_soi_case("polarization_splitter_rotator")
    result = power_comparison(case, "conversion", 6, 0.827823)
    assert result["same_ppw_agreement"] is False
    assert result["reference_agreement"] is None
    assert result["reference_eligible"] is False
    refined = power_comparison(case, "conversion", 20, 0.949)
    assert refined["reference_agreement"] is True
    assert refined["mesh_convergence_established"] is False
    assert power_comparison(case, "conversion", 20, 1.1)["reference_agreement"] is False


def test_exact_center_is_added_without_removing_reference_frequencies():
    case = load_passive_soi_case("mmi2x2")
    original = ExperimentOptions().frequencies(case, 20)
    exact = ExperimentOptions(exact_center=True).frequencies(case, 20)
    assert np.all(np.isin(original, exact))
    assert LIGHT_SPEED / 1.55e-6 in exact


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_time_ps": -1},
        {"boundary_thickness_um": float("nan")},
        {"monitor_offset_um": -1},
        {"source_profiles": 1.5},
        {"port_extension_policy": "invalid"},
    ],
)
def test_invalid_experiment_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ExperimentOptions(**kwargs)


def test_known_failure_wrapper_does_not_mask_runtime_errors():
    def numerical_failure():
        raise AssertionError("field decay too large")

    def infrastructure_failure():
        raise RuntimeError("allocation failed")

    with pytest.raises(FieldDecayFailure):
        check_known_failure(numerical_failure, FieldDecayFailure)
    with pytest.raises(RuntimeError):
        check_known_failure(infrastructure_failure, FieldDecayFailure)
