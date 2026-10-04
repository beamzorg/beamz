"""Guard the material mismatch in the radiating S-bend from issue #309."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from tests.characterization.sbend_monitor_case import Probe


@pytest.mark.hardware
@pytest.mark.slow
@pytest.mark.characterization
def test_radiating_sbend_plane_and_aperture_spread():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("The 3D S-bend convergence probe requires a GPU")
    scene = json.loads(
        (Path(__file__).with_name("fixtures") / "issue_309_sbend.json").read_text()
    )
    distances = (0.25, 0.5, 0.75, 1.0)
    # Retain the reproduction's boundary warning rather than hiding it. These
    # bounds characterize extraction at mesh 10; they do not certify convergence.
    with pytest.warns(RuntimeWarning, match="PML material varies"):
        result = Probe(scene, 10, widths=(2.7, 4.0), distances=distances).run()
    assert result["termination"]["converged"]
    assert all(result["valid_mask"])
    center = int(np.argmin(np.abs(np.asarray(result["wavelength_um"]) - 1.55)))
    narrow, wide = (
        np.asarray([result["s21_db"][f"w{width:g}_d{d:g}"][center] for d in distances])
        for width in (2.7, 4.0)
    )
    assert np.all(np.isfinite(narrow)) and np.all(np.isfinite(wide))
    # Scalar-only projection gives 0.315 dB / 0.267 dB respectively.
    assert np.ptp(narrow) < 0.12
    assert np.ptp(wide) < 0.08
    assert np.max(np.abs(narrow - wide)) < 0.04
