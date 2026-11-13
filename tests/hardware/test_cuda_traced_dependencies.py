"""Production CUDA full-state gate for the traced publication queue prototype."""

import numpy as np
import pytest

from beamz.simulation.cuda import mstep
from scripts.cuda_mstep_dependencies import tighten_first_queue
from tests.hardware.test_cuda_mstep import STATUS, _compare, _run, _seed, _simulation

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
    ),
]


@pytest.mark.parametrize(
    "material,boundary", [("binary", "all"), ("smooth", "pec"), ("lossy", "right")]
)
@pytest.mark.parametrize("steps", [4, 5])
def test_traced_first_queue_preserves_every_native_state_leaf(
    monkeypatch, material, boundary, steps
):
    import jax

    sim = _simulation(material, boundary, shape=(43, 65, 97), steps=steps)
    state = _seed(sim.compile(backend="jax", num_steps=steps))
    reference = _run(sim, state, "cuda_mstep", steps)
    # Different existing schedules have an established floating-point contract.
    # The queue optimization itself must preserve conservative mstep exactly.
    _compare(_run(sim, state, "cuda_streamed", steps), reference)
    original = mstep.build_mstep_plan

    def tightened(shapes, cpml, groups, monitor_indices=None):
        return tighten_first_queue(
            original(shapes, cpml, groups, monitor_indices), shapes, monitor_indices
        )

    monkeypatch.setattr(mstep, "build_mstep_plan", tightened)
    actual = _run(sim, state, "cuda_mstep", steps)
    for expected, observed in zip(
        jax.tree.leaves(reference), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
