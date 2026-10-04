"""Physical-state parity for opt-in cyclic CUDA storage layouts."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.execute import build_scan, initial_program_state
from scripts.benchmark_cuda_realistic import build_simulation

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


@pytest.mark.parametrize("axes", ["120", "201"])
@pytest.mark.parametrize(
    "material,workload",
    [
        ("binary", "mode"),
        ("smooth", "field"),
        ("binary", "sources"),
        ("smooth", "plain"),
    ],
)
@pytest.mark.parametrize("schedule", ["ordinary", "pair", "cpml_pair"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cyclic_graph(axes, material, workload, schedule, precision, monkeypatch):
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", "012")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", precision)
    monkeypatch.setenv(
        "BEAMZ_CUDA_TEMPORAL_STEPS", "1" if schedule == "ordinary" else "2"
    )
    monkeypatch.setenv(
        "BEAMZ_CUDA_CPML_TEMPORAL", "1" if schedule == "cpml_pair" else "0"
    )
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PAIR_TILE", "oriented")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_SPATIAL", "0")
    monkeypatch.setenv("BEAMZ_CUDA_FIELD_PADDING", "none")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_CORE_FUSION", "0")
    sim = build_simulation(
        SimpleNamespace(
            shape=(37, 49, 65),
            steps=35,
            pml=12,
            monitors=2,
            frequencies=3,
            material=material,
            source="mode",
            monitor_type="mode" if workload == "mode" else "field",
        )
    )
    if workload in ("sources", "plain"):
        sim = sim.updated_copy(monitors=[])
    if workload == "field":
        first = sim.monitors[0]
        sim = sim.updated_copy(
            monitors=[
                first.updated_copy(
                    size=(0, first.size[1] / 2, first.size[2] / 2),
                    freqs=first.freqs[:1],
                    fields=("Ex", "Hz"),
                ),
                sim.monitors[1],
            ]
        )
    if workload == "plain":
        sim = sim.updated_copy(sources=[])
    program = sim.compile(num_steps=33, backend="cuda_streamed")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=35)
    rng = np.random.default_rng(20260918)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        },
        **{
            name: tuple(
                (rng.normal(size=x.shape).astype(np.float32) * 1e2).astype(x.dtype)
                for x in getattr(state, name)
            )
            for name in ("cpml_psi_h_terms", "cpml_psi_e_terms")
        },
    )
    state = state._replace(
        dft_vec_re=rng.normal(size=state.dft_vec_re.shape).astype(np.float32),
        dft_vec_im=rng.normal(size=state.dft_vec_im.shape).astype(np.float32),
    )
    ref_exec = build_scan(program).lower(state, program.coefficients).compile()
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", axes)
    rotated_program = sim.compile(num_steps=33, backend="cuda_streamed")
    assert rotated_program is not program
    assert rotated_program.config.cuda_storage_axes == tuple(map(int, axes))
    rotated_exec = (
        build_scan(rotated_program).lower(state, rotated_program.coefficients).compile()
    )
    ref = ref_exec(state, program.coefficients)
    actual = rotated_exec(state, rotated_program.coefficients)
    for i, (expected, got) in enumerate(
        zip(jax.tree.leaves(ref), jax.tree.leaves(actual), strict=True)
    ):
        np.testing.assert_array_equal(got, expected, err_msg=f"leaf {i}")
    # Continue with a different step count from the canonical result layout.
    from dataclasses import replace

    tail = replace(program, config=replace(program.config, num_steps=2))
    ref_tail = build_scan(tail).lower(ref, program.coefficients).compile()
    rotated_tail_program = replace(
        rotated_program, config=replace(rotated_program.config, num_steps=2)
    )
    rotated_tail = (
        build_scan(rotated_tail_program)
        .lower(actual, rotated_program.coefficients)
        .compile()
    )
    ref = ref_tail(ref, program.coefficients)
    actual = rotated_tail(actual, rotated_program.coefficients)
    for i, (expected, got) in enumerate(
        zip(jax.tree.leaves(ref), jax.tree.leaves(actual), strict=True)
    ):
        np.testing.assert_array_equal(got, expected, err_msg=f"continued leaf {i}")


def test_cyclic_public_single_step(monkeypatch):
    """Public compilation selects the layout and handles a one-step graph."""
    from beamz.simulation.cuda import storage

    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", "012")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "fp32")
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "2")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_TEMPORAL", "1")
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PAIR_TILE", "oriented")
    sim = build_simulation(
        SimpleNamespace(
            shape=(37, 49, 65),
            steps=5,
            pml=12,
            monitors=2,
            frequencies=3,
            material="binary",
            source="mode",
            monitor_type="mode",
        )
    )
    program = sim.compile(backend="cuda_streamed")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=5)
    rng = np.random.default_rng(20260918)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    expected = sim.advance(state=state, num_steps=1, backend="cuda_streamed").state
    calls = []
    original = storage.wrap_native_calls

    def record(*args):
        calls.append(args[-1])
        return original(*args)

    monkeypatch.setattr(storage, "wrap_native_calls", record)
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", "120")
    actual = sim.advance(state=state, num_steps=1, backend="cuda_streamed").state
    assert calls == [(1, 2, 0)]
    for ref, got in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(got, ref)
