"""Acquisition contracts agree across host decoding, AD, and local ownership."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.devices.monitors.compiler import AcquisitionLayout, SamplingPlan


@pytest.mark.parametrize("complex_fields,scale", [(False, 2), (True, 1)])
def test_layout_normalizes_ragged_arenas_and_aligns_only_h(complex_fields, scale):
    layout = AcquisitionLayout(
        name="phasor",
        monitor_index=0,
        record_interval=1,
        power_scale=1,
        freq_count=2,
        freq_hz=jnp.array([0.25, 0.5]),
        dft_point_count=3,
        dft_value_offset=5,
        dft_weight_offset=2,
    )
    real = jnp.arange(45, dtype=jnp.float32)
    state = SimpleNamespace(
        ex=jnp.zeros((), dtype=jnp.complex64 if complex_fields else jnp.float32),
        dft_vec_re=real,
        dft_vec_im=-real,
        dft_weight_sum=jnp.array([99, 99, 2, 4, 99], dtype=jnp.float32),
    )
    expected = np.array(real[5:41] * (1 - 1j)).reshape(6, 2, 3)
    expected *= scale / np.array([2, 4])[None, :, None]
    np.testing.assert_allclose(layout.normalized_dft(state), expected)
    expected[3:] *= np.exp(-1j * np.pi * np.array([0.25, 0.5]) * 0.1)[None, :, None]
    np.testing.assert_allclose(
        layout.normalized_dft(state, align_h=True, dt=0.1), expected, rtol=1e-6
    )
    # Objective gradients may use only this monitor's arena, never its neighbors.
    gradient = jax.grad(lambda re: jnp.real(jnp.sum(layout.unpack_dft(re, -real))))(
        real
    )
    np.testing.assert_array_equal(
        gradient, np.r_[np.zeros(5), np.ones(36), np.zeros(4)]
    )


def test_local_sampling_ownership_sums_to_global_for_each_component_support():
    # The two components intentionally have different local Yee support lengths.
    fields = (jnp.arange(8.0).reshape(4, 2), jnp.arange(12.0).reshape(6, 2))
    indices = (np.array([[2, 4], [1, 6]]), np.array([[4, 6], [1, 10]]))
    plan = SamplingPlan(indices, (jnp.array([[0.25, 0.75], [0.5, 0.5]]),) * 2)
    expected = plan.sample(fields)
    contributions = []
    for rank in range(2):
        local = tuple(jnp.split(array, 2)[rank] for array in fields)
        contributions.append(
            plan.sample(
                local,
                local_axis=0,
                shard_index=rank,
                shapes=tuple(f.shape for f in fields),
            )
        )
    np.testing.assert_allclose(sum(contributions), expected)


@pytest.mark.contract
@pytest.mark.parametrize("monitor_type", ["FieldMonitor", "FluxMonitor", "ModeMonitor"])
def test_compiler_preserves_validated_monitor_frequency_snapshot_and_interval(
    monitor_type,
):
    import beamz as bz
    from beamz.devices.monitors.compiler import compile_monitor_specs

    frequencies = np.array([1.0, 2.0])
    monitor = getattr(bz, monitor_type)(
        center=(2.0, 2.0, 0.0),
        size=(0.0, 2.0, 1.0),
        freqs=frequencies,
        interval=np.int64(3),
    )
    frequencies[:] = np.nan
    fields = SimpleNamespace(
        permittivity=np.ones((4, 4)),
        **{c: np.zeros((4, 4)) for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")},
    )
    (layout,), max_records = compile_monitor_specs((monitor,), fields, 1.0, 10, 0.1)
    np.testing.assert_array_equal(layout.freq_hz, [1.0, 2.0])
    assert layout.freq_count == 2
    assert layout.record_interval == layout.dft_record_interval == 3
    assert max_records == 4
