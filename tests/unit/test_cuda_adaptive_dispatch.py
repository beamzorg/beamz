"""Adaptive execution must reach the same native controller as fixed runs."""

import numpy as np
import pytest

import beamz as bz
from beamz.simulation.cuda.engine import _runtime


def test_public_termination_policy_reaches_native_dispatch(monkeypatch):
    policy = bz.AutoTermination(chunk_steps=17)
    simulation = bz.Simulation(
        domain=(0.8e-6,) * 3, resolution=20e-9, time=np.arange(65) * 1e-17
    )
    observed = {}
    marker = object()

    def dispatch(self, **kwargs):
        assert self is simulation
        observed.update(kwargs)
        return marker, None

    monkeypatch.setattr(_runtime, "_dispatch", dispatch)
    assert simulation.run(backend="cuda", termination=policy) is marker
    assert observed["termination"] is policy
    assert observed["interop_reason"] == ""


def test_explicit_jax_termination_does_not_load_cuda(monkeypatch):
    from beamz.simulation import backend

    def unexpected_cuda_load():
        raise AssertionError("An explicit JAX request loaded CUDA")

    monkeypatch.setattr(backend, "_load_extension", unexpected_cuda_load)
    result, plan = _runtime._dispatch(
        backend="jax", termination=bz.AutoTermination(chunk_steps=17)
    )
    assert result is None
    assert plan.binding == "jax"


def diagnostic_program(encoding):
    from types import SimpleNamespace

    from beamz.lattice import component_axis_offsets_3d
    from beamz.simulation.cuda.engine._contracts import _FIELDS

    edges = {
        "x": np.asarray([0, 1, 2.3, 4, 5.1, 6.3]) * 20e-9,
        "y": np.asarray([0, 0.7, 1.8, 3.1, 4.5]) * 20e-9,
        "z": np.asarray([0, 0.8, 2, 3.7]) * 20e-9,
    }
    geometry = SimpleNamespace(
        **{a + "_edges": v for a, v in edges.items()},
        extent=tuple(np.ptp(edges[a]) for a in "xyz"),
        cell_widths=lambda a: np.diff(edges[a]),
    )
    grid = SimpleNamespace(
        geometry=geometry, material_grid=SimpleNamespace(uses_full_permittivity=False)
    )
    for index, name in enumerate(_FIELDS):
        offsets = component_axis_offsets_3d(name.capitalize())
        shape = tuple(len(edges[a]) - (offsets[a] == 0.5) for a in "zyx")
        setattr(grid, name.capitalize(), np.empty(shape, np.float32))
        value = np.float32(1.0 + index * 0.25)
        if encoding == "scalar":
            material = np.asarray(value)
        elif encoding == "broadcast":
            material = np.broadcast_to(value, shape)
        elif encoding == "rle":
            material = np.full(shape, value)
            material[1:] += np.float32(0.125)
        else:
            material = (
                value + np.arange(np.prod(shape), dtype=np.float32) * 0.0001
            ).reshape(shape)
        setattr(grid, ("eps_" if name[0] == "e" else "mu_") + name, material)
    return SimpleNamespace(grid=grid, config=SimpleNamespace(is_3d=True))


@pytest.mark.parametrize("encoding", ["scalar", "broadcast", "rle", "dense"])
def test_diagnostic_material_encoding_is_lossless_and_yee_weights_are_separable(
    encoding,
):
    import jax

    from beamz.simulation import execute
    from beamz.simulation.cuda.engine._contracts import _FIELDS
    from beamz.simulation.cuda.engine._prepare import _diagnostic_inputs

    program = diagnostic_program(encoding)
    materials, axes, volume, reserved = _diagnostic_inputs(program)
    with jax.default_device(jax.devices("cpu")[0]):
        terms, cross_terms, expected_volume, _ = execute._energy_terms(program)
    assert not cross_terms and volume == expected_volume
    term_by_field = {name: (material, measure) for name, material, measure, _ in terms}
    assert reserved >= sum(a.nbytes for m in materials for a in m.values())
    for name, spec, widths in zip(_FIELDS, materials, axes, strict=True):
        shape = getattr(program.grid, name.capitalize()).shape
        values, ends = spec["values"], spec["ends"]
        assert values.dtype == np.float32 and ends.dtype == np.int32
        if ends.size:
            assert np.all(np.diff(np.r_[0, ends]) > 0)
            assert ends[-1] == np.prod(shape)
            restored = np.repeat(values, np.diff(np.r_[0, ends])).reshape(shape)
        elif values.size == 1:
            restored = np.broadcast_to(values[0], shape)
        else:
            restored = values.reshape(shape)
        material, measure = term_by_field[name]
        expected = np.broadcast_to(np.asarray(material), shape)
        np.testing.assert_array_equal(
            np.ascontiguousarray(restored).view(np.uint32),
            np.ascontiguousarray(expected).view(np.uint32),
        )
        if encoding == "rle":
            assert ends.size and values.nbytes + ends.nbytes < expected.nbytes
        if encoding == "dense":
            assert ends.size == 0 and values.size == expected.size
        assert [len(w) for w in widths] == list(shape)
        assert all(w.dtype == np.float64 and w.ndim == 1 for w in widths)
        # Keep the reference's two roundings: form/cast the physical volume,
        # then divide in FP32. Dividing in double before casting is different.
        factored = (
            widths[0][:, None, None]
            * widths[1][None, :, None]
            * widths[2][None, None, :]
        ).astype(np.float32) / np.float32(volume)
        np.testing.assert_allclose(factored, np.asarray(measure), rtol=3e-7)


def test_diagnostic_payload_is_reserved_before_temporal_resource_queries():
    from types import SimpleNamespace

    from beamz.simulation.cuda.engine._contracts import _CPML
    from beamz.simulation.cuda.engine._plan import _choose

    inputs = [np.empty(0, np.float32) for _ in range(_CPML.input_count)]
    inputs[:6] = [
        SimpleNamespace(shape=(64, 64, 64), nbytes=64**3 * 4) for _ in range(6)
    ]
    inputs[6:12] = [np.asarray(1, np.float32) for _ in range(6)]
    module = SimpleNamespace(
        compiled_program_sha256="physics",
        moving_resource_plan=lambda *args, **kwargs: pytest.fail(
            "Queried temporal resources before rejecting the ordinary memory floor"
        ),
    )
    device = {"total_memory_bytes": 24 << 30}
    base = _choose(tuple(inputs), module, device, steps=65, requested="ordinary")
    budget = base.payload_bytes + base.reserve_bytes
    with pytest.raises(MemoryError, match="ordinary payload"):
        _choose(
            tuple(inputs),
            module,
            device,
            steps=65,
            budget=budget,
            extra_payload_bytes=1 << 20,
        )
    admitted = _choose(
        tuple(inputs),
        module,
        device,
        steps=65,
        requested="ordinary",
        budget=budget + (1 << 20),
        extra_payload_bytes=1 << 20,
    )
    assert admitted.payload_bytes == base.payload_bytes + (1 << 20)
