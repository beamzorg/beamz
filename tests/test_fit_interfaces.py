"""Geometry, SPD assembly, conservation and CUDA tests for interface FIT."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    FITCurrentSource,
    FITInterfaceMaterial,
    FITSimulation,
    PlanarDielectricInterface,
    UniformFITMesh,
)
from beamz.simulation.fit import topology

pytestmark = [pytest.mark.component, pytest.mark.compiled]

MESHES = [
    UniformFITMesh((5, 6), 1e-6, "te"),
    UniformFITMesh((5, 6), 1e-6, "tm"),
    UniformFITMesh((3, 4, 5), 1e-6),
]


def _interface(mesh, eps=(2.0, 12.0)):
    normal = (1.0, 0.7) if mesh.dimension == 2 else (1.0, 0.7, 0.4)
    return PlanarDielectricInterface(normal, 3.15e-6, *eps)


def test_exact_planar_fractions_in_square_and_cube():
    square = UniformFITMesh((2, 2), 1.0, "te")
    plane = PlanarDielectricInterface((1, 1), 1, 2, 12)
    np.testing.assert_allclose(
        plane.material(square).fraction_minus, [[0.5, 0], [0, 0]], atol=1e-15
    )
    cube = UniformFITMesh((2, 2, 2), 1.0)
    for offset, fraction in [(1, 1 / 6), (2, 5 / 6)]:
        plane = PlanarDielectricInterface((1, 1, 1), offset, 2, 12)
        assert plane.material(cube).fraction_minus[0, 0, 0] == pytest.approx(fraction)


@pytest.mark.parametrize("mesh", MESHES)
def test_clipping_complement_and_normal_scaling(mesh):
    plane = _interface(mesh)
    fraction = plane.material(mesh).fraction_minus
    reverse = PlanarDielectricInterface(
        tuple(-n for n in plane.normal), -plane.offset, 12, 2
    )
    scaled = PlanarDielectricInterface(
        tuple(5 * n for n in plane.normal), 5 * plane.offset, 2, 12
    )
    np.testing.assert_allclose(
        fraction + reverse.material(mesh).fraction_minus, 1, atol=2e-14
    )
    np.testing.assert_allclose(
        fraction, scaled.material(mesh).fraction_minus, atol=2e-14
    )


def test_effective_tensor_obeys_normal_and_tangential_laws():
    mesh = UniformFITMesh((2, 2), 1, "te")
    normal = np.array([3, 4]) / 5
    tangent = np.array([-4, 3]) / 5
    material = FITInterfaceMaterial(np.full(mesh.shape, 0.3), normal, 2, 12)
    tensor, _ = material.tensors(mesh)
    np.testing.assert_allclose(
        np.einsum("...ij,j->...i", tensor, normal),
        np.broadcast_to(normal * (1 / (0.3 / 2 + 0.7 / 12)), tensor.shape[:-1]),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        np.einsum("...ij,j->...i", tensor, tangent),
        np.broadcast_to(tangent * (0.3 * 2 + 0.7 * 12), tensor.shape[:-1]),
        rtol=1e-14,
    )


@pytest.mark.parametrize("mesh", MESHES)
def test_assembled_operator_is_symmetric_and_positive(mesh):
    sim = FITSimulation(mesh, interface=_interface(mesh))
    rng = np.random.default_rng(8)
    u = {
        name: jnp.asarray(
            np.where(mesh.pec_mask(name), 0, rng.normal(size=shape)), sim.dtype
        )
        for name, shape in mesh.electric_shapes.items()
    }
    v = {
        name: jnp.asarray(
            np.where(mesh.pec_mask(name), 0, rng.normal(size=shape)), sim.dtype
        )
        for name, shape in mesh.electric_shapes.items()
    }
    operator = jax.jit(sim.electric_operator.apply_relative_permittivity)
    ku, kv = operator(u), operator(v)
    lhs = sum(jnp.vdot(u[k], kv[k]) for k in u)
    rhs = sum(jnp.vdot(v[k], ku[k]) for k in u)
    np.testing.assert_allclose(lhs, rhs, rtol=3e-6, atol=1e-4)
    assert float(sum(jnp.vdot(u[k], ku[k]) for k in u)) > 0
    # Also inspect every eigenvalue of the actual assembled free-DOF operator.
    # Local tensor positivity alone would not catch a nontranspose scatter.
    indices = [
        (name, index)
        for name, shape in mesh.electric_shapes.items()
        for index in zip(*np.nonzero(~mesh.pec_mask(name)))
    ]
    basis = {
        name: np.zeros((len(indices), *shape), np.float32)
        for name, shape in mesh.electric_shapes.items()
    }
    for column, (name, index) in enumerate(indices):
        basis[name][(column, *index)] = 1
    applied = jax.jit(jax.vmap(sim.electric_operator.apply_relative_permittivity))(
        {name: jnp.asarray(value) for name, value in basis.items()}
    )
    matrix = np.stack(
        [np.asarray(applied[name])[(slice(None), *index)] for name, index in indices]
    )
    np.testing.assert_allclose(matrix, matrix.T, rtol=2e-6, atol=1e-4)
    assert np.linalg.eigvalsh(matrix.astype(float))[0] > 0


@pytest.mark.parametrize("mesh", MESHES)
def test_equal_dielectrics_reproduce_diagonal_fit(mesh):
    reference = FITSimulation(mesh, permittivity=2.25, courant=0.6)
    interface = FITSimulation(
        mesh, interface=_interface(mesh, eps=(2.25, 2.25)), dt=reference.dt
    )
    rng = np.random.default_rng(0)
    fields = {
        name: rng.normal(size=value.shape) * (1 if name.startswith("E") else 1e-3)
        for name, value in reference.fields.items()
    }
    reference.set_fields(**fields)
    interface.set_fields(**fields)
    expected, actual = reference.run(12), interface.run(12)
    for name in actual.fields:
        np.testing.assert_allclose(
            actual.fields[name],
            expected.fields[name],
            rtol=8e-5,
            atol=4e-6 if name.startswith("E") else 2e-8,
        )


@pytest.mark.parametrize("mesh", MESHES)
def test_coupled_step_preserves_leapfrog_energy_and_divergence(mesh):
    sim = FITSimulation(mesh, interface=_interface(mesh), courant=0.7)
    rng = np.random.default_rng(12)
    sim.set_fields(
        **{
            name: rng.normal(size=value.shape)
            for name, value in sim.fields.items()
            if name.startswith("E")
        }
    )
    initial = float(sim.energy(conserved=True))
    charge = np.asarray(sim.electric_charge())
    result = sim.run(100)
    assert abs(float(sim.energy(conserved=True)) / initial - 1) < 5e-5
    scale = max(float(jnp.max(jnp.abs(v))) for v in sim.state.b.values())
    np.testing.assert_allclose(sim.magnetic_divergence(), 0, atol=scale * 3e-5)
    # PEC surface charge may evolve; bulk charge stays constant.
    scale = max(float(jnp.max(jnp.abs(v))) for v in sim.state.d.values())
    interior = (slice(1, -1),) * mesh.dimension
    np.testing.assert_allclose(
        np.asarray(sim.electric_charge())[interior], charge[interior], atol=scale * 3e-5
    )
    for name, mask in sim.e_masks.items():
        np.testing.assert_array_equal(np.asarray(result.fields[name])[mask], 0)


def test_coupled_charge_continuity_with_current():
    mesh = MESHES[0]
    sim = FITSimulation(mesh, interface=_interface(mesh))
    current = {name: jnp.zeros(shape) for name, shape in mesh.electric_shapes.items()}
    current["Ex"] = current["Ex"].at[2, 2].set(1e-6)
    updated = sim.advance(sim.state, electric_current=current)
    expected = -sim.dt * topology.dual_divergence(
        current, dimension=2, polarization="te"
    )
    np.testing.assert_allclose(
        sim.electric_charge(updated), expected, rtol=2e-6, atol=1e-30
    )


def test_coupled_source_continuation_and_gpu_placement():
    mesh = MESHES[0]
    source = FITCurrentSource("Ex", (2e-6, 2e-6), [1, -1, 0.5])
    single = FITSimulation(mesh, interface=_interface(mesh), sources=[source])
    chunked = FITSimulation(mesh, interface=_interface(mesh), sources=[source])
    expected = single.run(11, record_fields=["Ex"])
    chunked.run(3)
    actual = chunked.run(8, record_fields=["Ex"])
    for name in actual.fields:
        np.testing.assert_allclose(
            actual.fields[name], expected.fields[name], rtol=3e-6, atol=1e-12
        )
    assert {device.platform for device in actual.fields["Ex"].devices()} == {
        jax.default_backend()
    }
    assert single.state.d is not None


def test_interface_solve_and_step_have_no_host_callback():
    sim = FITSimulation(MESHES[0], interface=_interface(MESHES[0]))
    sim.set_fields(Ex=np.ones(sim.fields["Ex"].shape))
    residual = sim.electric_operator.apply(sim.state.d)
    np.testing.assert_allclose(
        residual["Ex"], sim.fields["Ex"] * sim.mesh.spacing, rtol=2e-6
    )
    jaxpr = str(jax.make_jaxpr(sim.advance)(sim.state))
    assert "callback" not in jaxpr
    assert "custom_linear_solve" in jaxpr


@pytest.mark.parametrize(
    "interface",
    [
        PlanarDielectricInterface((0, 0), 1, 2, 12),
        PlanarDielectricInterface((1, 0), np.nan, 2, 12),
        PlanarDielectricInterface((1, 0), 1, -1, 12),
        FITInterfaceMaterial(np.full((5, 6), 1.1), (1, 0), 2, 12),
    ],
)
def test_invalid_interfaces_are_rejected(interface):
    with pytest.raises(ValueError):
        FITSimulation(MESHES[0], interface=interface)


def test_coupled_loss_and_ambiguous_materials_are_rejected():
    for kwargs in [{"conductivity": 1}, {"permittivity": 2}]:
        with pytest.raises(ValueError):
            FITSimulation(MESHES[0], interface=_interface(MESHES[0]), **kwargs)


def test_failed_constitutive_solve_is_reported():
    sim = FITSimulation(
        MESHES[0], interface=_interface(MESHES[0]), interface_solve_maxiter=1
    )
    rng = np.random.default_rng(42)
    sim.set_fields(
        **{
            name: rng.normal(size=value.shape)
            for name, value in sim.fields.items()
            if name.startswith("E")
        }
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        sim.run(2)


@pytest.mark.parametrize("angle,offset", [(0, 0.47), (30, 0.69)])
def test_tensor_interface_improves_cavity_modes_over_scalar_fraction(angle, offset):
    from scripts.benchmark_fit_interfaces import fem_reference, fit_modes

    plane = PlanarDielectricInterface(
        (np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))), offset, 2, 12
    )
    enable_x64 = (
        jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
    )
    with enable_x64():
        reference = fem_reference(plane, 64)
        scalar, _ = fit_modes(plane, 12, "scalar_fraction")
        tensor, _ = fit_modes(plane, 12, "tensor")
    scalar_error = np.mean(abs(scalar / reference - 1))
    tensor_error = np.mean(abs(tensor / reference - 1))
    assert tensor_error < 0.7 * scalar_error
