"""Physical and algebraic acceptance tests for the initial FIT backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz import PEC, Design, FITCurrentSource, FITSimulation, UniformFITMesh
from beamz.const import EPS_0, MU_0
from beamz.simulation.boundaries import (
    create_metallic_boundary_masks,
    initialize_full_pec_3d_state,
)
from beamz.simulation.fields import Fields
from beamz.simulation.fit import topology

pytestmark = [pytest.mark.component, pytest.mark.compiled]

MESHES = [
    UniformFITMesh((7, 9), 1e-6, "tm"),
    UniformFITMesh((7, 9), 1e-6, "te"),
    UniformFITMesh((5, 7, 9), 1e-6),
]


def _seed(sim, seed=5):
    rng = np.random.default_rng(seed)
    values = {
        name: rng.standard_normal(value.shape) * (1 if name.startswith("E") else 1e-3)
        for name, value in sim.fields.items()
    }
    sim.set_fields(**values)


@pytest.mark.parametrize("mesh", MESHES)
def test_curl_has_exact_transpose_and_divergence_identity(mesh):
    rng = np.random.default_rng(1)
    e = {
        k: jnp.asarray(rng.normal(size=s), jnp.float32)
        for k, s in mesh.electric_shapes.items()
    }
    h = {
        k: jnp.asarray(rng.normal(size=s), jnp.float32)
        for k, s in mesh.magnetic_shapes.items()
    }
    args = dict(dimension=mesh.dimension, polarization=mesh.polarization)
    ce, cth = topology.curl(e, **args), topology.curl_transpose(h, **args)
    lhs = sum(jnp.vdot(ce[k], h[k]) for k in h)
    rhs = sum(jnp.vdot(e[k], cth[k]) for k in e)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(topology.divergence(ce, **args), 0, atol=1e-6)
    np.testing.assert_allclose(topology.dual_divergence(cth, **args), 0, atol=1e-6)
    potential = jnp.asarray(
        rng.normal(size=tuple(n + 1 for n in mesh.shape)), jnp.float32
    )
    cg = topology.curl(topology.gradient(potential, **args), **args)
    for value in cg.values():
        np.testing.assert_allclose(value, 0, atol=1e-6)


@pytest.mark.parametrize("mesh", MESHES)
def test_uniform_fit_matches_existing_yee_fields(mesh):
    # Current FDTD magnetic update coefficients use MU_0 without relative mu.
    # Compare the supported nonmagnetic case; FIT mu is tested analytically.
    sim = FITSimulation(mesh, permittivity=2.25, courant=0.6)
    _seed(sim)
    # TE compact storage has one fewer face/cell than its material raster.
    reference_shape = (
        tuple(n + 1 for n in mesh.shape)
        if mesh.dimension == 2 and mesh.polarization == "te"
        else mesh.shape
    )
    reference = Fields(
        np.full(reference_shape, 2.25),
        np.zeros(reference_shape),
        np.ones(reference_shape),
        mesh.spacing,
    )
    reference.boundaries = [PEC(edges="all")]
    if mesh.dimension == 3:
        reference.full_pec_3d_state = initialize_full_pec_3d_state(reference)
        target = reference.full_pec_3d_state
    else:
        reference.set_metallic_masks(
            create_metallic_boundary_masks(
                reference, ["left", "right", "top", "bottom"], is_3d=False
            )
        )
        target = reference
    for name, value in sim.fields.items():
        setattr(target, name, value)
    for _ in range(12):
        reference.update_h(sim.dt)
        reference.update_e(sim.dt)
        sim.step()
        for name, value in sim.fields.items():
            np.testing.assert_allclose(
                value,
                getattr(target, name),
                rtol=4e-5,
                atol=3e-6 if name.startswith("E") else 2e-8,
            )


@pytest.mark.parametrize("mesh", MESHES)
def test_source_free_conserved_energy_and_magnetic_divergence(mesh):
    sim = FITSimulation(mesh, courant=0.75)
    _seed(sim)
    initial_energy = float(sim.energy(conserved=True))
    initial_divergence = np.asarray(sim.magnetic_divergence())
    result = sim.run(160)
    assert result.final_step == 160
    assert result.history == {}
    assert initial_energy > 0
    assert abs(float(sim.energy(conserved=True)) / initial_energy - 1) < 2e-5
    scale = max(float(jnp.max(jnp.abs(v))) for v in sim.state.b.values())
    np.testing.assert_allclose(
        sim.magnetic_divergence(), initial_divergence, atol=scale * 2e-5
    )
    for name, mask in sim.e_masks.items():
        np.testing.assert_array_equal(np.asarray(sim.fields[name])[mask], 0)


def test_charge_continuity_in_interior_with_impressed_current():
    mesh = UniformFITMesh((6, 8), 1e-6, "te")
    sim = FITSimulation(mesh)
    current = {k: jnp.zeros(shape) for k, shape in mesh.electric_shapes.items()}
    current["Ex"] = current["Ex"].at[3, 4].set(1e-6)
    next_state = sim.advance(sim.state, electric_current=current)
    expected = -sim.dt * topology.dual_divergence(
        current, dimension=2, polarization="te"
    )
    np.testing.assert_allclose(
        sim.electric_charge(next_state), expected, rtol=2e-6, atol=1e-30
    )


def test_point_current_units_and_source_chunking():
    mesh = UniformFITMesh((8, 10), 1e-6)
    source = FITCurrentSource("Ez", (5e-6, 4e-6), [2.0, -1.0, 0.5])
    sim = FITSimulation(mesh, sources=[source])
    initial = sim.step()
    assert float(sim.fields["Ez"][4, 5]) == pytest.approx(-2 * sim.dt / EPS_0, rel=2e-6)
    assert float(jnp.max(jnp.abs(initial.b["Bx"]))) == 0
    continuous = FITSimulation(mesh, sources=[source]).run(
        11, record_fields=["Ez", "Hx"]
    )
    chunked_sim = FITSimulation(mesh, sources=[source])
    chunked_sim.run(3)
    chunked = chunked_sim.run(8, record_fields=["Ez", "Hx"])
    for name in continuous.fields:
        np.testing.assert_allclose(
            chunked.fields[name], continuous.fields[name], rtol=2e-6, atol=1e-12
        )
    np.testing.assert_allclose(
        chunked.history["Ez"], continuous.history["Ez"][3:], rtol=2e-6, atol=1e-12
    )
    ds = chunked.to_xarray()
    np.testing.assert_allclose(ds.t_e, chunked.times)
    np.testing.assert_allclose(ds.t_h, chunked.times - 0.5 * sim.dt)
    assert ds.Ez.attrs["units"] == "V/m"
    assert ds.Hx.attrs["units"] == "A/m"
    assert ds.x_Ez[-1] == pytest.approx(10e-6)


@pytest.mark.parametrize("permeability", [1.0, 1.3])
def test_tm_cavity_matches_discrete_and_continuum_frequency(permeability):
    mesh = UniformFITMesh((20, 24), 1e-6)
    sim = FITSimulation(mesh, permittivity=2.25, permeability=permeability, courant=0.7)
    y, x = mesh.coordinates("Ez")
    sim.set_fields(
        Ez=np.sin(np.pi * y[:, None] / (mesh.shape[0] * mesh.spacing))
        * np.sin(np.pi * x[None, :] / (mesh.shape[1] * mesh.spacing))
    )
    trace = np.asarray(sim.run(700, record_fields=["Ez"]).history["Ez"])[:, 10, 12]
    # A single discrete eigenmode follows this exact leapfrog recurrence.
    c = 1 / np.sqrt(EPS_0 * MU_0 * 2.25 * permeability)
    omega_space = (
        2
        * c
        / mesh.spacing
        * np.sqrt(sum(np.sin(np.pi / (2 * n)) ** 2 for n in mesh.shape))
    )
    omega_dt = 2 * np.arcsin(0.5 * sim.dt * omega_space)
    residual = trace[2:] - 2 * np.cos(omega_dt) * trace[1:-1] + trace[:-2]
    # CUDA fusion changes FP32 rounding; budget a few dozen ULPs for the
    # recurrence after hundreds of steps, rather than using a CPU-only bound.
    assert np.max(abs(residual)) < 32 * np.finfo(trace.dtype).eps * max(
        1, np.max(abs(trace))
    )
    omega_continuum = (
        np.pi * c * np.sqrt(sum(1 / (n * mesh.spacing) ** 2 for n in mesh.shape))
    )
    assert abs(omega_dt / sim.dt / omega_continuum - 1) < 0.002


def test_ohmic_loss_dissipates_energy():
    mesh = UniformFITMesh((12, 14), 1e-6)
    sim = FITSimulation(mesh, conductivity=1000.0)
    _seed(sim)
    initial = float(sim.energy(conserved=True))
    for _ in range(8):
        sim.run(20)
        energy = float(sim.energy(conserved=True))
        assert energy <= initial * (1 + 1e-5)
        initial = energy


def test_from_design_preserves_uniform_materials():
    from beamz import Material

    design = Design(width=10e-6, height=8e-6, material=Material(permittivity=2.25))
    sim = FITSimulation.from_design(design, resolution=1e-6)
    assert sim.mesh.shape == (8, 10)
    np.testing.assert_allclose(
        sim.m_epsilon["Ez"][1:-1, 1:-1], EPS_0 * 2.25e-12, rtol=2e-6
    )
    with pytest.raises(ValueError, match="integer multiples"):
        FITSimulation.from_design(design, resolution=1.1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"permittivity": 0},
        {"permeability": -1},
        {"conductivity": -1},
        {"courant": 1.1},
        {"dt": np.nan},
        {"dt": 1.0},
        {"permittivity": np.ones((2, 2))},
    ],
)
def test_invalid_materials_and_timesteps_are_rejected(kwargs):
    with pytest.raises(ValueError):
        FITSimulation(MESHES[0], **kwargs)


@pytest.mark.parametrize(
    "shape,spacing",
    [((1, 4), 1), ((4.5, 4), 1), ((4,), 1), ((4, 4), 0), ((4, 4), np.nan)],
)
def test_invalid_mesh_is_rejected(shape, spacing):
    with pytest.raises(ValueError):
        UniformFITMesh(shape, spacing)


def test_invalid_sources_and_recording_are_rejected():
    for source in [
        FITCurrentSource("Ex", (2e-6, 2e-6), [1]),
        FITCurrentSource("Ez", (0, 0), [1]),
        FITCurrentSource("Ez", (-1, 1), [1]),
        FITCurrentSource("Ez", (2e-6, 2e-6), []),
    ]:
        with pytest.raises(ValueError):
            FITSimulation(MESHES[0], sources=[source])
    sim = FITSimulation(MESHES[0])
    for fields in [["Ez", "Ez"], ["Ex"]]:
        with pytest.raises(ValueError):
            sim.run(1, record_fields=fields)
    with pytest.raises(ValueError):
        sim.run(-1)
    assert sim.run(0).final_step == 0


def test_jitted_advance_differentiates_initial_fields():
    sim = FITSimulation(MESHES[0])
    state = sim.state

    def objective(amplitude):
        e = dict(state.e)
        e["Ez"] = e["Ez"].at[3, 4].set(amplitude)
        updated = sim.advance(state._replace(e=e))
        return jnp.sum(updated.e["Ez"] ** 2)

    derivative = float(jax.jit(jax.grad(objective))(1.0))
    difference = float((objective(1.001) - objective(0.999)) / 0.002)
    assert derivative == pytest.approx(difference, rel=2e-4)
