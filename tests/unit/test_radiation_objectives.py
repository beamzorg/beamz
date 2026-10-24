"""Radiation objective semantics, coherent phase, validation, and JAX gradients."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.analysis import (
    FarFieldProjection,
    FarFieldSurface,
    near_field_box,
    near_to_far,
)
from beamz.analysis.far_field import SurfaceFieldData
from beamz.const import LIGHT_SPEED
from beamz.optimization import (
    CollectionConePower,
    DirectionalPower,
    ModePower,
    TargetBeamOverlap,
)
from tests.far_field_case import box_quadrature, dipole_fields, sphere_quadrature


def radiation_data():
    points, normals, weights = box_quadrature(count=8)
    e, h = dipole_fields(points)
    directions, angular_weights = sphere_quadrature(4, 8)
    data, surfaces = {}, []
    for i in range(6):
        name = f"s{i}"
        sl = slice(i * 64, (i + 1) * 64)
        data[name] = SurfaceFieldData(
            e[None, sl],
            h[None, sl],
            points[sl],
            weights[sl],
            (LIGHT_SPEED,),
            i // 2,
            tuple("E" + a for a in "xyz") + tuple("H" + a for a in "xyz"),
        )
        surfaces.append(FarFieldSurface(name, -1 if i % 2 == 0 else 1))
    projection = FarFieldProjection(surfaces, directions, angular_weights)
    return data, projection


def test_collection_normalization_and_modal_composition():
    data, projection = radiation_data()
    cone = CollectionConePower(projection)
    far = projection(data)
    mask = np.asarray(projection.directions)[:, 2] >= 0
    np.testing.assert_allclose(
        cone(data), far.power(np.asarray(projection.solid_angle_weights) * mask).mean()
    )
    doubled_power = {
        n: replace(
            d,
            electric=d.electric * np.sqrt(2),
            magnetic=d.magnetic * np.sqrt(2),
            source_power=2,
        )
        for n, d in data.items()
    }
    np.testing.assert_allclose(cone(doubled_power), cone(data), rtol=2e-6)
    objective = cone - 0.2 * ModePower("output")
    assert objective.weights == (1.0, -0.2)
    assert len(objective.frequency_requests) == 7
    assert hash(cone)


def test_coherent_overlap_rejects_wrong_phase_and_polarization():
    data, projection = radiation_data()
    far = projection(data)
    target = np.asarray(far.electric[0])
    matched = TargetBeamOverlap(projection, target)
    np.testing.assert_allclose(
        matched(data), far.power(projection.solid_angle_weights)[0], rtol=2e-6
    )
    scaled = TargetBeamOverlap(projection, target * (2 + 3j))
    np.testing.assert_allclose(scaled(data), matched(data), rtol=2e-6)
    orthogonal = np.cross(np.asarray(projection.directions), target.conj())
    assert TargetBeamOverlap(projection, orthogonal)(data) < matched(data) * 1e-10
    scrambled = target * np.exp(1j * np.arange(len(target)))[:, None]
    assert TargetBeamOverlap(projection, scrambled)(data) < matched(data) * 0.05


@pytest.mark.parametrize("kind", ["direction", "cone", "overlap"])
def test_complex_surface_gradients_match_finite_differences(kind):
    data, projection = radiation_data()
    target = np.asarray(projection(data).electric[0])
    objective = {
        "direction": DirectionalPower(projection),
        "cone": CollectionConePower(projection, 0.8),
        "overlap": TargetBeamOverlap(projection, target),
    }[kind]
    rng = np.random.default_rng(4)
    # Perturb both real and imaginary E/H, including coherence between surfaces.
    variation = {
        n: (
            rng.normal(size=d.electric.shape) + 1j * rng.normal(size=d.electric.shape),
            rng.normal(size=d.magnetic.shape) + 1j * rng.normal(size=d.magnetic.shape),
        )
        for n, d in data.items()
    }

    def fun(x):
        return objective(
            {
                n: replace(
                    d,
                    electric=jnp.asarray(d.electric) * (1 + x * variation[n][0]),
                    magnetic=jnp.asarray(d.magnetic) * (1 + x * variation[n][1]),
                )
                for n, d in data.items()
            }
        )

    gradient = jax.jit(jax.grad(fun))(0.1)
    fd = (fun(0.102) - fun(0.098)) / 0.004
    assert abs(gradient) > 1e-7
    np.testing.assert_allclose(gradient, fd, rtol=1e-3, atol=1e-7)


def test_box_records_tangential_fields_with_outward_normals():
    monitors, surfaces = near_field_box(
        center=(1, 2, 3), size=(2, 4, 6), freqs=[LIGHT_SPEED]
    )
    assert len(monitors) == len(surfaces) == 6
    for m, s in zip(monitors, surfaces, strict=True):
        axis = "xyz".index(m.plane_normal)
        assert s.monitor == m.name
        assert len(m.fields) == 4
        assert all(c[-1] != m.plane_normal for c in m.fields)
        assert m.center[axis] == (1, 2, 3)[axis] + s.normal_sign * (2, 4, 6)[axis] / 2


def test_invalid_projection_configuration_and_missing_fields():
    data, projection = radiation_data()
    for kwargs in [
        dict(directions=[(0, 0, 2)]),
        dict(refractive_index=-1),
        dict(refractive_index=1 + 1j),
        dict(solid_angle_weights=[-1]),
        dict(frequencies=[0]),
        dict(surfaces=[]),
        dict(surfaces=[projection.surfaces[0]] * 2),
    ]:
        with pytest.raises(ValueError):
            replace(projection, **kwargs)
    with pytest.raises(ValueError, match="tangential"):
        projection({**data, "s0": replace(data["s0"], components=("Ey",))})
    with pytest.raises(ValueError, match="exactly once"):
        replace(projection, frequencies=(2 * LIGHT_SPEED,))(data)
    with pytest.raises(ValueError, match="solid_angle"):
        CollectionConePower(replace(projection, solid_angle_weights=None))
    with pytest.raises(ValueError, match="numerical_aperture"):
        CollectionConePower(projection, 1.1)
    with pytest.raises(ValueError, match="transverse"):
        TargetBeamOverlap(projection, np.zeros((len(projection.directions), 3)))


def test_kernel_shape_errors_are_explicit():
    with pytest.raises(ValueError, match="shapes"):
        near_to_far(
            np.zeros((1, 2, 3)),
            np.zeros((1, 2, 3)),
            np.zeros((3, 3)),
            np.zeros((2, 3)),
            np.ones(2),
            np.ones(1),
            np.eye(3),
        )


def test_float32_angular_grids_and_selected_frequency_order():
    data, projection = radiation_data()
    projection = replace(
        projection, directions=np.asarray(projection.directions, np.float32)
    )
    np.testing.assert_allclose(
        np.linalg.norm(projection.directions, axis=-1), 1, atol=1e-12
    )
    # Surface spectra can have different recorded ordering and extra entries.
    expanded = {
        n: replace(
            d,
            electric=jnp.concatenate((d.electric, 2 * d.electric)),
            magnetic=jnp.concatenate((d.magnetic, 2 * d.magnetic)),
            frequencies=(LIGHT_SPEED, 1.1 * LIGHT_SPEED),
        )
        for n, d in data.items()
    }
    full = projection(expanded)
    ordered = replace(projection, frequencies=(1.1 * LIGHT_SPEED, LIGHT_SPEED))(
        expanded
    )
    np.testing.assert_allclose(ordered.electric, full.electric[::-1], rtol=2e-6)
    reordered = {
        n: replace(
            d,
            electric=d.electric[::-1],
            magnetic=d.magnetic[::-1],
            frequencies=d.frequencies[::-1],
        )
        for n, d in expanded.items()
    }
    selected = replace(projection, frequencies=(LIGHT_SPEED,))(reordered)
    np.testing.assert_allclose(selected.electric, full.electric[:1], rtol=2e-6)
