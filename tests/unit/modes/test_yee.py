import numpy as np
import pytest

from beamz.devices.modes._yee import (
    _center_to_edge_difference,
    _edge_to_center_difference,
    _forward_difference,
    refine_x_mode_at_fixed_beta,
    validate_x_mode_refinement,
)


def _refinement_fixture():
    nz, ny = 8, 9
    shapes = {
        "Ex": (nz, ny),
        "Ey": (nz, ny - 1),
        "Ez": (nz - 1, ny),
        "Hx": (nz - 1, ny - 1),
        "Hy": (nz - 1, ny),
        "Hz": (nz, ny - 1),
    }
    z = np.sin(np.pi * (np.arange(nz) + 0.5) / nz)[:, None]
    y = np.sin(np.pi * (np.arange(ny - 1) + 1.0) / ny)[None, :]
    profiles = {
        name: np.zeros(shape, dtype=np.complex128) for name, shape in shapes.items()
    }
    profiles["Ey"] = z * y
    profiles["Hz"] = profiles["Ey"] / 250.0
    indices = {name: (slice(None), slice(None)) for name in shapes}
    eps = {
        name: np.full(shape, 2.25)
        for name, shape in shapes.items()
        if name.startswith("E")
    }
    mu = {
        name: np.ones(shape) for name, shape in shapes.items() if name.startswith("H")
    }
    kwargs = {
        "component_permittivity": eps,
        "component_permeability": mu,
        "omega": 2.0 * np.pi * 193.0e12,
        "dt": 8.0e-17,
        "resolution": 80.0e-9,
        "k_num": 1.1e7,
        "direction_sign": 1.0,
    }
    return shapes, profiles, indices, kwargs


def test_rectilinear_difference_operators_preserve_uniform_limit():
    from scipy import sparse

    spacing = 0.3
    widths = np.full(5, spacing)
    forward = _forward_difference(sparse, widths.size, spacing)

    np.testing.assert_allclose(
        _center_to_edge_difference(sparse, widths).toarray(),
        forward.toarray(),
    )
    np.testing.assert_allclose(
        _edge_to_center_difference(sparse, widths).toarray(),
        (-forward.T).toarray(),
    )


def test_rectilinear_difference_operators_use_primal_and_dual_spacings():
    from scipy import sparse

    widths = np.asarray([0.2, 0.3, 0.5])

    np.testing.assert_allclose(
        _center_to_edge_difference(sparse, widths).toarray(),
        [
            [-1.0 / 0.25, 1.0 / 0.25, 0.0],
            [0.0, -1.0 / 0.4, 1.0 / 0.4],
        ],
    )
    np.testing.assert_allclose(
        _edge_to_center_difference(sparse, widths).toarray(),
        [
            [1.0 / 0.2, 0.0],
            [-1.0 / 0.3, 1.0 / 0.3],
            [0.0, -1.0 / 0.5],
        ],
    )


def test_fixed_beta_refinement_returns_normalized_yee_shapes():
    shapes, profiles, indices, kwargs = _refinement_fixture()

    refined, residual, ratio, k_num, initial_ratio = refine_x_mode_at_fixed_beta(
        profiles,
        indices,
        **kwargs,
    )

    assert {name: value.shape for name, value in refined.items()} == shapes
    assert residual < 1e-8
    assert abs(ratio - 1.0) < abs(initial_ratio - 1.0)
    assert k_num > 0.0
    assert initial_ratio > 0.0


def test_refinement_validator_accepts_balanced_maxwell_mode():
    _shapes, profiles, indices, kwargs = _refinement_fixture()
    refined, _residual, _ratio, k_num, _initial_ratio = refine_x_mode_at_fixed_beta(
        profiles, indices, **kwargs
    )

    accepted, diagnostics = validate_x_mode_refinement(
        profiles,
        refined,
        indices,
        **{**kwargs, "k_num": k_num},
    )

    assert accepted
    assert diagnostics["magnetic_overlap"] > 0.9
    assert diagnostics["impedance_change"] < 1.01
    assert diagnostics["energy_ratio_change"] < 1.01
    assert diagnostics["faraday_residual"] < 1e-12
    assert diagnostics["ampere_residual"] < 5e-3


def test_refinement_validator_rejects_broken_e_h_balance():
    _shapes, profiles, indices, kwargs = _refinement_fixture()
    refined, _residual, _ratio, k_num, _initial_ratio = refine_x_mode_at_fixed_beta(
        profiles, indices, **kwargs
    )
    broken = dict(refined)
    for component in ("Ex", "Ey", "Ez"):
        broken[component] = refined[component] * 1e11

    accepted, diagnostics = validate_x_mode_refinement(
        profiles,
        broken,
        indices,
        **{**kwargs, "k_num": k_num},
    )

    assert not accepted
    assert diagnostics["impedance_change"] > 1e10
    assert diagnostics["energy_ratio_change"] > 1e20
    assert "E/H impedance change" in diagnostics["rejection_reason"]
    assert "electric/magnetic energy change" in diagnostics["rejection_reason"]


def test_refinement_validator_treats_electric_overlap_as_diagnostic_by_default():
    _shapes, profiles, indices, kwargs = _refinement_fixture()
    refined, _residual, _ratio, k_num, _initial_ratio = refine_x_mode_at_fixed_beta(
        profiles, indices, **kwargs
    )
    electric = np.concatenate(
        [refined[name].reshape(-1) for name in ("Ex", "Ey", "Ez")]
    )
    orthogonal = np.roll(electric, 1)
    orthogonal -= electric * (
        np.vdot(electric, orthogonal) / np.vdot(electric, electric)
    )
    orthogonal *= np.linalg.norm(electric) / np.linalg.norm(orthogonal)
    low_overlap = 0.1 * electric + np.sqrt(1.0 - 0.1**2) * orthogonal
    seed = {name: np.asarray(value).copy() for name, value in refined.items()}
    split = np.cumsum([refined[name].size for name in ("Ex", "Ey")])
    for name, values in zip(
        ("Ex", "Ey", "Ez"), np.split(low_overlap, split), strict=True
    ):
        seed[name] = values.reshape(refined[name].shape)

    accepted, diagnostics = validate_x_mode_refinement(
        seed,
        refined,
        indices,
        **{**kwargs, "k_num": k_num},
    )
    explicitly_rejected, _ = validate_x_mode_refinement(
        seed,
        refined,
        indices,
        **{**kwargs, "k_num": k_num},
        minimum_electric_overlap=0.2,
    )

    assert diagnostics["electric_overlap"] == pytest.approx(0.1)
    assert accepted
    assert not explicitly_rejected


def test_refinement_mode_identity_is_invariant_to_eigenvector_scale(monkeypatch):
    from beamz.devices.modes import _yee

    _shapes, profiles, indices, kwargs = _refinement_fixture()
    reference, *_ = refine_x_mode_at_fixed_beta(profiles, indices, **kwargs)
    eigensolve = _yee._converged_shift_invert_eigenpairs

    def rescaled(*args, **kw):
        values, vectors = eigensolve(*args, **kw)
        vectors = vectors.copy()
        overlaps = np.abs(kw["seed"].conj() @ vectors) / np.linalg.norm(vectors, axis=0)
        # Arbitrary ARPACK scaling must not promote a different eigenmode.
        other = (int(np.argmax(overlaps)) + 1) % vectors.shape[1]
        vectors[:, other] *= 1e8
        return values, vectors

    monkeypatch.setattr(_yee, "_converged_shift_invert_eigenpairs", rescaled)
    actual, *_ = refine_x_mode_at_fixed_beta(profiles, indices, **kwargs)
    for fields in (reference, actual):
        norm = np.linalg.norm(
            np.concatenate([fields[k].ravel() for k in ("Ex", "Ey", "Ez")])
        )
        for name in fields:
            fields[name] = fields[name] / norm
    for name in reference:
        np.testing.assert_allclose(actual[name], reference[name], atol=1e-8, rtol=1e-6)


def test_refinement_converges_frequency_before_returning():
    _shapes, profiles, indices, kwargs = _refinement_fixture()
    _fields, _residual, ratio, _beta, initial = refine_x_mode_at_fixed_beta(
        profiles, indices, **kwargs
    )
    assert abs(initial - 1) > 1e-4
    assert abs(ratio - 1) < 1e-8


def test_node_curl_metrics_use_actual_primal_and_neighboring_dual_widths():
    from beamz.design.grid import RectilinearGrid
    from beamz.devices.modes._yee import transverse_grid_metrics

    edges = np.asarray([0, 1, 3, 7, 15], dtype=float)
    grid = RectilinearGrid(edges, edges, edges)
    metrics, _ = transverse_grid_metrics({"Ex": (slice(1, 4), slice(1, 4), 0)}, grid)
    # Nodes 1,2,3 span cells of width 2 and 4. The outer cells of width
    # 1 and 8 still contribute to the endpoint dual distances.
    for primal, dual in metrics:
        np.testing.assert_array_equal(primal, [2, 4])
        np.testing.assert_array_equal(dual, [1.5, 3, 6])


@pytest.mark.parametrize(
    "loss,elliptical", [(False, False), (True, False), (False, True)]
)
def test_fixed_beta_phase_basis_preserves_complex_modes(loss, elliptical):
    from scipy import sparse
    from scipy.sparse import linalg

    from beamz.devices.modes._yee import _converged_shift_invert_eigenpairs

    phase = np.array([1j, 1j, 1, 1, 1, 1])
    core = np.diag(np.arange(2.0, 8.0)) + np.diag(np.ones(5), 1)
    operator = sparse.csc_matrix(phase[:, None] * core * phase.conj())
    operator = operator * (1 + 0.01j * loss)
    mass = sparse.eye(6)
    seed = phase.copy()
    if elliptical:
        seed[-1] *= 1j

    seen = []

    class Solver:
        def eigs(self, a, **kwargs):
            seen.append(a.dtype.kind)
            return linalg.eigs(a, **kwargs)

    values, modes = _converged_shift_invert_eigenpairs(
        Solver(),
        operator,
        mass,
        seed=seed,
        count=2,
        target=4.6,
        longitudinal_size=2,
    )
    assert seen == ["c" if (loss or elliptical) else "f"]
    np.testing.assert_allclose(operator @ modes, (mass @ modes) * values, atol=1e-7)


def test_corrected_propagation_constant_tracks_an_isolated_real_mode():
    from scipy import sparse
    from scipy.sparse.linalg import aslinearoperator

    from beamz.devices.modes._yee import _track_corrected_mode

    angle = 1e-5
    basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    target = 1.0
    values = np.array([target + 1e-9, 2.0])
    original = np.array([target - 1e-4, 2.0])
    operator = sparse.csc_matrix((basis * values) @ basis.T)
    mass = sparse.eye(2)
    previous = aslinearoperator(np.diag(1 / (original - target)))
    value, vector = _track_corrected_mode(
        operator, mass, target, np.array([1.0, 0.0]), previous
    )
    np.testing.assert_allclose(value, values[0], rtol=1e-12)
    np.testing.assert_allclose(abs(np.vdot(vector, basis[:, 0])), 1, atol=1e-12)
    assert _track_corrected_mode(operator, mass, target, np.ones(2), previous) is None


def test_transverse_projection_restores_full_maxwell_eigenpairs_on_a_graded_grid():
    from scipy import linalg, sparse
    from scipy.sparse import linalg as spla

    from beamz.devices.modes._yee import _transverse_eigenpairs, _yee_curl_operators

    shapes, profiles, _, _ = _refinement_fixture()
    shape = shapes["Ex"]
    rng = np.random.default_rng(119)
    metrics = tuple(
        (rng.uniform(0.7, 1.3, n - 1), rng.uniform(0.8, 1.4, n)) for n in shape
    )
    delta = -0.53j
    curl_e, curl_h, *_ = _yee_curl_operators(
        sparse, profiles, resolution=1, delta=delta, transverse_metrics=metrics
    )
    eps = rng.uniform(1, 5, size=curl_e.shape[1])
    mu = rng.uniform(0.5, 2, size=curl_e.shape[0])
    operator = curl_h @ sparse.diags(1 / mu) @ curl_e
    values, vectors = linalg.eig(operator.toarray(), np.diag(eps))
    index = np.argsort(values.real)[2 * len(values) // 3]
    target = values[index].real * 1.0002
    seed = vectors[:, index]

    def project(epsilon, beta):
        return _transverse_eigenpairs(
            sparse, spla, curl_h, curl_e, epsilon, mu, beta, seed, target, shape, {}
        )

    projected, modes = project(eps, delta)
    k = np.argmin(abs(projected - values[index]))
    np.testing.assert_allclose(projected[k], values[index], rtol=1e-11)
    np.testing.assert_allclose(abs(np.vdot(seed, modes[:, k])), 1, atol=1e-10)
    np.testing.assert_allclose(
        operator @ modes, eps[:, None] * modes * projected, atol=1e-10
    )
    assert project(eps, 0j) is None
    assert project(eps + 1e-5j, delta) is None
