"""Independent coordinate/operator checks without importing either FDTD engine."""

import numpy as np
import pytest

from scripts.fdtdx_release_adapter import (
    component_shape,
    monitor_gathers,
    source_coordinates,
)


def affine_monitor(shape=(32, 32, 32)):
    data = {}
    monitor = {"components": {}}
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        sites = [(15, 15, 15), (15, 15, 16), (0, 0, 0)]
        data[component + "_indices"] = np.array(
            [
                [
                    np.ravel_multi_index(p, component_shape(shape, component))
                    for p in sites
                ]
            ]
        )
        data[component + "_weights"] = np.array([[0.25, 0.75, 0]], np.float32)
        monitor["components"][component] = {
            "indices": component + "_indices",
            "weights": component + "_weights",
        }
    return monitor, data


def test_common_gather_preserves_affine_physical_field_and_ignores_zero_padding():
    monitor, data = affine_monitor()
    lower, upper, indices, weights = monitor_gathers(monitor, data, (32, 32, 32))
    assert lower == (15, 15, 15)
    assert upper == (17, 16, 16)
    xyz = np.indices(tuple(b - a for a, b in zip(lower, upper, strict=True)))
    xyz += np.array(lower)[:, None, None, None]
    for component in range(6):
        field = 3 * xyz[0] + 2 * xyz[1] + xyz[2] + component * 100
        actual = np.sum(field.ravel()[indices[component]] * weights[component], axis=0)
        np.testing.assert_array_equal(actual, [92.25 + 100 * component])


def test_gather_rejects_a_nonzero_weight_on_missing_terminal_node():
    monitor, data = affine_monitor()
    data["Ex_indices"][0, 0] = np.ravel_multi_index(
        (32, 15, 15), component_shape((32,) * 3, "Ex")
    )
    with pytest.raises(ValueError, match="absent FDTDX terminal"):
        monitor_gathers(monitor, data, (32,) * 3)


def test_slab_source_transpose_preserves_each_coefficients_physical_site():
    coefficients = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    term = {"slab_starts": [4, 8, 12], "slab_sizes": [2, 3, 2], "coeff": "c"}
    xyz, values = source_coordinates(term, {"c": coefficients}, (32,) * 3)
    for xyz_point, coefficient in zip(xyz.T, values, strict=True):
        x, y, z = xyz_point
        assert coefficient == coefficients[z - 4, y - 8, x - 12]


def test_source_rejects_absent_terminal_nodes():
    term = {"slab_starts": [32, 8, 12], "slab_sizes": [1, 1, 1], "coeff": "c"}
    with pytest.raises(ValueError, match="absent FDTDX terminal"):
        source_coordinates(term, {"c": np.ones(1)}, (32,) * 3)
