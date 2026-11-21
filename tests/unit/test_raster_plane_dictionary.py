"""Physical plane storage preserves material bits and immutable NumPy semantics."""

import gc

import numpy as np
import pytest

from beamz._plane_array import (
    mapped_planes,
    mean_diagonal,
    physical_crop,
    plane_dictionary,
)
from beamz.devices._immutable import readonly_array
from beamz.simulation.cuda.storage.material_blocks import (
    pack_material_blocks,
    unpack_material_blocks,
)


@pytest.fixture
def bank():
    native = pytest.importorskip("beamz.design.raster._native")
    if not hasattr(native, "map_planes"):
        pytest.skip("private mmap plane storage requires Linux")
    planes = np.random.default_rng(30).random((4, 9, 21), dtype=np.float32)
    planes[:, 0, :5] = (-0.0, 0.0, -0.0, 1.0, 1.001)
    ids = np.array(
        [[0, 1, 0, 2, 3, 3, 0], [1, 3, 1, 0, 0, 2, 0], [0, 3, 0, 1, 3, 2, 1]]
    )
    return planes, ids


def test_immutable_physical_views_and_owned_result(bank):
    planes, ids = bank
    array = mapped_planes(planes, ids)
    reference = planes[ids]
    material = readonly_array(array[:, ::-2, 2:7:2, 0:11:3])
    view = array[0, :, 1:4, 0:7]
    canonical, shared = plane_dictionary(view)
    np.testing.assert_array_equal(
        canonical[shared].view(np.uint32), reference[0, :, 1:4, :7].view(np.uint32)
    )
    for value in (array, view, canonical):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert np.array_equal(
        physical_crop(array.T[::-2, ::2, :, 0]), reference.T[::-2, ::2, :, 0]
    )
    np.testing.assert_array_equal(
        mean_diagonal(array).view(np.uint32), np.mean(reference, axis=0).view(np.uint32)
    )
    del array, view
    gc.collect()
    assert not material.flags.writeable
    np.testing.assert_array_equal(
        material.view(np.uint32), reference[:, ::-2, 2:7:2, 0:11:3].view(np.uint32)
    )


@pytest.mark.parametrize("width", [1, 9, 21])
def test_unaligned_plane_pool_reuses_literals_and_preserves_bits(bank, width):
    planes, ids = bank
    fields = mapped_planes(planes, ids)
    # Repeat enough xy planes to exercise every alignment of 32-cell material blocks.
    volumes = tuple(field[:, :3, :width] for field in fields)
    pool, headers = pack_material_blocks(volumes, max_bytes=float("inf"), host=True)
    for array, expected, p, h in zip(volumes, planes[ids], pool, headers, strict=True):
        decoded = unpack_material_blocks(p, h, array.shape)
        np.testing.assert_array_equal(
            decoded.view(np.uint32), expected[:, :3, :width].view(np.uint32)
        )
    assert pack_material_blocks(volumes, max_bytes=1, host=True) is None


def test_invalid_plane_addresses_rejected(bank):
    from beamz.design.raster import _native

    planes, _ = bank
    for indices, shape in (
        ([9], (1, 1, 9, 21)),
        ([0, 0], (1, 1, 9, 21)),
        ([], (1, 0, 9, 21)),
    ):
        with pytest.raises(ValueError):
            _native.map_planes(planes, indices, shape)


def test_result_snapshots_share_only_immutable_independent_storage():
    source = np.arange(40, dtype=np.float64).reshape(5, 8)
    snap = readonly_array(source)
    again = readonly_array(snap)
    assert not np.shares_memory(again, snap)
    with pytest.raises(ValueError):
        snap.setflags(write=True)
    source[:] = 17
    assert snap[0, 1] == 1
    crop = readonly_array(snap[:1, :2])
    assert not np.shares_memory(crop, snap)
    repeated = readonly_array(np.broadcast_to(snap[:, :1], (5, 24)))
    assert repeated.strides[1] == 0
    np.testing.assert_array_equal(repeated[:, 0], snap[:, 0])


def test_full_and_cropped_snapshots_copy_only_selected_repeated_layers(bank):
    from beamz._plane_array import plane_owner

    planes, _ = bank
    ids = np.arange(2048) % 3
    volume = mapped_planes(planes, ids.reshape(1, -1))[0]
    copied = readonly_array(volume)
    assert plane_owner(copied) is not None
    assert not np.shares_memory(copied, volume)
    np.testing.assert_array_equal(
        copied[::321].view(np.uint32), planes[ids[::321]].view(np.uint32)
    )
    assert len(np.unique(plane_dictionary(copied)[1])) == 3


def test_cpml_extrusion_preserves_mapped_cell_and_yee_materials(bank):
    from beamz import PML
    from beamz.design import MaterialGrid
    from beamz.devices._boundary_compile import lower_boundaries
    from beamz.lattice import component_shapes

    planes, ids = bank
    volume = mapped_planes(1.0 + planes, ids[:1])[0]
    shape = (6, 8, 20)
    supports = component_shapes(shape)
    permittivity = volume[:6, :8, :20]
    ex_z, ex_y, ex_x = supports["Ex"]
    yee_x = volume[:ex_z, :ex_y, :ex_x]
    dx = 1e-7
    grid = MaterialGrid(
        permittivity,
        np.float32(0),
        np.float32(1),
        dx,
        shape,
        yee_materials={"eps_x": yee_x},
    )
    boundaries = (PML(edges=("left", "back"), thickness=2 * dx, formulation="cpml"),)
    lowered = lower_boundaries(
        grid,
        supports,
        boundaries,
        tuple(n * dx for n in reversed(shape)),
        1e-16,
        region_setup=True,
    )
    for original, actual in (
        (permittivity, lowered.permittivity),
        (yee_x, lowered.yee_materials["eps_x"]),
    ):
        assert plane_dictionary(actual) is not None
        expected = original.copy()
        expected[:, :, :2] = expected[:, :, 2:3]
        expected[-2:] = expected[-3:-2]
        np.testing.assert_array_equal(actual, expected)
