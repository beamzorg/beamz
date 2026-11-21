"""Access the physical dictionary behind an immutable layered raster."""

import sys
from importlib import import_module

import numpy as np


def plane_owner(value):
    while isinstance(value, np.ndarray):
        value = value.base
    if value is None:
        return None
    native = sys.modules.get("beamz.design.raster._native")
    if isinstance(value, getattr(native, "PlaneMapping", ())) and not value.physical:
        return value
    return None


def physical_crop(value):
    """Gather a strided crop without faulting every repeated virtual mapping."""
    owner = plane_owner(value)
    if owner is None or not value.size:
        return value
    bank = owner.physical_planes()
    ids = owner.plane_ids()
    pitch = owner.pitch
    position = value.ctypes.data - owner.address
    outer = tuple(
        i for i, s in enumerate(value.strides) if abs(s) >= pitch and value.shape[i] > 1
    )
    inner = tuple(i for i in range(value.ndim) if i not in outer)
    shape = tuple(value.shape[i] for i in inner)
    strides = tuple(value.strides[i] if value.shape[i] > 1 else 0 for i in inner)
    lo = sum(min(0, (n - 1) * s) for n, s in zip(shape, strides, strict=True))
    hi = (
        sum(max(0, (n - 1) * s) for n, s in zip(shape, strides, strict=True))
        + value.itemsize
    )
    start = position % pitch
    # Operations such as flattening adjacent contiguous z planes can merge
    # axes. Ordinary NumPy indexing remains correct for such uncommon views.
    if (
        start + lo < 0
        or start + hi > pitch
        or any(value.strides[i] % pitch for i in outer)
    ):
        return value

    def read(offset):
        plane, within = divmod(offset, pitch)
        return np.ndarray(
            shape, value.dtype, bank, int(ids[plane]) * pitch + within, strides
        )

    if not outer:
        return read(position)
    result = np.empty(value.shape, value.dtype)
    for indices in np.ndindex(*(value.shape[i] for i in outer)):
        key = [slice(None)] * value.ndim
        offset = position
        for axis, index in zip(outer, indices, strict=True):
            key[axis] = index
            offset += index * value.strides[axis]
        result[tuple(key)] = read(offset)
    return result


def plane_dictionary(array):
    """Return the canonical 2D crops and per-z indices for a Yee volume."""
    owner = plane_owner(array)
    if owner is None or array.ndim != 3 or array.dtype != np.float32 or not array.size:
        return None
    _, _, _, nx = owner.logical_shape
    pitch = owner.pitch
    if array.strides != (pitch, nx * 4, 4):
        return None
    first, within = divmod(array.ctypes.data - owner.address, pitch)
    if within + (array.shape[1] - 1) * nx * 4 + array.shape[2] * 4 > pitch:
        return None
    ids = owner.plane_ids()[first : first + array.shape[0]]
    raw = owner.physical_planes()
    bank = np.ndarray(
        (raw.shape[0], *array.shape[1:]), array.dtype, raw, within, (pitch, nx * 4, 4)
    )
    return bank, ids


def _plane_dictionaries(arrays):
    dictionaries = []
    for array in arrays:
        dictionary = plane_dictionary(array)
        if dictionary is None:
            return None
        dictionaries.append(dictionary)
    return dictionaries


def material_planes(value, *, unique=False):
    """Iterate material in logical C-order or visit each physical plane once."""
    if plane_owner(value) is None or value.ndim not in (3, 4):
        return None
    arrays = value if value.ndim == 4 else (value,)
    dictionaries = _plane_dictionaries(arrays)
    if dictionaries is None:
        return None
    return (
        bank[i]
        for bank, ids in dictionaries
        for i in (np.unique(ids) if unique else ids)
    )


def mapped_planes(rows, ids):
    native = import_module("beamz.design.raster._native")

    bank = np.ascontiguousarray(rows, dtype=np.float32)
    c, z = ids.shape
    try:
        return native.map_planes(bank, ids.ravel(), (c, z, *bank.shape[1:]))
    except OSError:
        dense = bank[ids]
        dense.setflags(write=False)
        return dense


def detached_planes(value):
    """Own a compact independent snapshot when the selected planes repeat."""
    if plane_owner(value) is None or value.ndim not in (3, 4) or value.nbytes < 1 << 20:
        return None
    rows = value if value.ndim == 4 else (value,)
    parts = _plane_dictionaries(rows)
    if not parts:
        return None
    physical = parts[0][0]
    if any(bank.ctypes.data != physical.ctypes.data for bank, _ in parts):
        return None
    ids = np.stack([index for _, index in parts])
    observed, inverse = np.unique(ids, return_inverse=True)
    if 2 * observed.size >= ids.size:
        return None
    snapshot = mapped_planes(physical[observed], inverse.reshape(ids.shape))
    return snapshot if value.ndim == 4 else snapshot[0]


def mean_diagonal(diagonal):
    dictionaries = _plane_dictionaries(diagonal)
    if dictionaries is None:
        return None
    banks, indices = zip(*dictionaries, strict=True)
    combinations, inverse = np.unique(
        np.stack(indices, axis=1), axis=0, return_inverse=True
    )
    rows = [
        np.mean([bank[i] for bank, i in zip(banks, combo, strict=True)], axis=0)
        for combo in combinations
    ]
    return mapped_planes(rows, inverse.reshape(1, -1))[0]
