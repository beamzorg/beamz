"""Exact, differentiable constant-block material storage for JAX propagation."""

from dataclasses import dataclass, replace
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

BLOCK_SIZE = 32


def block_pool_indices(headers, size):
    """Map sample indices to a 32-wide constant-block pool."""
    linear = jnp.arange(size, dtype=jnp.int32)
    header = jnp.asarray(headers).reshape(-1)[linear >> 5]
    return jnp.where(header < 0, ~header, header + (linear & 31))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MaterialBlocks:
    """A value pool and fixed block map on one logical Yee support.

    Uint8 headers are per-sample IDs into a 256-entry pool. Otherwise negative
    int32 headers reference one constant via ``~header`` and other headers
    address 32 literal values. Values are differentiable; integer topology is
    fixed. Recompile or supply dense arrays to untie shared coefficients.
    """

    values: jax.Array | np.ndarray
    headers: jax.Array | np.ndarray
    shape: tuple[int, ...]

    @property
    def size(self):
        return prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def nbytes(self):
        return self.values.nbytes + self.headers.nbytes

    def tree_flatten(self):
        return (self.values, self.headers), self.shape

    @classmethod
    def tree_unflatten(cls, shape, leaves):
        values, headers = leaves
        return cls(values, headers, shape)

    def dense(self):
        """Gather at use sites so XLA can fuse reads into the field update."""
        if self.headers.dtype == np.uint8:
            return jnp.asarray(self.values)[jnp.asarray(self.headers)].reshape(
                self.shape
            )
        index = block_pool_indices(self.headers, self.size)
        return jnp.asarray(self.values)[index].reshape(self.shape)


def material_values(value):
    return value.dense() if isinstance(value, MaterialBlocks) else value


def reciprocal_material(value):
    if isinstance(value, MaterialBlocks):
        return replace(value, values=jnp.reciprocal(value.values))
    return jnp.reciprocal(value)


def compact_material(value):
    """Keep exact scalars/blocks only when they save at least 25% of storage."""
    array = np.asarray(value)
    if (
        array.ndim != 3
        or not array.size
        or array.dtype.kind != "f"
        or array.dtype.itemsize not in (4, 8)
        or not np.isfinite(array).all()
    ):
        return value
    bits = array.ravel().view(np.uint32 if array.dtype.itemsize == 4 else np.uint64)
    if np.all(bits == bits[0]):
        return array.ravel()[0].copy()
    # A byte map uses one gather instead of two dependent gathers. Padding the
    # table to 256 entries proves every uint8 index is in bounds to XLA.
    table = np.empty(0, dtype=bits.dtype)
    chunk = 1 << 20
    for start in range(0, bits.size, chunk):
        table = np.union1d(table, np.unique(bits[start : start + chunk]))
        if table.size > 256:
            break
    if table.size <= 256 and bits.size + 256 * array.itemsize <= 0.75 * array.nbytes:
        codes = np.empty(bits.size, dtype=np.uint8)
        for start in range(0, bits.size, chunk):
            codes[start : start + chunk] = np.searchsorted(
                table, bits[start : start + chunk]
            )
        pool = np.pad(table, (0, 256 - table.size), mode="edge").view(array.dtype)
        return MaterialBlocks(pool, codes, array.shape)
    pages = np.pad(bits, (0, -bits.size % BLOCK_SIZE), mode="edge").reshape(
        -1, BLOCK_SIZE
    )
    constant = np.all(pages == pages[:, :1], axis=1)
    constants = np.unique(pages[constant, 0])
    mixed_count = np.count_nonzero(~constant)
    pool_size = constants.size + BLOCK_SIZE * mixed_count
    if (
        pool_size * array.itemsize + constant.size * 4 > 0.75 * array.nbytes
        or pool_size >= np.iinfo(np.int32).max
        or array.size >= np.iinfo(np.int32).max
    ):
        return value
    pool = np.concatenate((constants, pages[~constant].ravel())).view(array.dtype)
    headers = np.empty(constant.size, np.int32)
    headers[constant] = ~np.searchsorted(constants, pages[constant, 0]).astype(np.int32)
    headers[~constant] = constants.size + BLOCK_SIZE * np.arange(
        mixed_count, dtype=np.int32
    )
    return MaterialBlocks(pool, headers, array.shape)
