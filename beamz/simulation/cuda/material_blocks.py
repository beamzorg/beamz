"""Lossless constant-block storage for native electric coefficients.

Blocks contain 32 consecutive samples on each component's own Yee support.
Negative headers name one constant; nonnegative headers point at 32 literal
values. Classification is bit-exact, so mixed interfaces and anisotropy are
never approximated. The three components share a coefficient pool.
"""

import jax.numpy as jnp
import numpy as np

from beamz.simulation.material_blocks import BLOCK_SIZE, block_pool_indices


def differentiable_coefficients(coeffs, state):
    """Expand native lossless encodings before the existing material adjoint.

    Keep this gather outside the native custom VJP: JAX then accumulates shared
    pool gradients correctly. Ordinary trainable Yee arrays pass through.
    """
    changes = {}
    for c in "xyz":
        codes = getattr(coeffs, "e_source_" + c)
        if codes.dtype != jnp.int32 or codes.ndim not in (1, 2):
            continue
        field = getattr(state, "e" + c)
        linear = jnp.arange(field.size, dtype=jnp.int32)
        if codes.ndim == 2:
            index = block_pool_indices(codes, field.size)
        else:
            word = codes.astype(jnp.uint32)[linear >> 2]
            index = (word >> (8 * (linear & 3))) & 255
        pool = getattr(coeffs, "e_decay_" + c)
        changes["e_source_" + c] = pool[index].reshape(field.shape)
        changes["e_decay_" + c] = jnp.float32(1)
    return coeffs._replace(**changes) if changes else coeffs


def unpack_material_blocks(pool, headers, shape):
    """Restore exact coefficients at setup if layout tuning selects a rotation."""
    values = np.asarray(pool)
    headers = np.asarray(headers).ravel()
    result = np.empty(headers.size * BLOCK_SIZE, np.float32)
    lanes = np.arange(BLOCK_SIZE, dtype=np.int32)
    for start in range(0, headers.size, 1 << 15):
        h = headers[start : start + (1 << 15), None]
        addresses = np.where(h < 0, ~h, h + lanes)
        result[start * BLOCK_SIZE : (start + h.size) * BLOCK_SIZE] = values[
            addresses
        ].ravel()
    return result[: np.prod(shape)].reshape(shape)


def pack_material_blocks(values, *, max_bytes):
    """Return (three shared pools, three header arrays), or retain the baseline.

    Rank-two int32 headers distinguish this encoding from the existing rank-one
    byte IDs. Entirely uniform grids should use scalar coefficients instead.
    """
    blocks, constant = [], []
    for value in values:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 3 or not array.size or not np.isfinite(array).all():
            return None
        bits = array.ravel().view(np.uint32)
        padded = np.pad(bits, (0, -bits.size % BLOCK_SIZE), mode="edge")
        pages = padded.reshape(-1, BLOCK_SIZE)
        blocks.append(pages)
        constant.append(np.all(pages == pages[:, :1], axis=1))
    unique = np.unique(
        np.concatenate([b[c, 0] for b, c in zip(blocks, constant, strict=True)])
    )
    mixed_count = sum(np.count_nonzero(~c) for c in constant)
    pool_size = unique.size + BLOCK_SIZE * mixed_count
    total_bytes = 4 * (pool_size + sum(c.size for c in constant))
    if total_bytes > max_bytes or pool_size >= np.iinfo(np.int32).max:
        return None
    pool = np.empty(pool_size, dtype=np.uint32)
    pool[: unique.size] = unique
    headers, offset = [], unique.size
    for pages, uniform in zip(blocks, constant, strict=True):
        header = np.empty(uniform.size, dtype=np.int32)
        header[uniform] = ~np.searchsorted(unique, pages[uniform, 0]).astype(np.int32)
        count = np.count_nonzero(~uniform)
        header[~uniform] = offset + BLOCK_SIZE * np.arange(count, dtype=np.int32)
        pool[offset : offset + count * BLOCK_SIZE] = pages[~uniform].ravel()
        offset += count * BLOCK_SIZE
        headers.append(jnp.asarray(header[None, :]))
    shared = jnp.asarray(pool.view(np.float32))
    return (shared,) * 3, tuple(headers)
