"""Lossless constant-block storage for native electric coefficients.

Blocks contain 32 consecutive samples on each component's own Yee support.
Negative headers name one constant; nonnegative headers point at 32 literal
values. Classification is bit-exact, so mixed interfaces and anisotropy are
never approximated. The three components share a coefficient pool.
"""

from itertools import chain
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from beamz._plane_array import physical_crop, plane_dictionary
from beamz._region_array import TILE_BYTES, RegionArray, broadcast_base, flat_tiles

BLOCK_SIZE = 32


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
            header = codes.reshape(-1)[linear >> 5]
            index = jnp.where(header < 0, ~header, header + (linear & 31))
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


def _layer_runs(array):
    """Flatten repeated trailing axes into bit-exact homogeneous material runs."""
    if not isinstance(array, np.ndarray) or array.strides[-1] != 0:
        return None
    axis = array.ndim
    while axis and array.strides[axis - 1] == 0:
        axis -= 1
    length = int(np.prod(array.shape[axis:], dtype=np.int64))
    # Every run must contain a complete constant block even when unaligned.
    if length < 2 * BLOCK_SIZE or array.size // length * 4 > TILE_BYTES:
        return None
    prefix = array[(slice(None),) * axis + (0,) * (array.ndim - axis)]
    bits = np.asarray(prefix, np.float32).reshape(-1).view(np.uint32)
    starts = np.r_[0, np.flatnonzero(bits[1:] != bits[:-1]) + 1]
    return starts * length, bits[starts]


def _pack_plane_dictionaries(dictionaries, max_bytes, asarray):
    """Classify each physical plane and block alignment once, reusing literals."""
    chunks = []
    components = []

    def classify(bits):
        pages = bits.reshape(-1, BLOCK_SIZE)
        mixed = np.any(pages != pages[:, :1], axis=1)
        chunk = (pages[:, 0].copy(), mixed, pages[mixed].copy())
        chunks.append(chunk)
        return len(chunks) - 1

    for bank, indices in dictionaries:
        blocks = []
        physical = {}
        aligned = {}
        carry = np.empty(0, np.uint32)
        for plane in indices:
            if plane not in physical:
                values = np.ascontiguousarray(bank[plane]).ravel()
                if not np.isfinite(values).all():
                    return None
                physical[plane] = values.view(np.uint32)
            row = physical[plane]
            start = 0
            if carry.size:
                start = min(BLOCK_SIZE - carry.size, row.size)
                carry = np.concatenate((carry, row[:start]))
                if carry.size == BLOCK_SIZE:
                    blocks.append(classify(carry))
                    carry = np.empty(0, np.uint32)
                elif start == row.size:
                    continue
            end = start + (row.size - start) // BLOCK_SIZE * BLOCK_SIZE
            key = (plane, start)
            if end > start:
                if key not in aligned:
                    aligned[key] = classify(row[start:end])
                blocks.append(aligned[key])
            carry = row[end:]
        if carry.size:
            blocks.append(
                classify(np.pad(carry, (0, BLOCK_SIZE - carry.size), mode="edge"))
            )
        components.append(blocks)

    constants = np.unique(np.concatenate([bits[~mixed] for bits, mixed, _ in chunks]))
    words = constants.size + sum(literals.size for _, _, literals in chunks)
    count = sum(sum(chunks[k][0].size for k in blocks) for blocks in components)
    if 4 * (count + words) > max_bytes or words >= np.iinfo(np.int32).max:
        return None
    pool = np.empty(words, np.uint32)
    pool[: constants.size] = constants
    offset = constants.size
    templates = []
    for bits, mixed, literals in chunks:
        header = ~np.searchsorted(constants, bits).astype(np.int32)
        header[mixed] = offset + BLOCK_SIZE * np.arange(
            literals.shape[0], dtype=np.int32
        )
        pool[offset : offset + literals.size] = literals.ravel()
        offset += literals.size
        templates.append(header)
    headers = []
    for blocks in components:
        header = np.concatenate([templates[k] for k in blocks])
        header.setflags(write=False)
        headers.append(asarray(header[None, :]))
    pool.setflags(write=False)
    return (asarray(pool.view(np.float32)),) * len(components), tuple(headers)


def pack_material_blocks(
    values: Sequence[Any], *, max_bytes: float, host: bool = False
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    """Return one shared pool reference and header array per input component.

    Rank-two int32 headers distinguish this encoding from the existing rank-one
    byte IDs. Native sessions upload directly from immutable host storage.
    """
    asarray = np.asarray if host else jnp.asarray
    if len(values) == 0:
        return None
    arrays = []
    for value in values:
        array = value if isinstance(value, RegionArray) else np.asarray(value)
        if array.ndim != 3 or not array.size:
            return None
        arrays.append(array)
    # Two bounded passes discover constants, then write the final pool/headers.
    # Flat block boundaries span rows/planes; a carry preserves their exact order.
    unique = np.empty(0, np.uint32); mixed_count = 0; headers = []
    # Full constant regions preserve the incoming block carry in their recorded
    # output. Reuse their bit-exact scalar and final carry on the writing pass.
    constant_regions = [{} for _ in arrays]
    header_count = sum((array.size + BLOCK_SIZE - 1) // BLOCK_SIZE for array in arrays)
    if 4 * header_count > max_bytes:
        return None
    dictionaries = [plane_dictionary(array) for array in arrays]
    if all(item is not None for item in dictionaries):
        return _pack_plane_dictionaries(dictionaries, max_bytes, asarray)
    runs = [_layer_runs(array) for array in arrays]
    for write in (False, True):
        if write:
            pool_size = unique.size + BLOCK_SIZE * mixed_count
            if 4 * (pool_size + header_count) > max_bytes or pool_size >= np.iinfo(np.int32).max:
                return None
            pool = np.empty(pool_size, np.uint32); pool[:unique.size] = unique; offset = unique.size
        for array, summaries, run in zip(arrays, constant_regions, runs, strict=True):
            if run is not None:
                starts, bits = run
                mixed = np.flatnonzero(starts % BLOCK_SIZE)
                if not write:
                    if not np.isfinite(bits.view(np.float32)).all():
                        return None
                    unique = np.union1d(unique, bits)
                    mixed_count += len(mixed)
                    continue
                # Assign each page to the layer containing its first sample.
                # Only pages crossing an unaligned interface require literals.
                first_pages = (starts + BLOCK_SIZE - 1) // BLOCK_SIZE
                pages = (array.size + BLOCK_SIZE - 1) // BLOCK_SIZE
                repeats = np.diff(first_pages, append=pages)
                ids = np.searchsorted(unique, bits).astype(np.int32)
                header = np.repeat(~ids, repeats)
                for i in range(0, mixed.size, 32768):
                    transitions = mixed[i:i + 32768]
                    positions = starts[transitions]
                    header[positions // BLOCK_SIZE] = offset + BLOCK_SIZE * np.arange(len(transitions))
                    left = np.arange(BLOCK_SIZE) < (positions % BLOCK_SIZE)[:, None]
                    samples = np.where(left, bits[transitions - 1, None], bits[transitions, None])
                    pool[offset:offset + samples.size] = samples.ravel()
                    offset += samples.size
                header.setflags(write=False)
                headers.append(asarray(header[None, :]))
                continue
            carry = np.empty(0, np.uint32); position = 0
            if write: header = np.empty((array.size + 31) // 32, np.int32)
            for number, region in enumerate(chain(flat_tiles(array.shape, array.dtype), (None,))):
                if write and number in summaries:
                    length, value, carry = summaries[number]
                    header[position:position + length] = ~np.int32(np.searchsorted(unique, value))
                    position += length
                    continue
                if region is None:
                    if not carry.size: continue
                    bits = np.pad(carry, (0, BLOCK_SIZE-carry.size), mode="edge"); carry = np.empty(0, np.uint32)
                else:
                    raw = np.asarray(physical_crop(array[region]))
                    if not write and 0 in raw.strides:
                        base = np.asarray(broadcast_base(raw), np.float32).view(np.uint32)
                        scalar = base.flat[0]
                        if np.all(base == scalar) and np.all(carry == scalar):
                            if not np.isfinite(scalar.view(np.float32)):
                                return None
                            pages, remainder = divmod(carry.size + raw.size, BLOCK_SIZE)
                            carry = np.full(remainder, scalar, np.uint32)
                            if pages:
                                unique = np.union1d(unique, scalar)
                            summaries[number] = (pages, scalar, carry)
                            continue
                    tile = np.asarray(raw, np.float32).reshape(-1)
                    if not np.isfinite(tile).all(): return None
                    bits = tile.view(np.uint32)
                    if carry.size: bits = np.concatenate((carry, bits))
                    complete = bits.size // BLOCK_SIZE * BLOCK_SIZE
                    carry = bits[complete:].copy(); bits = bits[:complete]
                pages = bits.reshape(-1, BLOCK_SIZE)
                if not write and bits.size and bits[0] == bits[-1] and np.all(bits == bits[0]):
                    value = bits[0]
                    unique = np.union1d(unique, value)
                    summaries[number] = (len(pages), value, carry)
                    continue
                uniform = np.all(pages == pages[:, :1], axis=1)
                count = int(np.count_nonzero(~uniform))
                if not write:
                    unique = np.union1d(unique, np.unique(pages[uniform, 0])); mixed_count += count
                    if 4 * (header_count + unique.size + BLOCK_SIZE * mixed_count) > max_bytes:
                        return None
                else:
                    target = header[position:position+len(pages)]
                    target[uniform] = ~np.searchsorted(unique, pages[uniform, 0]).astype(np.int32)
                    target[~uniform] = offset + BLOCK_SIZE * np.arange(count, dtype=np.int32)
                    pool[offset:offset+count*BLOCK_SIZE] = pages[~uniform].ravel()
                    offset += count * BLOCK_SIZE; position += len(pages)
            if write:
                header.setflags(write=False)
                headers.append(asarray(header[None, :]))
    pool.setflags(write=False)
    shared = asarray(pool.view(np.float32))
    return (shared,) * len(arrays), tuple(headers)
