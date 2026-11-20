from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from typing import TypeAlias, cast

import numpy as np

from beamz._plane_array import material_planes

PrimitiveToken: TypeAlias = bool | int | float | str | None
HashToken: TypeAlias = PrimitiveToken | tuple["HashToken", ...]


def object_kind(obj: object) -> str:
    return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"


def array_cache_token(value: object) -> HashToken | None:
    arr = np.asarray(value)
    if arr.dtype == object:
        return None
    physical = material_planes(arr)
    if physical is not None:
        digest = hashlib.blake2b(digest_size=16)
        for plane in physical:
            digest.update(np.ascontiguousarray(plane).view(np.uint8).data)
    elif arr.flags.c_contiguous:
        raw = arr.reshape(-1).view(np.uint8)
        digest = hashlib.blake2b(raw.data, digest_size=16)
    else:
        # Preserve the dense C-order digest for strided or broadcast material
        # volumes while bounding the temporary used for each noncontiguous slice.
        digest = hashlib.blake2b(digest_size=16)
        for chunk in np.nditer(
            arr, flags=["buffered", "external_loop", "zerosize_ok"], order="C",
            buffersize=max(1, (1 << 20) // max(1, arr.itemsize)),
        ):
            digest.update(np.ascontiguousarray(chunk).view(np.uint8).data)
    return ("array", tuple(int(v) for v in arr.shape), str(arr.dtype), digest.hexdigest())


def cache_token(value: object, *, legacy_defaults: bool = False) -> HashToken:
    tokenize = _legacy_cache_token if legacy_defaults else cache_token
    if isinstance(value, np.generic):
        value = cast(object, value.item())
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, slice):
        return ("slice", tuple(int(v) for v in value.indices(2**63 - 1)))
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        token = array_cache_token(value)
        if token is not None:
            return token
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return tuple(
            sorted(
                ((tokenize(k), tokenize(v)) for k, v in mapping.items()),
                key=repr,
            )
        )
    if isinstance(value, tuple | list):
        return tuple(tokenize(v) for v in cast(Iterable[object], value))
    if isinstance(value, set | frozenset):
        return tuple(
            sorted(
                (tokenize(v) for v in cast(Iterable[object], value)),
                key=repr,
            )
        )
    if callable(value):
        return (
            "callable",
            str(getattr(value, "__module__", "")),
            str(getattr(value, "__qualname__", repr(value))),
        )
    for hook in ("cache_spec", "canonical_spec"):
        method = getattr(value, hook, None)
        if callable(method):
            return (object_kind(value), hook, tokenize(method()))
    if is_dataclass(value):
        return (
            object_kind(value),
            dataclass_cache_spec(value, legacy_defaults=legacy_defaults),
        )
    return (object_kind(value), repr(value))


def _legacy_cache_token(value: object) -> HashToken:
    return cache_token(value, legacy_defaults=True)


def dataclass_cache_spec(value, *, legacy_defaults: bool = False) -> HashToken:
    tokenize = _legacy_cache_token if legacy_defaults else cache_token
    tokens = []
    for field in fields(value):
        if field.name.startswith("_") or not field.metadata.get("beamz_cache", True):
            continue
        item = getattr(value, field.name)
        if (
            legacy_defaults
            and field.metadata.get("beamz_checkpoint_legacy_default")
            and item == field.default
        ):
            continue
        tokens.append((field.name, tokenize(item)))
    return tuple(tokens)
