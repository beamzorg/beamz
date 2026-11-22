from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from typing import TypeAlias, cast

import numpy as np

PrimitiveToken: TypeAlias = bool | int | float | str | None
HashToken: TypeAlias = PrimitiveToken | tuple["HashToken", ...]


def object_kind(obj: object) -> str:
    return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"


def array_cache_token(value: object) -> HashToken | None:
    arr = np.asarray(value)
    if arr.dtype == object:
        return None
    # Hash the existing C-order buffer without allocating a second full-volume
    # bytes object. Non-contiguous inputs still need one contiguous conversion.
    raw = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
    digest = hashlib.blake2b(raw.data, digest_size=16).hexdigest()
    return ("array", tuple(int(v) for v in arr.shape), str(arr.dtype), digest)


def cache_token(value: object) -> HashToken:
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
                ((cache_token(k), cache_token(v)) for k, v in mapping.items()),
                key=repr,
            )
        )
    if isinstance(value, tuple | list):
        return tuple(cache_token(v) for v in cast(Iterable[object], value))
    if isinstance(value, set | frozenset):
        return tuple(
            sorted(
                (cache_token(v) for v in cast(Iterable[object], value)),
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
            return (object_kind(value), hook, cache_token(method()))
    if is_dataclass(value):
        return (object_kind(value), dataclass_cache_spec(value))
    return (object_kind(value), repr(value))


def dataclass_cache_spec(value) -> HashToken:
    return tuple(
        (field.name, cache_token(getattr(value, field.name)))
        for field in fields(value)
        if not field.name.startswith("_") and field.metadata.get("beamz_cache", True)
    )
