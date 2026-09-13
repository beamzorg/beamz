"""Shared atomic storage for numerical optimization checkpoints."""

import json
import os
import tempfile
from pathlib import Path

import jax
import numpy as np


def save_checkpoint(path, metadata, **arrays):
    """Replace a checkpoint only after all arrays and finite metadata are written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream, metadata=json.dumps(metadata, allow_nan=False), **arrays
            )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_optimizer_state(saved, template, *, count, prefix):
    """Restore arrays against the known optimizer tree, never a serialized tree."""
    expected, tree = jax.tree.flatten(template)
    if count != len(expected):
        raise ValueError("Incompatible optimizer checkpoint.")
    leaves = [np.array(saved[f"{prefix}{i}"]) for i in range(count)]
    if any(
        value.shape != target.shape
        or value.dtype != target.dtype
        or not np.isfinite(value).all()
        for value, target in zip(leaves, expected, strict=True)
    ):
        raise ValueError("Invalid optimizer arrays.")
    return jax.tree.unflatten(tree, leaves)
