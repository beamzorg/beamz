"""Portable, non-pickle checkpoints for topology optimization."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .topology import TopologyState


@dataclass(frozen=True)
class OptimizationStep:
    """One completed update, evaluated at the recorded projection strength."""

    step: int
    beta: float
    objective_before: float
    objective: float
    gradient_norm: float
    max_update: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TopologyResult:
    """Optimization state and history; the owning problem supplies geometry maps.

    ``state`` is the state *after* all recorded updates. ``beta`` is the strength
    used for the final recorded objective. A checkpoint also records the total
    continuation length, so restarting cannot silently alter the beta schedule.
    """

    state: TopologyState
    history: tuple[OptimizationStep, ...]
    initial_objective: float
    beta: float
    total_steps: int
    problem_fingerprint: str

    @property
    def completed_steps(self):
        return len(self.history)

    @property
    def objective(self):
        return self.history[-1].objective if self.history else self.initial_objective

    def save(self, path):
        """Atomically save arrays and JSON metadata without serializing code."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        leaves = jax.tree.leaves(self.state.optimizer_state)
        metadata = {
            "schema": 1,
            "history": [asdict(item) for item in self.history],
            "initial_objective": self.initial_objective,
            "beta": self.beta,
            "total_steps": self.total_steps,
            "problem_fingerprint": self.problem_fingerprint,
            "optimizer_leaf_count": len(leaves),
        }
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as stream:
                payload: dict[str, Any] = {
                    "metadata": json.dumps(metadata, allow_nan=False),
                    "density": self.state.density,
                    **{
                        f"optimizer_{i}": np.asarray(leaf)
                        for i, leaf in enumerate(leaves)
                    },
                }
                np.savez_compressed(stream, **payload)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def load_result(path, *, topology, fingerprint):
    """Load optimizer arrays using the problem's known optimizer tree schema."""
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"]))
        if meta.get("schema") != 1:
            raise ValueError("Unsupported topology checkpoint schema.")
        if meta["problem_fingerprint"] != fingerprint:
            raise ValueError("Checkpoint belongs to a different optimization problem.")
        density = np.array(saved["density"])
        if (
            density.shape != topology.region_mask.shape
            or not np.isfinite(density).all()
        ):
            raise ValueError("Invalid checkpoint density.")
        if np.any((density < 0) | (density > 1)):
            raise ValueError("Checkpoint density must lie in [0, 1].")
        template = topology._optimizer().init(jnp.asarray(density, dtype=jnp.float32))
        expected, tree = jax.tree.flatten(template)
        if meta["optimizer_leaf_count"] != len(expected):
            raise ValueError("Checkpoint optimizer state is incompatible.")
        leaves = []
        for i, target in enumerate(expected):
            leaf = np.array(saved[f"optimizer_{i}"])
            if (
                leaf.shape != target.shape
                or leaf.dtype != target.dtype
                or not np.isfinite(leaf).all()
            ):
                raise ValueError("Checkpoint optimizer arrays are incompatible.")
            leaves.append(leaf)
    history = tuple(OptimizationStep(**item) for item in meta["history"])
    if [item.step for item in history] != list(range(1, len(history) + 1)):
        raise ValueError("Checkpoint history is not consecutive.")
    if not 0 <= len(history) <= meta["total_steps"]:
        raise ValueError("Checkpoint history exceeds its continuation schedule.")
    state = TopologyState(
        density,
        jax.tree.unflatten(tree, leaves),
        tuple(item.objective for item in history),
    )
    return TopologyResult(
        state,
        history,
        meta["initial_objective"],
        meta["beta"],
        meta["total_steps"],
        fingerprint,
    )
