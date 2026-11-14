"""Restore and validate optimization checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict

import jax
import numpy as np

from beamz.devices._immutable import readonly_array
from beamz.optimization.design.validation import check_density
from beamz.optimization.optimizer.checkpoint import load_optimizer_state
from beamz.optimization.optimizer.result import InverseDesignResult, OptimizationStep


def load_result(path, *, design, fingerprint, optimizer, shape, schedule=None):
    """Restore against a known design and optimizer; never deserialize code."""
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"]))
        if meta.get("schema") != 2:
            raise ValueError(
                "Unsupported optimization checkpoint schema; start a new run."
            )
        if meta.get("fingerprint") != fingerprint:
            raise ValueError(
                "Checkpoint belongs to a different design or a different optimization problem."
            )
        params, grads = np.array(saved["params"]), np.array(saved["grad"])
        try:
            history = tuple(OptimizationStep(**h) for h in meta["history"])
            n = len(history)
            if [h.step for h in history] != list(range(1, n + 1)) or any(
                not np.isscalar(v) or not np.isfinite(v)
                for h in history
                for v in asdict(h).values()
                if v is not None
            ):
                raise ValueError("Invalid scalar checkpoint history.")
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid scalar checkpoint history.") from exc
        if params.ndim != len(shape) + 1 or not 1 <= len(params) <= n + 1:
            raise ValueError("Invalid checkpoint parameter history.")
        check_density(params, (len(params), *shape), "Checkpoint parameters")
        if (
            grads.shape != (0,)
            and (
                grads.ndim != len(shape) + 1
                or grads.shape[1:] != shape
                or not 0 <= len(grads) <= n
            )
        ) or not np.isfinite(grads).all():
            raise ValueError("Invalid checkpoint gradient shape or values.")
        state = load_optimizer_state(
            saved,
            optimizer.init(jax.numpy.asarray(params[-1], dtype=jax.numpy.float32)),
            count=meta["leaf_count"],
            prefix="opt_",
        )
    total = meta.get("total_steps")
    if total is not None and not 0 <= n <= total:
        raise ValueError("Checkpoint history exceeds its continuation schedule.")
    return InverseDesignResult(
        design,
        tuple(readonly_array(v) for v in params),
        state,
        history,
        tuple(readonly_array(v) for v in grads),
        fingerprint,
        schedule,
        meta.get("initial_objective"),
        total,
        meta.get("beta"),
    )
