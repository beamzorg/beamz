"""Run gradient updates and record optimization history."""

import time
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from beamz.optimization.design.validation import step_count
from beamz.optimization.optimizer.result import OptimizationStep


def _optimize(
    result,
    stop,
    *,
    evaluate,
    optimizer,
    settings,
    parameter_mask=None,
    evaluate_after=None,
    snapshots=False,
    score_sign=1,
    checkpoint=None,
    checkpoint_every=1,
    callback=None,
):
    """Shared gradient/update/history loop for rectangular and arbitrary-mask designs."""
    step_count(checkpoint_every, "checkpoint_every", minimum=1)
    if stop < result.completed_steps:
        raise ValueError("steps must be at least the number of completed steps.")
    initial_steps = result.completed_steps
    params = jnp.asarray(result.final_params)
    # Checkpoints decode NumPy leaves; keep resumed arithmetic identical to live JAX state.
    state = jax.tree.map(jnp.asarray, result.optimizer_state)
    for index in range(result.completed_steps, stop):
        start = time.perf_counter()
        current = settings(index)
        (score, (post, penalty)), gradient = evaluate(params, current)
        if (
            not np.isfinite([float(score), float(post), float(penalty)]).all()
            or not np.isfinite(gradient).all()
        ):
            raise FloatingPointError("Nonfinite objective, penalty, or gradient.")
        if parameter_mask is not None:
            gradient = jnp.where(parameter_mask, gradient, 0)
        updates, state = optimizer.update(-gradient, state, params)
        updated = jnp.clip(params + cast(jax.Array, updates), 0, 1)
        if parameter_mask is not None:
            updated = jnp.where(parameter_mask, updated, params)
        max_update = float(jnp.max(jnp.abs(updated - params)))
        after = None if evaluate_after is None else evaluate_after(updated, current)
        record = OptimizationStep(
            index + 1,
            cast(float | None, current.get("beta")),
            float(score) * score_sign,
            after,
            float(np.linalg.norm(gradient)),
            max_update,
            time.perf_counter() - start,
            float(post),
            float(penalty),
        )
        params = updated
        result = result._append(
            params,
            state,
            record,
            snapshots=snapshots,
            gradient=gradient,
        )
        if checkpoint is not None and (
            result.completed_steps % checkpoint_every == 0
            or result.completed_steps == stop
        ):
            result.save(checkpoint)
        if callback is not None:
            callback(result)
    if checkpoint is not None and result.completed_steps == initial_steps:
        result.save(checkpoint)
    return result
