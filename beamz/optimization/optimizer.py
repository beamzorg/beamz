"""Adam workflow and inspectable, non-pickle optimization histories."""

import hashlib
import inspect
import time
import types
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from beamz._cache_tokens import cache_token
from beamz.simulation.differentiable import TOPOLOGY_NUMERICS_VERSION

from ._validation import step_count
from .design import InverseDesign
from .objectives import ModalObjective
from .region import FilterProject, Specification
from .result import InverseDesignResult, OptimizationStep, load_result
from .schedules import OptimizationSchedule
from .topology import _readonly_array as _frozen


def _optimize(
    result,
    stop,
    *,
    evaluate,
    optimizer,
    settings,
    update=None,
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
    params = jnp.asarray(result.final_params) if update is None else result.final_params
    state = result.optimizer_state
    for index in range(result.completed_steps, stop):
        start = time.perf_counter()
        current = settings(index)
        (score, (post, penalty)), gradient = evaluate(params, current)
        if (
            not np.isfinite([float(score), float(post), float(penalty)]).all()
            or not np.isfinite(gradient).all()
        ):
            raise FloatingPointError("Nonfinite objective, penalty, or gradient.")
        if update is None:
            updates, state = optimizer.update(-gradient, state, params)
            updated = jnp.clip(params + cast(jax.Array, updates), 0, 1)
            max_update = float(jnp.max(jnp.abs(updated - params)))
        else:
            updated, state, max_update = update(params, state, gradient)
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
            gradient=gradient if update is None else None,
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


def _function_token(function, seen=None):
    """Include closure/configuration values, not merely a callable's name."""
    seen = set() if seen is None else seen
    if id(function) in seen:
        return (function.__module__, function.__qualname__)
    if not isinstance(function, types.FunctionType):
        raise TypeError("For callable objects, supply objective_id.")
    seen.add(id(function))

    def token(value):
        if isinstance(value, types.ModuleType):
            return ("module", value.__name__)
        if isinstance(value, types.FunctionType):
            return (
                _function_token(value, seen)
                if value.__module__ == function.__module__
                else cache_token(value)
            )
        if isinstance(value, types.CodeType):
            return (
                value.co_code.hex(),
                tuple(token(v) for v in value.co_consts),
                value.co_names,
            )
        return cache_token(value)

    closure = inspect.getclosurevars(function)
    return (
        token(function.__code__),
        token(function.__defaults__),
        token(function.__kwdefaults__),
        tuple(
            sorted(
                (name, token(value))
                for name, value in {**closure.globals, **closure.nonlocals}.items()
            )
        ),
    )


@dataclass(frozen=True)
class AdamOptimizer(Specification):
    """Run or resume a differentiable objective to a target total step count.

    Scalar history and the latest state are retained by default. Enable
    store_full_results to retain parameter and gradient snapshots as well.
    """

    design: InverseDesign
    learning_rate: float
    maximize: bool = True
    store_full_results: bool = False
    objective_id: str | None = None
    schedule: OptimizationSchedule | None = None

    def __post_init__(self):
        if self.schedule is not None:
            if not isinstance(self.schedule, OptimizationSchedule):
                raise TypeError("schedule must be an OptimizationSchedule.")
            if self.schedule.beta is not None and not any(
                isinstance(t, FilterProject)
                for t in self.design.design_region.transformations
            ):
                raise ValueError(
                    "A beta schedule requires a FilterProject transformation."
                )
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if self.objective_id is not None and (
            not isinstance(self.objective_id, str) or not self.objective_id
        ):
            raise ValueError("objective_id must be a nonempty string.")

    def _fingerprint(self, post_process_fn):
        if not callable(post_process_fn):
            raise TypeError("Supply a callable post_process_fn.")
        function = self.objective_id or (
            cache_token(post_process_fn)
            if isinstance(post_process_fn, ModalObjective)
            else _function_token(post_process_fn)
        )
        # Also track captured configuration in user-defined transformations and
        # penalties; a function's qualified name alone does not identify it.
        region = self.design.design_region
        pipeline = tuple(
            _function_token(fn)
            if isinstance(fn, types.FunctionType)
            else cache_token(fn)
            for fn in region.transformations + region.penalties
        )
        token = cache_token(
            (
                TOPOLOGY_NUMERICS_VERSION,
                self.design,
                self.learning_rate,
                self.maximize,
                function,
                pipeline,
                self.schedule,
                region.penalty_input,
                self.design.adjoint_source_grouping,
            )
        )
        return hashlib.sha256(repr(token).encode()).hexdigest()

    def run(
        self,
        objective,
        callback=None,
        *,
        steps,
        resume=None,
        checkpoint=None,
        checkpoint_every=1,
    ):
        """Reach steps total updates; resume accepts a result or checkpoint path.

        callback(result) receives each completed update. The final checkpoint is
        always saved, including when checkpoint_every exceeds the run length.
        """
        steps = step_count(steps, "steps")
        fingerprint = self._fingerprint(objective)
        if resume is None:
            params = self.design.design_region.initial_parameters
            result = InverseDesignResult(
                self.design,
                (_frozen(params),),
                optax.adam(self.learning_rate).init(jnp.asarray(params)),
                fingerprint=fingerprint,
                schedule=self.schedule,
            )
        else:
            result = (
                resume
                if isinstance(resume, InverseDesignResult)
                else self.load_result(resume, objective)
            )
            if result.fingerprint != fingerprint:
                raise ValueError(
                    "Result belongs to a different design, objective, or optimizer."
                )
        optimizer = optax.adam(self.learning_rate)
        evaluate = jax.value_and_grad(
            lambda params, settings: self.design._evaluate(
                params, objective, self.maximize, **settings
            ),
            has_aux=True,
        )
        return _optimize(
            result,
            steps,
            evaluate=evaluate,
            optimizer=optimizer,
            settings=lambda i: {} if self.schedule is None else self.schedule.values(i),
            snapshots=self.store_full_results,
            score_sign=1 if self.maximize else -1,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            callback=callback,
        )

    def load_result(self, path, post_process_fn):
        return load_result(
            path,
            design=self.design,
            fingerprint=self._fingerprint(post_process_fn),
            optimizer=optax.adam(self.learning_rate),
            shape=self.design.design_region.params_shape,
            schedule=self.schedule,
        )
