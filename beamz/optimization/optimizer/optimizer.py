"""Optimizer configuration and resumable execution."""

import hashlib
import types
from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import numpy as np

from beamz._cache_tokens import cache_token
from beamz.devices._immutable import readonly_array
from beamz.optimization.design.initialization import FilterProject, Specification
from beamz.optimization.design.masked import MaskedDesignRegion
from beamz.optimization.design.spec import InverseDesign
from beamz.optimization.design.validation import step_count
from beamz.optimization.objectives.base import ModalObjective
from beamz.optimization.optimizer.result import InverseDesignResult
from beamz.optimization.optimizer.runtime.identity import (
    _function_token,
    _legacy_checkpoint_token,
)
from beamz.optimization.optimizer.runtime.load import load_result
from beamz.optimization.optimizer.runtime.loop import _optimize
from beamz.optimization.optimizer.schedules import OptimizationSchedule
from beamz.simulation.api.differentiable import TOPOLOGY_NUMERICS_VERSION


@dataclass(frozen=True)
class Optimizer(Specification):
    """Run or resume an Adam or SGD differentiable objective to a target total step count.

    Scalar history and the latest state are retained by default. Enable
    store_full_results to retain parameter and gradient snapshots as well.
    """

    design: InverseDesign
    learning_rate: float
    method: str = "adam"
    evaluate_after: bool = False
    maximize: bool = True
    store_full_results: bool = False
    objective_id: str | None = None
    schedule: OptimizationSchedule | None = None

    def __post_init__(self):
        if self.method not in ("adam", "sgd"):
            raise ValueError("method must be adam or sgd.")
        if self.schedule is not None:
            if not isinstance(self.schedule, OptimizationSchedule):
                raise TypeError("schedule must be an OptimizationSchedule.")
            if (
                self.schedule.beta is not None
                and not isinstance(self.design.design_region, MaskedDesignRegion)
                and not any(
                    isinstance(t, FilterProject)
                    for t in getattr(self.design.design_region, "transformations", ())
                )
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
            for fn in getattr(region, "transformations", ()) + region.penalties
        )
        token = cache_token(
            (
                TOPOLOGY_NUMERICS_VERSION,
                self.design,
                self.learning_rate,
                self.method,
                self.evaluate_after,
                self.maximize,
                function,
                pipeline,
                self.schedule,
                region.penalty_input,
                self.design.adjoint_source_grouping,
            )
        )
        return hashlib.sha256(
            repr(_legacy_checkpoint_token(token)).encode()
        ).hexdigest()

    def run(
        self,
        objective=None,
        callback=None,
        *,
        steps,
        initial_params=None,
        stop_after=None,
        resume=None,
        checkpoint=None,
        checkpoint_every=1,
    ):
        """Reach steps total updates; resume accepts a result or checkpoint path.

        callback(result) receives each completed update. The final checkpoint is
        always saved, including when checkpoint_every exceeds the run length.
        """
        import optax

        steps = step_count(steps, "steps")
        objective = self.design.objective if objective is None else objective
        fingerprint = self._fingerprint(objective)
        stop = steps if stop_after is None else step_count(stop_after, "stop_after")
        if stop > steps:
            raise ValueError("stop_after must not exceed steps.")
        optimizer = getattr(optax, self.method)(self.learning_rate)
        if resume is None:
            params = (
                self.design.design_region.initial_parameters
                if initial_params is None
                else initial_params
            )
            self.design.design_region.check_params(params)
            result = InverseDesignResult(
                self.design,
                (readonly_array(params),),
                optimizer.init(jnp.asarray(params)),
                fingerprint=fingerprint,
                schedule=self.schedule,
                total_steps=steps if stop_after is not None else None,
                _initial_beta=None
                if self.schedule is None
                else cast(float | None, self.schedule.values(0)["beta"]),
            )
        else:
            if initial_params is not None:
                raise ValueError("Use initial_params or resume, not both.")
            result = (
                resume
                if isinstance(resume, InverseDesignResult)
                else self.load_result(resume, objective)
            )
            if result.fingerprint != fingerprint:
                raise ValueError(
                    "Result belongs to a different design, objective, or optimizer."
                )
        if result.total_steps is not None and result.total_steps != steps:
            raise ValueError("Resume must preserve the total schedule horizon.")
        evaluate = self.design._objective_gradient(objective, self.maximize)
        return _optimize(
            result,
            stop,
            evaluate=evaluate,
            optimizer=optimizer,
            settings=lambda i: {} if self.schedule is None else self.schedule.values(i),
            parameter_mask=getattr(self.design.design_region, "region_mask", None),
            evaluate_after=(
                (
                    lambda params, current: (
                        float(
                            self.design._evaluate(
                                params, objective, self.maximize, **current
                            )[0]
                        )
                        * (1 if self.maximize else -1)
                    )
                )
                if self.evaluate_after
                else None
            ),
            snapshots=self.store_full_results,
            score_sign=1 if self.maximize else -1,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            callback=callback,
        )

    def load_result(self, path, post_process_fn=None):
        import optax

        return load_result(
            path,
            design=self.design,
            fingerprint=self._fingerprint(
                self.design.objective if post_process_fn is None else post_process_fn
            ),
            optimizer=getattr(optax, self.method)(self.learning_rate),
            shape=self.design.design_region.params_shape,
            schedule=self.schedule,
        )
