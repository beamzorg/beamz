"""Adam workflow and inspectable, non-pickle optimization histories."""

import hashlib
import inspect
import json
import os
import tempfile
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from beamz._cache_tokens import cache_token
from beamz.simulation.differentiable import TOPOLOGY_NUMERICS_VERSION

from .design import InverseDesign
from .region import Specification


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


def _frozen(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class InverseDesignResult:
    design: InverseDesign
    params: tuple
    objective_fn_val: tuple = ()
    grad: tuple = ()
    penalty: tuple = ()
    post_process_val: tuple = ()
    opt_state: tuple = ()
    fingerprint: str = ""

    @property
    def history(self):
        return {
            key: getattr(self, key)
            for key in (
                "params",
                "objective_fn_val",
                "grad",
                "penalty",
                "post_process_val",
                "opt_state",
            )
        }

    @property
    def keys(self):
        return tuple(self.history)

    def get(self, key, index=-1):
        return self.history[key][index]

    def get_last(self, key):
        return self.get(key, -1)

    def get_sim(self, index=-1):
        return self.design.to_simulation(self.params[index])

    @property
    def sim_last(self):
        return self.get_sim()

    def get_sim_data(self, index=-1):
        return self.design.to_simulation_data(self.params[index])

    def sim_data_last(self):
        return self.get_sim_data()

    def plot_optimization(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.objective_fn_val, label="Objective")
        ax.plot(self.post_process_val, label="Post-process")
        ax.plot(self.penalty, label="Penalty")
        ax.set(xlabel="Iteration", ylabel="Value")
        ax.legend()
        return ax

    def to_file(self, path):
        """Save numeric history and optimizer arrays; never pickle user code."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        leaves = jax.tree.leaves(self.opt_state)
        metadata = dict(
            schema=1,
            fingerprint=self.fingerprint,
            state_count=len(self.opt_state),
            leaf_count=len(leaves),
        )
        arrays: dict[str, Any] = {
            key: np.asarray(value)
            for key, value in self.history.items()
            if key != "opt_state"
        }
        arrays.update({f"opt_{i}": np.asarray(v) for i, v in enumerate(leaves)})
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, metadata=json.dumps(metadata), **arrays)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class AdamOptimizer(Specification):
    design: InverseDesign
    learning_rate: float
    num_steps: int
    maximize: bool = True
    results_cache_fname: str | None = None
    store_full_results: bool = True
    objective_id: str | None = None

    def __post_init__(self):
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps < 1
        ):
            raise ValueError("num_steps must be a positive integer.")
        if self.objective_id is not None and (
            not isinstance(self.objective_id, str) or not self.objective_id
        ):
            raise ValueError("objective_id must be a nonempty string.")

    def _fingerprint(self, post_process_fn):
        if not callable(post_process_fn):
            raise TypeError("Supply a callable post_process_fn.")
        function = self.objective_id or _function_token(post_process_fn)
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
                self.store_full_results,
                function,
                pipeline,
            )
        )
        return hashlib.sha256(repr(token).encode()).hexdigest()

    def run(self, post_process_fn, callback=None):
        params = self.design.design_region.initial_parameters
        result = InverseDesignResult(
            self.design,
            (_frozen(params),),
            opt_state=(optax.adam(self.learning_rate).init(jnp.asarray(params)),),
            fingerprint=self._fingerprint(post_process_fn),
        )
        return self.continue_run(
            result, post_process_fn=post_process_fn, callback=callback
        )

    def continue_run(self, result, num_steps=None, post_process_fn=None, callback=None):
        if result.fingerprint != self._fingerprint(post_process_fn):
            raise ValueError(
                "Result belongs to a different design, objective, or optimizer."
            )
        done = len(result.objective_fn_val)
        count = max(0, self.num_steps - done) if num_steps is None else num_steps
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("num_steps must be a nonnegative integer.")
        optimizer = optax.adam(self.learning_rate)
        evaluate = jax.value_and_grad(
            lambda params: self.design._evaluate(
                params, post_process_fn, self.maximize
            ),
            has_aux=True,
        )
        params, state = jnp.asarray(result.params[-1]), result.opt_state[-1]
        for index in range(done, done + count):
            (score, (post, penalty)), gradient = evaluate(params)
            if (
                not np.isfinite([float(score), float(post), float(penalty)]).all()
                or not np.isfinite(gradient).all()
            ):
                raise FloatingPointError("Nonfinite objective, penalty, or gradient.")
            updates, state = optimizer.update(-gradient, state, params)
            params = jnp.clip(params + cast(jax.Array, updates), 0, 1)

            def keep(old, value):
                return (*old, value) if self.store_full_results else (value,)

            result = replace(
                result,
                params=keep(result.params, _frozen(params)),
                grad=keep(result.grad, _frozen(gradient)),
                opt_state=keep(result.opt_state, state),
                objective_fn_val=(
                    *result.objective_fn_val,
                    float(score) * (1 if self.maximize else -1),
                ),
                post_process_val=(*result.post_process_val, float(post)),
                penalty=(*result.penalty, float(penalty)),
            )
            if self.results_cache_fname:
                result.to_file(self.results_cache_fname)
            if callback:
                callback(
                    result,
                    step_index=index,
                    aux_data={
                        "post_process_val": float(post),
                        "penalty": float(penalty),
                    },
                )
        return result

    def load_result(self, path, post_process_fn):
        """Restore against this optimizer and an explicitly supplied objective."""
        fingerprint = self._fingerprint(post_process_fn)
        template = optax.adam(self.learning_rate).init(
            jnp.asarray(self.design.design_region.initial_parameters)
        )
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            if meta.get("schema") != 1 or meta.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Checkpoint belongs to a different design, objective, or optimizer."
                )
            history = {
                key: np.array(saved[key])
                for key in (
                    "params",
                    "grad",
                    "objective_fn_val",
                    "post_process_val",
                    "penalty",
                )
            }
            for params in history["params"]:
                self.design.design_region.check_params(params)
            if not all(np.isfinite(value).all() for value in history.values()):
                raise ValueError("Checkpoint history must be finite.")
            count = len(history["objective_fn_val"])
            if (
                len(history["post_process_val"]) != count
                or len(history["penalty"]) != count
            ):
                raise ValueError("Inconsistent scalar checkpoint history.")
            expected_params = count + 1 if self.store_full_results else 1
            expected_grads = count if self.store_full_results else min(count, 1)
            if (
                len(history["params"]) != expected_params
                or len(history["grad"]) != expected_grads
                or meta["state_count"] != expected_params
            ):
                raise ValueError("Inconsistent vector checkpoint history.")
            if (
                expected_grads
                and history["grad"].shape[1:] != self.design.design_region.params_shape
            ):
                raise ValueError("Invalid checkpoint gradient shape.")
            expected, tree = jax.tree.flatten((template,) * meta["state_count"])
            if meta["leaf_count"] != len(expected):
                raise ValueError("Incompatible optimizer checkpoint.")
            leaves = [np.array(saved[f"opt_{i}"]) for i in range(len(expected))]
            if any(
                v.shape != e.shape or v.dtype != e.dtype or not np.isfinite(v).all()
                for v, e in zip(leaves, expected, strict=True)
            ):
                raise ValueError("Invalid optimizer arrays.")
        return InverseDesignResult(
            self.design,
            tuple(_frozen(v) for v in history["params"]),
            objective_fn_val=tuple(history["objective_fn_val"]),
            grad=tuple(_frozen(v) for v in history["grad"]),
            post_process_val=tuple(history["post_process_val"]),
            penalty=tuple(history["penalty"]),
            opt_state=jax.tree.unflatten(tree, leaves),
            fingerprint=fingerprint,
        )

    def complete_run_from_history(self, post_process_fn, callback=None):
        if self.results_cache_fname is None:
            raise ValueError("Set results_cache_fname to resume a saved run.")
        return self.continue_run(
            self.load_result(self.results_cache_fname, post_process_fn),
            post_process_fn=post_process_fn,
            callback=callback,
        )

    def continue_run_from_history(
        self, num_steps=None, post_process_fn=None, callback=None
    ):
        if self.results_cache_fname is None:
            raise ValueError("Set results_cache_fname to resume a saved run.")
        return self.continue_run_from_file(
            self.results_cache_fname,
            num_steps=num_steps,
            post_process_fn=post_process_fn,
            callback=callback,
        )

    def continue_run_from_file(
        self, fname, num_steps=None, post_process_fn=None, callback=None
    ):
        return self.continue_run(
            self.load_result(fname, post_process_fn),
            num_steps=num_steps,
            post_process_fn=post_process_fn,
            callback=callback,
        )
