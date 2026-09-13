"""One inspectable result and non-pickle checkpoint format for inverse design."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import jax
import numpy as np

from ._checkpoint import load_optimizer_state, save_checkpoint
from ._validation import check_density
from .schedules import OptimizationSchedule
from .topology import _readonly_array


@dataclass(frozen=True)
class OptimizationStep:
    """One update; objective is None unless the new parameters were evaluated."""

    step: int
    beta: float | None
    objective_before: float
    objective: float | None = None
    gradient_norm: float = 0.0
    max_update: float = 0.0
    elapsed_seconds: float = 0.0
    post_process_val: float = 0.0
    penalty: float = 0.0


@dataclass(frozen=True)
class InverseDesignResult:
    """Latest resumable state, scalar update records, and optional snapshots.

    History scores belong to the parameters before each update. Use
    simulation_data() to evaluate final_params, which includes the last update.
    Only the latest optimizer state is stored, even when snapshots are enabled.
    """

    design: Any
    params: tuple
    optimizer_state: Any
    history: tuple[OptimizationStep, ...] = ()
    grad: tuple = ()
    fingerprint: str = ""
    schedule: OptimizationSchedule | None = None
    initial_objective: float | None = None
    total_steps: int | None = None
    _initial_beta: float | None = None

    @property
    def completed_steps(self):
        return len(self.history)

    @property
    def final_params(self):
        return self.params[-1]

    @property
    def beta(self):
        return self.history[-1].beta if self.history else self._initial_beta

    @property
    def objective(self):
        return self.history[-1].objective if self.history else self.initial_objective

    def settings(self, index=-1):
        index = range(len(self.params))[index]
        step = min(
            max(0, self.completed_steps - 1),
            self.completed_steps + 1 - len(self.params) + index,
        )
        return {} if self.schedule is None else self.schedule.values(step)

    @property
    def schedule_history(self):
        return (
            ()
            if self.schedule is None
            else tuple(self.schedule.values(i) for i in range(self.completed_steps))
        )

    def to_simulation(self, index=-1):
        beta = self.beta if self.schedule is None else self.settings(index)["beta"]
        return self.design.to_simulation(self.params[index], beta=beta)

    def simulation_data(self, index=-1):
        beta = self.beta if self.schedule is None else self.settings(index)["beta"]
        return self.design.to_simulation_data(self.params[index], beta=beta)

    def export_design(self, index=-1, *, threshold=0.5):
        if hasattr(self.design, "topology"):
            return self.design.export_design(self, threshold=threshold)
        return self.design.export_design(
            self.params[index],
            threshold=threshold,
            beta=self.settings(index).get("beta"),
        )

    def plot_optimization(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        for label, values in (
            ("Objective", "objective_before"),
            ("Post-process", "post_process_val"),
            ("Penalty", "penalty"),
        ):
            ax.plot([getattr(h, values) for h in self.history], label=label)
        ax.set(xlabel="Iteration", ylabel="Value")
        ax.legend()
        return ax

    def _append(
        self, params, optimizer_state, record, *, gradient=None, snapshots=False
    ):
        params = _readonly_array(params)
        return replace(
            self,
            params=(*self.params, params) if snapshots else (params,),
            optimizer_state=optimizer_state,
            history=(*self.history, record),
            grad=()
            if gradient is None
            else (
                (*self.grad, _readonly_array(gradient))
                if snapshots
                else (_readonly_array(gradient),)
            ),
            initial_objective=(
                record.objective_before
                if self.initial_objective is None
                else self.initial_objective
            ),
        )

    def save(self, path):
        leaves = jax.tree.leaves(self.optimizer_state)
        save_checkpoint(
            path,
            dict(
                schema=2,
                fingerprint=self.fingerprint,
                history=[asdict(h) for h in self.history],
                initial_objective=self.initial_objective,
                total_steps=self.total_steps,
                beta=self.beta,
                leaf_count=len(leaves),
            ),
            params=np.asarray(self.params),
            grad=np.asarray(self.grad),
            **{f"opt_{i}": np.asarray(v) for i, v in enumerate(leaves)},
        )


TopologyResult = InverseDesignResult


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
        tuple(_readonly_array(v) for v in params),
        state,
        history,
        tuple(_readonly_array(v) for v in grads),
        fingerprint,
        schedule,
        meta.get("initial_objective"),
        total,
        meta.get("beta"),
    )
