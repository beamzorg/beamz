"""Immutable optimization history and result state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import jax
import numpy as np

from beamz.devices._immutable import readonly_array
from beamz.optimization.optimizer.checkpoint import save_checkpoint
from beamz.optimization.optimizer.schedules import OptimizationSchedule


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

    def export_design(
        self, index=-1, *, threshold=0.5, mode="contour", sample_spacing=None
    ):
        """Export physical density as contours or exact pixels (lengths in metres)."""
        options = dict(threshold=threshold, mode=mode, sample_spacing=sample_spacing)
        return self.design.export_design(
            self.params[index],
            **options,
            beta=self.beta
            if self.schedule is None
            else self.settings(index).get("beta"),
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
        params = readonly_array(params)
        return replace(
            self,
            params=(*self.params, params) if snapshots else (params,),
            optimizer_state=optimizer_state,
            history=(*self.history, record),
            grad=()
            if gradient is None
            else (
                (*self.grad, readonly_array(gradient))
                if snapshots
                else (readonly_array(gradient),)
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
