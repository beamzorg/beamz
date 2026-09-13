"""Deterministic continuation schedules indexed by the absolute Adam step."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ._validation import step_count as _step
from .region import Specification


@dataclass(frozen=True)
class LinearSchedule(Specification):
    start: float
    stop: float
    num_steps: int

    def __post_init__(self):
        if not np.isfinite([self.start, self.stop]).all():
            raise ValueError("Schedule endpoints must be finite.")
        if _step(self.num_steps) < 2:
            raise ValueError("LinearSchedule requires at least two steps.")

    def __call__(self, step):
        fraction = min(_step(step) / (self.num_steps - 1), 1.0)
        return float(self.start * (1 - fraction) + self.stop * fraction)


@dataclass(frozen=True)
class StepSchedule(Specification):
    before: float
    after: float
    switch_step: int

    def __post_init__(self):
        _step(self.switch_step)
        if not np.isfinite([self.before, self.after]).all():
            raise ValueError("Schedule values must be finite.")

    def __call__(self, step):
        return float(self.before if _step(step) < self.switch_step else self.after)


ScheduleValue = float | LinearSchedule | StepSchedule


def _endpoints(value):
    if isinstance(value, LinearSchedule):
        return (value.start, value.stop)
    if isinstance(value, StepSchedule):
        return (value.before, value.after)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Use a finite scalar, LinearSchedule, or StepSchedule.")
    if not np.isfinite(value):
        raise ValueError("Schedule values must be finite.")
    return (value,)


def _value(value, step):
    return value(step) if callable(value) else float(value)


@dataclass(frozen=True)
class OptimizationSchedule(Specification):
    """Schedule projection beta, the penalty multiplier, and objective kwargs.

    Linear schedules have their own fixed horizon. Pausing an optimizer by
    shortening num_steps therefore never changes the continuation trajectory.
    Beta overrides all FilterProject transformations in the design region.
    """

    beta: ScheduleValue | None = None
    penalty_weight: ScheduleValue = 1.0
    objective_kwargs: Mapping[str, ScheduleValue] = field(default_factory=dict)

    def __post_init__(self):
        if self.beta is not None and min(_endpoints(self.beta)) <= 0:
            raise ValueError("Scheduled beta must be positive.")
        if min(_endpoints(self.penalty_weight)) < 0:
            raise ValueError("Scheduled penalty_weight must be nonnegative.")
        for name, value in self.objective_kwargs.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("Objective keyword names must be valid identifiers.")
            _endpoints(value)
        object.__setattr__(
            self, "objective_kwargs", MappingProxyType(dict(self.objective_kwargs))
        )

    def values(self, step):
        _step(step)
        return dict(
            beta=None if self.beta is None else _value(self.beta, step),
            penalty_weight=_value(self.penalty_weight, step),
            post_process_kwargs={
                name: _value(value, step)
                for name, value in self.objective_kwargs.items()
            },
        )
