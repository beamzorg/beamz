import numpy as np
import pytest

from beamz.optimization import LinearSchedule, OptimizationSchedule, StepSchedule


def test_reference_wdm_schedule_and_fixed_horizon():
    schedule = OptimizationSchedule(
        beta=LinearSchedule(1, 50, 50),
        objective_kwargs={"leak_weight": StepSchedule(0, 1, 17)},
    )
    np.testing.assert_allclose(
        [schedule.values(i)["beta"] for i in range(50)], np.linspace(1, 50, 50)
    )
    assert schedule.values(16)["post_process_kwargs"] == {"leak_weight": 0}
    assert schedule.values(17)["post_process_kwargs"] == {"leak_weight": 1}
    assert schedule.values(100)["beta"] == 50
    with pytest.raises(TypeError):
        schedule.objective_kwargs["leak_weight"] = 3


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinearSchedule(1, np.nan, 50),
        lambda: LinearSchedule(1, 50, 1),
        lambda: LinearSchedule(1, 50, True),
        lambda: StepSchedule(0, 1, -1),
        lambda: StepSchedule(0, np.inf, 1),
        lambda: OptimizationSchedule(beta=LinearSchedule(-1, 3, 4)),
        lambda: OptimizationSchedule(penalty_weight=-1),
        lambda: OptimizationSchedule(objective_kwargs={"not valid": 1}),
    ],
)
def test_rejects_invalid_schedules(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()
