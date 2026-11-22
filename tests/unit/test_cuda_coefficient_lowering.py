from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("setup_device", ["cpu", "default"])
def test_native_preparation_never_materializes_a_yee_material_volume(
    monkeypatch: pytest.MonkeyPatch,
    setup_device: str,
) -> None:
    import beamz as bz
    from beamz._region_array import RegionArray
    from beamz.design.discretization import MaterialGrid
    from beamz.simulation.backend import CUDA_ABI_VERSION
    from beamz.simulation.cuda.engine._prepare import _compile

    shape = (40, 41, 42)
    epsilon = np.ones(shape, np.float32)
    epsilon[18:22, 15:25, 12:30] = 4
    grid = MaterialGrid(epsilon, np.float32(0), np.float32(1), 10e-9, shape)
    simulation = bz.Simulation(
        material_grid=grid,
        setup_device=setup_device,
        time=np.arange(4) * 1e-17,
        boundaries=[bz.PML(thickness=120e-9, formulation="cpml")],
    )
    device_calls = []

    def device_table(values, dt, eps0):
        device_calls.append(len(values))
        return np.float32(dt) / (np.float32(eps0) * values)

    module = SimpleNamespace(
        jax_independent=True,
        __abi_version__=CUDA_ABI_VERSION,
        compiled_electric_coefficient_math="cuda_fast_divide_f32",
        electric_coefficient_table=device_table,
    )

    def no_array(*args: object, **kwargs: object) -> None:
        raise AssertionError("Native preparation materialized a Yee material volume")

    monkeypatch.setattr(RegionArray, "__array__", no_array)
    try:
        program = _compile(simulation, module, steps=3)
        assert bool(device_calls) == (setup_device == "default")
        assert isinstance(program.grid.eps_x, RegionArray)
        assert all(
            getattr(program.coefficients, "e_source_" + c).dtype == np.int32
            for c in "xyz"
        )
    finally:
        simulation.clear_compiled_cache()
