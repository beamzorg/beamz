from types import SimpleNamespace
import jax
import numpy as np
import pytest
import beamz.simulation.cuda as cuda
from beamz.simulation.execute import build_scan, initial_program_state
from scripts.benchmark_cuda_realistic import build_simulation
from cyclic_storage import wrap_program_call

@pytest.mark.parametrize('axes', [(1,2,0),(2,0,1)])
@pytest.mark.parametrize('material', ['binary','smooth'])
@pytest.mark.parametrize('schedule', ['ordinary','pair','cpml_pair'])
def test_cyclic_graph(axes, material, schedule, monkeypatch):
    monkeypatch.setenv('BEAMZ_CUDA_CPML_PSI_PRECISION','fp32')
    monkeypatch.setenv('BEAMZ_CUDA_TEMPORAL_STEPS','1' if schedule=='ordinary' else '2')
    monkeypatch.setenv('BEAMZ_CUDA_CPML_TEMPORAL','1' if schedule=='cpml_pair' else '0')
    monkeypatch.setenv('BEAMZ_CUDA_CPML_PAIR_TILE','oriented')
    monkeypatch.setenv('BEAMZ_CUDA_CPML_SPATIAL','0')
    monkeypatch.setenv('BEAMZ_CUDA_FIELD_PADDING','none')
    monkeypatch.setenv('BEAMZ_CUDA_CPML_CORE_FUSION','0')
    sim=build_simulation(SimpleNamespace(shape=(37,49,65),steps=35,pml=12,
        monitors=2,frequencies=3,material=material,source='mode',monitor_type='field'))
    program=sim.compile(num_steps=33,backend='cuda_streamed')
    state=initial_program_state(program,t=0,current_step=0,monitor_steps=35)
    rng=np.random.default_rng(20260918)
    state=state._replace(**{name:rng.normal(size=getattr(state,name).shape).astype(np.float32)*1e-3
        for name in ('ex','ey','ez','hx','hy','hz')},
        **{name:tuple(rng.normal(size=x.shape).astype(np.float32)*1e2 for x in getattr(state,name))
            for name in ('cpml_psi_h_terms','cpml_psi_e_terms')})
    ref_exec=build_scan(program).lower(state,program.coefficients).compile()
    original=cuda.run_program_steps
    monkeypatch.setattr(cuda,'run_program_steps',wrap_program_call(original,axes))
    rotated_exec=build_scan(program).lower(state,program.coefficients).compile()
    monkeypatch.setattr(cuda,'run_program_steps',original)
    ref=ref_exec(state,program.coefficients)
    actual=rotated_exec(state,program.coefficients)
    for i,(expected,got) in enumerate(zip(jax.tree.leaves(ref),jax.tree.leaves(actual),strict=True)):
        np.testing.assert_array_equal(got,expected,err_msg=f'leaf {i}')
    # Continue with a different step count from the canonical result layout.
    from dataclasses import replace
    tail=replace(program,config=replace(program.config,num_steps=2))
    ref_tail=build_scan(tail).lower(ref,program.coefficients).compile()
    monkeypatch.setattr(cuda,'run_program_steps',wrap_program_call(original,axes))
    rotated_tail=build_scan(tail).lower(actual,program.coefficients).compile()
    monkeypatch.setattr(cuda,'run_program_steps',original)
    ref=ref_tail(ref,program.coefficients)
    actual=rotated_tail(actual,program.coefficients)
    for i,(expected,got) in enumerate(zip(jax.tree.leaves(ref),jax.tree.leaves(actual),strict=True)):
        np.testing.assert_array_equal(got,expected,err_msg=f'continued leaf {i}')
