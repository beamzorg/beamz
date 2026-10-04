"""Same-physics storage/layout study; conversion cost included in timings."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace
import jax
import numpy as np
import beamz._cuda as extension
import beamz.simulation.cuda as cuda
from beamz.simulation import _cuda_abi as abi
from beamz.simulation.execute import build_scan,initial_program_state
from scripts.benchmark_cuda_realistic import build_simulation
from cyclic_storage import wrap_program_call

p=argparse.ArgumentParser()
p.add_argument('--shape',nargs=3,type=int,required=True)
p.add_argument('--material',default='binary')
p.add_argument('--presteps',type=int,default=0)
p.add_argument('--output',required=True)
a=p.parse_args()
sim=build_simulation(SimpleNamespace(shape=tuple(a.shape),steps=256,presteps=a.presteps,
    pml=12,monitors=2,frequencies=3,material=a.material,source='mode',monitor_type='mode'))
base=sim.compile(num_steps=256,backend='cuda_streamed')
state=initial_program_state(base,t=0,current_step=0,monitor_steps=256)
if a.presteps:
    state=sim.advance(state=state,num_steps=a.presteps,backend='cuda_streamed').state
jax.block_until_ready(state)
original=cuda.run_program_steps
executables={}
orders=((0,1,2),(1,2,0),(2,0,1))
for name,extra in [('one_step',0),('cpml_pair',abi.CUDA_TEMPORAL_PAIR|abi.CUDA_CPML_PAIR)]:
    for axes in orders:
        key=name+'_'+''.join(map(str,axes))
        flags=(base.config.cuda_flags & ~(abi.CUDA_TEMPORAL_PAIR|abi.CUDA_CPML_PAIR))|extra
        program=replace(base,config=replace(base.config,cuda_flags=flags))
        cuda.run_program_steps=wrap_program_call(original,axes)
        try:
            executables[key]=build_scan(program).lower(state,program.coefficients).compile()
        finally:
            cuda.run_program_steps=original
def run(key):
    return jax.block_until_ready(executables[key](state,base.coefficients))
reference=jax.device_get(run('one_step_012'))
hashes=[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in jax.tree.leaves(reference)]
for key in executables:
    result=jax.device_get(run(key))
    for i,(ref,got) in enumerate(zip(jax.tree.leaves(reference),jax.tree.leaves(result),strict=True)):
        np.testing.assert_array_equal(got,ref,err_msg=f'{key} leaf {i}')
    del result
    for _ in range(3):run(key)
    print(key,'exact complete-state parity',flush=True)
del reference
samples={key:[] for key in executables}
names=list(executables);random.Random(20260918).shuffle(names)
timing_orders=[]
for reverse in (False,True):
    for offset in range(len(names)):
        order=names[offset:]+names[:offset]
        if reverse:order=order[::-1]
        timing_orders.append(order)
        for key in order:
            start=time.perf_counter();result=run(key);elapsed=time.perf_counter()-start
            samples[key].append(elapsed);del result
cells=int(np.prod(a.shape))
data=dict(shape=a.shape,steps=256,presteps=a.presteps,material=a.material,pml=12,
    precision='fp32',source='mode',monitors=2,frequencies=3,monitor_type='mode',
    logical_cells=cells,conversion_cost_included=True,
    native_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
    prototype_sha256=hashlib.sha256(Path(__file__).with_name('cyclic_storage.py').read_bytes()).hexdigest(),
    state_sha256=hashes,samples_s=samples,timing_orders=timing_orders,
    median_gcups={k:cells*256/statistics.median(v)/1e9 for k,v in samples.items()})
Path(a.output).write_text(json.dumps(data,indent=2)+'\n')
print(data['median_gcups'],flush=True)
