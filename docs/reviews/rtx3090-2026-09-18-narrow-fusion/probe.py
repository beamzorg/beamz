"""Private paired study for the isolated narrow-fusion native build."""
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
from beamz.simulation import _cuda_abi as abi
from beamz.simulation.execute import build_scan, initial_program_state
from scripts.benchmark_cuda_realistic import build_simulation

p = argparse.ArgumentParser()
p.add_argument('--shape', nargs=3, type=int, required=True)
p.add_argument('--output', required=True)
p.add_argument('--presteps', type=int, default=0)
p.add_argument('--material', default='binary')
a = p.parse_args()
config = SimpleNamespace(shape=tuple(a.shape), steps=256, presteps=a.presteps, pml=12, monitors=2,
    frequencies=3, material=a.material, source='mode', monitor_type='mode')
sim = build_simulation(config)
base = sim.compile(num_steps=config.steps, backend='cuda_streamed')
state = initial_program_state(base, t=0, current_step=0, monitor_steps=256)
if a.presteps:
    state = sim.advance(state=state, num_steps=a.presteps, backend='cuda_streamed').state
jax.block_until_ready(state)
executables = {}
for shell, flags in [('64x4', 0), ('32x8', abi.CUDA_SHELL32X8)]:
    program = replace(base, config=replace(base.config, cuda_flags=
        (base.config.cuda_flags & ~(abi.CUDA_SHELL32X8 | abi.CUDA_SHELL32X4)) | flags))
    executables[shell] = build_scan(program, donate_state=False).lower(state, base.coefficients).compile()
choices = {
    'ordinary64': ('64x4', '0', 'auto'),
    'ordinary32': ('32x8', '0', 'auto'),
    '20x12x16': ('32x8', '1', '32x8x8'),
    '40x6x16': ('32x8', '1', '64x4x8'),
    '16x16x16': ('32x8', '1', '32x4x8'),
}
def run(name):
    shell, fused, tile = choices[name]
    os.environ['BEAMZ_CUDA_CPML_CORE_FUSION'] = fused
    os.environ['BEAMZ_CUDA_CPML_TILE'] = tile
    return jax.block_until_ready(executables[shell](state, base.coefficients))

reference = jax.device_get(run('ordinary64'))
hashes = [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in jax.tree.leaves(reference)]
for name in choices:
    result = jax.device_get(run(name))
    for i, (ref, got) in enumerate(zip(jax.tree.leaves(reference), jax.tree.leaves(result), strict=True)):
        np.testing.assert_array_equal(got, ref, err_msg=f'{name} leaf {i}')
    del result
    for _ in range(4):
        run(name)
    print(name, 'exact full-state parity', flush=True)
del reference
samples = {k: [] for k in choices}
orders = []
rng = random.Random(20260918)
names = list(choices)
# Balanced cyclic ordering followed by its reverse: each variant occupies
# each position twice, reducing a monotonic clock/temperature drift.
base_order = names[:]
rng.shuffle(base_order)
for reverse in (False, True):
    for offset in range(len(names)):
        order = base_order[offset:] + base_order[:offset]
        if reverse:
            order = order[::-1]
        orders.append(order)
        for name in order:
            start = time.perf_counter()
            result = run(name)
            elapsed = time.perf_counter() - start
            samples[name].append(elapsed)
            del result
cells = int(np.prod(a.shape))
data = dict(shape=a.shape, steps=256, presteps=a.presteps, pml=12, material=a.material,
    source='mode', monitors=2, frequencies=3, monitor_type='mode', precision='fp32',
    logical_cells=cells, orders=orders, state_sha256=hashes,
    native_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
    samples_s=samples, median_gcups={k: cells*256/statistics.median(v)/1e9 for k,v in samples.items()},
    choices=choices)
Path(a.output).write_text(json.dumps(data, indent=2)+'\n')
print(data['median_gcups'], flush=True)
