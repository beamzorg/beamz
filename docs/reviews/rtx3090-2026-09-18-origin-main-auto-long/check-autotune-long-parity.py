import os,json
from pathlib import Path
from types import SimpleNamespace
for key in list(os.environ):
 if key.startswith('BEAMZ_CUDA_'):del os.environ[key]
os.environ['BEAMZ_CUDA_AUTOTUNE']='off'
import jax
from scripts.benchmark_cuda_realistic import build_simulation
from beamz.simulation.execute import build_scan,initial_program_state
from beamz.simulation.cuda.tuning import _state_hashes
root=Path.cwd();out=root/'docs/reviews/rtx3090-2026-09-18-origin-main-auto-long'
sim=build_simulation(SimpleNamespace(shape=(1024,256,64),steps=1025,presteps=0,pml=12,monitors=2,frequencies=3,material='binary',source='mode',monitor_type='mode'))
p=sim.compile(num_steps=1025,backend='cuda_streamed');s=initial_program_state(p,t=0,current_step=0,monitor_steps=1025)
exe=build_scan(p).lower(s,p.coefficients).compile();result=jax.block_until_ready(exe(s,p.coefficients))
expected=json.loads((out/'narrow_long-branch_auto-1.json').read_text())
assert _state_hashes(result)==[x['sha256'] for x in expected['final_state_validation']]
(out/'branch-long-exact-parity.json').write_text(json.dumps({'steps':1025,'shape':[1024,256,64],'complete_state_bitwise_equal':True,'reference':'narrow_long-branch_auto-1.json','canonical_axes':p.config.cuda_storage_axes},indent=2)+'\n')
print('All 30 state leaves match the automatic 1025-step run bitwise.')
