import os,json,pathlib,time
from types import SimpleNamespace
from dataclasses import replace
import jax
from scripts.benchmark_compile_capacity import prepare
from beamz.simulation.execute import build_scan
out=pathlib.Path('/workspace/evidence')
a=SimpleNamespace(devices=8,shape=(512,128,256),axis='z',backend='cuda_streamed',workload='modal',steps=32,frequencies=101)
p,s,c=prepare(a,jax.devices()[0]);jax.block_until_ready((s,c))
p=replace(p,config=replace(p.config,cuda_memory_policy='capacity'))
rows=[]
for name,src,mon in [('modal',True,True),('no_sources',False,True),('no_monitors',True,False),('bare',False,False)]:
 q=replace(p,sources=p.sources if src else (),monitors=p.monitors if mon else ())
 start=time.monotonic();exe=build_scan(q,donate_state=True).lower(s,c).compile();m=exe.memory_analysis()
 row=dict(case=name,shape=a.shape,compile_s=time.monotonic()-start,memory={k:getattr(m,k) for k in ('argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','temp_size_in_bytes')})
 rows.append(row);(out/'workspace-ablation.json').write_text(json.dumps(rows,indent=2));(out/f'workspace-{name}.hlo').write_text(exe.as_text());print(row,flush=True)
 del exe
