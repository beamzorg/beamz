import os,json,argparse
from dataclasses import replace
from argparse import Namespace
import jax,numpy as np
from scripts.benchmark_cuda_realistic import build_simulation
from beamz.simulation import _cuda_abi as abi
from beamz.simulation.execute import build_scan,initial_program_state
ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=256);ap.add_argument('--jax',action='store_true');ap.add_argument('--output',required=True);args=ap.parse_args()
os.environ['BEAMZ_CUDA_CPML_CORE_FUSION']='1'
sim=build_simulation(Namespace(shape=(61,73,97),steps=args.steps,pml=12,monitors=2,frequencies=3,material='smooth',source='mode',monitor_type='field'))
base=sim.compile(num_steps=args.steps,backend='cuda_streamed')
state=initial_program_state(base,t=0,current_step=0,monitor_steps=args.steps)
rng=np.random.default_rng(20260922)
state=state._replace(**{name:rng.normal(0,1e-3,getattr(state,name).shape).astype(np.float32) for name in ('ex','ey','ez','hx','hy','hz')})
results={}
for name,tile,flag in [('fused','16x8x16',0),('rolling','16x8x16',abi.CUDA_TEMPORAL_PAIR),('monitors','single',abi.CUDA_TEMPORAL_PAIR)]:
 os.environ['BEAMZ_CUDA_PAIR_TILE']=tile
 prog=replace(base,config=replace(base.config,cuda_flags=(base.config.cuda_flags&~abi.CUDA_TEMPORAL_PAIR)|flag))
 exe=build_scan(prog,donate_state=False).lower(state,base.coefficients).compile()
 results[name]=jax.device_get(exe(state,base.coefficients))
 print(name,flush=True)
if args.jax:
 prog=sim.compile(num_steps=args.steps,backend='jax');results['jax']=jax.device_get(build_scan(prog,donate_state=False)(state,prog.coefficients))
out={}
for refname in ['fused']+(['jax'] if args.jax else []):
 out[refname]={}
 for name in ['rolling','monitors']:
  vals=[]
  for (path,ref),actual in zip(jax.tree_util.tree_flatten_with_path(results[refname])[0],jax.tree_util.tree_leaves(results[name]),strict=True):
   ref,actual=np.asarray(ref),np.asarray(actual)
   if not np.issubdtype(ref.dtype,np.inexact):continue
   assert np.isfinite(ref).all() and np.isfinite(actual).all()
   peak=float(np.max(abs(ref),initial=0));err=float(np.max(abs(actual-ref),initial=0));norm=np.linalg.norm(ref.reshape(-1));
   vals.append(dict(path=str(path),peak=peak,error=err,relative_peak=err/max(peak,1e-30),relative_l2=float(np.linalg.norm((actual-ref).reshape(-1))/max(norm,1e-30))))
  out[refname][name]=vals
import beamz._cuda as ext,hashlib
out['native_sha256']=hashlib.sha256(open(ext.__file__,'rb').read()).hexdigest()
open(args.output,'w').write(json.dumps(out,indent=2)+'\n')
for ref,v in out.items():
 if not isinstance(v,dict):continue
 for name,leaves in v.items():print(ref,name,'peak',max(x['relative_peak'] for x in leaves),'l2',max(x['relative_l2'] for x in leaves))
