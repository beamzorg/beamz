import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root))
from scripts.benchmark_cuda_shapes import telemetry
out=root/'docs/reviews/rtx3090-2026-09-18-origin-main-auto-comparison'
out.mkdir(exist_ok=True)
main=root/'.cache/perf/origin-main-comparison'
refroot=root/'.cache/perf/origin-main-auto-comparison-states'
worker=root/'scripts/benchmark_cuda_revision.py'
variants={'main':(main,'012','auto'),'branch_auto':(root,'012','auto'),'branch_layout':(root,'120','0')}
# Balance forward/reverse fresh-process order separately for each workload.
cases=[('narrow',(1024,256,64),'binary',['main','branch_auto','branch_auto','main']),('wide',(128,256,512),'binary',['main','branch_auto','branch_auto','main']),('irregular',(257,193,341),'smooth',['main','branch_auto','branch_auto','main'])]
meta={'main_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=main,text=True).strip(),'branch_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cases':cases,'build':'CUDA 13.3.73, Release, SM86, fast math OFF, GNU 16.1.1, --allow-unsupported-compiler','branch_native_source_difference':'retained source differs from working tree only by two explanatory comment lines in temporal_cpml.cuh'}
for name,checkout in [('main',main),('branch',root)]:
 h=hashlib.sha256()
 for p in sorted([*checkout.glob('beamz/**/*.py'),*checkout.glob('cuda/src/*')]):
  if p.is_file():h.update(str(p.relative_to(checkout)).encode());h.update(p.read_bytes())
 meta[name+'_source_sha256']=h.hexdigest()
meta['branch_working_diff']=str(out/'branch-working.diff')
(out/'branch-working.diff').write_bytes(subprocess.check_output(['git','diff','--','beamz','cuda']))
(out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
records=[]
for case,shape,material,order in cases:
 counts={}
 for index,name in enumerate(order):
  counts[name]=counts.get(name,0)+1
  tag=f'{case}-{name}-{counts[name]}'
  checkout,axes,fusion=variants[name]
  env=dict(os.environ,PYTHONPATH=str(checkout),XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.45',PYTHONUNBUFFERED='1')
  cmd=[sys.executable,str(worker),'--root',str(checkout),'--shape',*map(str,shape),'--material',material,'--output',str(out/(tag+'.json')),'--reference',str(refroot/case),'--axes',axes,'--fusion',fusion,'--tuning-cache',str(root/'docs/reviews/rtx3090-2026-09-18-autotune/cache')]
  if index==0:cmd+=['--save-reference']
  if (out/(tag+'.json')).exists():
   data=json.loads((out/(tag+'.json')).read_text());records.append(dict(case=case,variant=name,repeat=counts[name],median_gcups=data['median_gcups']));print('EXISTING',tag,flush=True);continue
  print('START',tag,flush=True)
  readings=[];start=time.monotonic()
  t=telemetry()
  if t['free_mib']<4096 or t['temperature_c']>=80:raise RuntimeError(t)
  with (out/(tag+'.log')).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=checkout,env=env,stdout=log,stderr=log)
   try:
    while proc.poll() is None:
     t=telemetry();t['elapsed_s']=time.monotonic()-start;readings.append(t)
     if t['free_mib']<2048 or t['temperature_c']>=85 or t['elapsed_s']>600:raise RuntimeError(t)
     time.sleep(1)
   finally:
    if proc.poll() is None:
     proc.terminate()
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:proc.kill();proc.wait()
    (out/(tag+'.telemetry.json')).write_text(json.dumps(readings,indent=2)+'\n')
   if proc.returncode:raise RuntimeError(tag+' failed; see '+str(out/(tag+'.log')))
  data=json.loads((out/(tag+'.json')).read_text())
  records.append(dict(case=case,variant=name,repeat=counts[name],median_gcups=data['median_gcups']))
  print('DONE',tag,data['median_gcups'],flush=True)
  (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
print('COMPLETE',flush=True)
