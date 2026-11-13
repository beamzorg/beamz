"""Collect bounded software SASS metrics after proving device-code identity.

This is diagnostic GPU work under the campaign lock and deadline. Direct launch
timings are unusable. No privileged hardware counters or installed module edits.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from scripts.analyze_cuda_sass_range import analyze,instructions

ROOT=Path(__file__).resolve().parents[1]
TOOLKIT=Path('/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda')


def run(diagnostic,parent,contract,workers,fixture,output,kernel,*,match_index=64):
    diagnostic,parent,output=Path(diagnostic).resolve(),Path(parent).resolve(),Path(output).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=json.loads((diagnostic/'frozen-execution-manifest.json').read_text());perf=json.loads((parent/'frozen-execution-manifest.json').read_text());module=Path(manifest['module']);parent_module=Path(perf['module'])
    if sha(module)!=manifest['sha256'] or sha(parent_module)!=perf['sha256']:raise ValueError('Frozen diagnostic/performance module hashes required')
    proof=json.loads((diagnostic/'sass-diagnostic-proof.json').read_text())
    if proof['performance_candidate'] is not False or sha(parent/'source/src/asymmetric_native.cuh')!=proof['parent_sha256']:raise ValueError('Host-only diagnostic must bind exact performance parent')
    campaign=json.loads(Path(contract).read_text());deadline=datetime.datetime.fromisoformat(campaign['gpu_deadline_utc'].replace('Z','+00:00')).timestamp()
    output.mkdir(exist_ok=False);identities=[];assemblies=[]
    for label,binary in (('performance',parent_module),('diagnostic',module)):
        dump=subprocess.run([str(TOOLKIT/'bin/cuobjdump'),'--dump-sass','--function',kernel,str(binary)],capture_output=True,text=True,check=True).stdout;(output/(label+'.sass')).write_text(dump)
        decoded=instructions(dump);normalized=json.dumps(sorted((pc,i['text']) for pc,i in decoded.items()));identities.append(dict(label=label,instructions=len(decoded),sha256=hashlib.sha256(normalized.encode()).hexdigest()));assemblies.append(decoded)
    if assemblies[0]!=assemblies[1]:raise ValueError('Diagnostic changed selected device instructions')
    identity=dict(identical=True,kernel=kernel,rows=identities);(output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    collector=ROOT/'.cache/cuda-trace-150min-2026-11-13-1620/sass-metrics-probe/range-v2.so';collector_contract=json.loads(collector.with_name('range-v2-contract.json').read_text())
    if sha(collector)!=collector_contract['library_sha256']:raise ValueError('Frozen double precision collector hash required')
    workers=Path(workers).resolve();shutil.copytree(workers,output/'workers');worker=output/'workers/benchmark_native.py';deps={}
    for path in (workers/'dependencies').rglob('*'):
        if path.is_file():
            name=path.relative_to(workers/'dependencies');current=ROOT/name
            # Some historical snapshots used a flattened script directory.
            if not current.is_file():current=ROOT/'scripts'/name
            if not current.is_file() or sha(current)!=sha(path):raise ValueError('Worker dependency changed:'+str(name))
            deps[str(current)]=sha(current)
    python=ROOT/'.cache/cuda-native-autoresearch-2026-11-11-1607/native-only/bin/python';metrics=output/'metrics.json'
    command=[str(python),str(worker),'--backend','native','--fixture',str(Path(fixture).resolve()),'--module',str(module),'--method','moving','--moving-depth','16','--ownership','preserve','--warmups','0','--samples','1','--output',str(output/'parity.json')]
    record=dict(diagnostic_only=True,timings_usable=False,kernel_identity=identity,module=manifest,performance_parent=perf,collector_sha256=sha(collector),worker_sha256=sha(worker),dependencies=deps,fixture_sha256=sha(fixture),match_index=match_index,command=command)
    env=dict(os.environ,BEAMZ_SASS_RANGE_LIBRARY=str(collector),BEAMZ_SASS_RANGE_OUTPUT=str(metrics),BEAMZ_SASS_MATCH_INDEX=str(match_index),PYTHONPATH=str(ROOT),LD_LIBRARY_PATH=str(TOOLKIT/'targets/x86_64-linux/lib'))
    with (diagnostic.parent/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);remaining=deadline-time.time()
        if remaining<120:raise RuntimeError('Insufficient diagnostic time before GPU deadline')
        record['start_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(output/'process.json').write_text(json.dumps(record,indent=2)+'\n')
        with (output/'run.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=child.pid
            try:record['returncode']=child.wait(timeout=min(180,remaining))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                record['returncode']=child.returncode;record['timed_out']=True
        record['finish_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(output/'process.json').write_text(json.dumps(record,indent=2)+'\n')
        if record['returncode']:raise RuntimeError('Software diagnostic failed:'+str(output/'run.log'))
    for path,value in deps.items():
        if sha(path)!=value:raise ValueError('Dependency changed during diagnostic:'+path)
    analysis=analyze(metrics,output/'diagnostic.sass')
    if analysis['function']!=kernel:raise ValueError('Software metrics came from a different kernel')
    (output/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    return {k:v for k,v in analysis.items() if k not in ('instructions','dynamic_opcodes','totals')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic','parent','contract','workers','fixture','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--kernel',required=True);p.add_argument('--match-index',type=int,default=64);a=p.parse_args();print(json.dumps(run(a.diagnostic,a.parent,a.contract,a.workers,a.fixture,a.output,a.kernel,match_index=a.match_index)))
