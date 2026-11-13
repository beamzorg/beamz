"""Fast A/B of world-equivalent cyclic storage with strict inverse-world parity."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.cache/cuda-native-autoresearch-2026-11-11-1607/native-only/bin/python'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--world-validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    frozen=json.loads((a.candidate/'frozen-execution-manifest.json').read_text());contract=json.loads(a.contract.read_text())
    if contract.get('ABBA') is not False or contract.get('RunPod') is not False:raise ValueError('Explicit local fast A/B contract required')
    module=Path(frozen['module']);control=Path(contract['control_module'])
    if digest(module)!=frozen['sha256'] or digest(control)!=contract['control_sha256']:raise ValueError('Frozen binaries changed')
    validation=json.loads(a.world_validation.read_text())
    if not validation.get('passed') or validation.get('module_sha256')!=frozen['sha256'] or len(validation.get('cases',[]))!=72:
        raise ValueError('Complete72-state inverse-world source/admission validation required')
    internal=json.loads(a.fixture.read_text());world=Path(internal['cyclic_layout']['world_fixture'])
    if digest(world)!=internal['cyclic_layout']['world_fixture_sha256']:raise ValueError('Bound world fixture changed')
    out=a.output.resolve();out.mkdir(exist_ok=False)
    baseline_worker=out/'baseline-worker.py';text=(ROOT/'scripts/benchmark_cuda_large_runtime.py').read_text()
    anchor='"module_sha256": file_digest(args.module),'
    if text.count(anchor)!=1:raise ValueError('Native report identity binding changed')
    text=text.replace(anchor,'"module": str(args.module.resolve()),\n            '+anchor)
    text=text.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1]))',f'sys.path.insert(0,{str(ROOT)!r})')
    baseline_worker.write_text(text);compile(text,str(baseline_worker),'exec')
    files=[ROOT/'scripts'/n for n in ('benchmark_cuda_large_runtime.py','benchmark_cuda_cyclic_runtime.py','cuda_cyclic_fixture.py','cuda_fixture_store.py','cuda_research_compare.py','compare_cuda_cyclic_runtime.py')]
    identities={str(f):digest(f) for f in files};rows=[]
    report=dict(schema='beamz.cyclic-architecture-experiment.v1',contract=contract,candidate=frozen,
                world_validation=str(a.world_validation.resolve()),world_validation_sha256=digest(a.world_validation),
                worker_digests=identities,baseline_worker_sha256=digest(baseline_worker),rows=rows,
                setup_and_inverse_conversion_excluded_from_timed_solve=True)
    deadline=datetime.datetime.fromisoformat(contract['gpu_deadline_utc'].replace('Z','+00:00')).timestamp()
    def run(label,command,limit):
        remaining=deadline-time.time()
        if remaining<30:raise RuntimeError('Campaign GPU deadline')
        row=dict(label=label,command=list(map(str,command)),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());rows.append(row)
        with (out/(label+'.log')).open('w') as log:
            start=time.monotonic();child=subprocess.Popen(row['command'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,
                    env=dict(os.environ,PYTHONPATH=str(ROOT)),start_new_session=True);row['pid']=child.pid
            (out/'processes.json').write_text(json.dumps(report,indent=2)+'\n')
            try:row['returncode']=child.wait(timeout=min(limit,remaining))
            except subprocess.TimeoutExpired:
                row['timed_out']=True;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                row['returncode']=child.returncode
            row['wall_s']=time.monotonic()-start
        (out/'processes.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
        if row['returncode']:raise RuntimeError('Cyclic architecture stage failed: '+label)
    with (a.contract.parent/'gpu.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        common=['--backend','native','--method','moving','--moving-depth','16','--ownership','preserve','--warmups','0','--samples','1']
        run('baseline',[PYTHON,baseline_worker,'--fixture',world,'--module',control,'--expected-component-sha',contract['control_sha256'],*common,'--output',out/'baseline.json'],900)
        run('candidate',[PYTHON,ROOT/'scripts/benchmark_cuda_cyclic_runtime.py','--fixture',a.fixture,'--module',module,'--expected-component-sha',frozen['sha256'],*common,'--output',out/'candidate.json'],900)
        run('compare',[PYTHON,ROOT/'scripts/compare_cuda_cyclic_runtime.py','--baseline',out/'baseline.json','--candidate',out/'candidate.json','--expected-control-sha256',contract['control_sha256'],'--expected-candidate-sha256',frozen['sha256'],'--output',out/'comparison.json'],900)
    if identities!={str(f):digest(f) for f in files}:raise ValueError('Benchmark or comparison dependencies changed during execution')
    report['dependency_digests_unchanged']=True
    (out/'processes.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
