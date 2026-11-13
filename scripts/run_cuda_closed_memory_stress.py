"""Execute one immutable standalone stress under the research GPU contract."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stress',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--tool',choices=('memcheck','racecheck','initcheck','synccheck'));p.add_argument('--kernel-filter');p.add_argument('--launch-count',type=int);a=p.parse_args();folder=a.stress.resolve();candidate=a.candidate.resolve()
    if a.launch_count is not None and a.launch_count<1:raise ValueError('Positive sanitizer launch count required')
    manifest=json.loads((folder/'manifest.json').read_text());contract=json.loads(a.contract.read_text());host=json.loads((candidate/'host-closed-pipeline-proof/result.json').read_text());binary=Path(manifest['binary'])
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(binary)!=manifest['sha256'] or not host['passed'] or host['helpers']!=manifest['helpers']:raise ValueError('Immutable binary and actual CPU helper gate required')
    log=(folder/'build.log').read_text();rows=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    if len(rows)!=manifest.get('expected_stress_entries',3) or any(int(v) for row in rows for v in row):raise ValueError('Exact zero-stack/spill stress entries required')
    deadline=datetime.datetime.fromisoformat(contract['gpu_deadline_utc'].replace('Z','+00:00')).timestamp()
    with (a.contract.resolve().parent/'gpu.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);remaining=deadline-time.time()
        if remaining<180:raise RuntimeError('GPU deadline too near for standalone stress')
        command=[str(binary)]
        if a.tool:command=['/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/compute-sanitizer/compute-sanitizer','--tool',a.tool,'--error-exitcode','86',*(['--kernel-name',a.kernel_filter] if a.kernel_filter else []),*(['--launch-count',str(a.launch_count)] if a.launch_count else []),*command]
        start=time.monotonic();run=subprocess.run(command,capture_output=True,text=True,timeout=min(180,remaining));wall=time.monotonic()-start
    stem='execution'+('-'+a.tool if a.tool else '')
    (folder/(stem+'.stdout')).write_text(run.stdout);(folder/(stem+'.stderr')).write_text(run.stderr)
    report=dict(binary_sha256=sha(binary),helpers=manifest['helpers'],returncode=run.returncode,wall_s=wall,GPU_executed=True,
                scope='Synthetic packet visibility/retirement/queue stress; not FDTD numerical parity or throughput')
    report['command']=command;report['sanitizer']=a.tool
    if run.returncode==0:report['result']=json.loads(next(line for line in run.stdout.splitlines() if line.startswith('{')))
    (folder/(stem+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if run.returncode or not report.get('result',{}).get('passed'):raise RuntimeError('Standalone GPU stress failed')
