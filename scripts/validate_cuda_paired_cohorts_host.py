"""Compile actual paired planner with independent positive/negative lifetime audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def validate(source,header,output):
    src=Path(source).resolve();header=Path(header).resolve();out=Path(output).resolve();out.mkdir(exist_ok=False);rows=[]
    actual=header.read_text();driver=Path(__file__).with_name('prove_cuda_paired_cohorts_host.cc').read_text()
    for name in ('actual','missing-reader-completion','missing-upper-ownership','missing-pending-completion'):
        folder=out/name;folder.mkdir();code=actual
        if name=='missing-reader-completion':code=code.replace('readers.insert(old+dz*plane+dy*nx+dx);','(void)old;')
        if name=='missing-upper-ownership':code=code.replace('for(int t:p.tile_ids){','for(int t:p.tile_ids){if((t/plane)&1)continue;')
        if name=='missing-pending-completion':code=code.replace('++q.initial_pending[s];++q.finish_edges;','++q.finish_edges;')
        (folder/'native_paired_cohorts.h').write_text(code);cc=folder/'proof.cc';cc.write_text(driver);binary=folder/'proof'
        command=['g++','-std=c++20','-O2','-fsanitize=address,undefined','-I'+str(folder),'-I'+str(src),str(cc),str(src/'native_moving.cc'),'-o',str(binary)]
        build=subprocess.run(command,capture_output=True,text=True);(folder/'build.log').write_text(build.stdout+build.stderr)
        if build.returncode:raise RuntimeError('Paired CPU planner compile failed:'+str(folder/'build.log'))
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180);(folder/'run.log').write_text(run.stdout+run.stderr)
        if (run.returncode!=0 if name=='actual' else run.returncode==0):raise RuntimeError('Unexpected paired lifetime audit:'+name+run.stdout+run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr[-1500:]))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result=dict(passed=True,paired_planner_sha256=sha(header),native_planner_sha256=sha(src/'native_moving.cc'),native_header_sha256=sha(src/'native_moving.h'),closed_planner_sha256=sha(src/'native_closed_cohorts.h'),driver_sha256=sha(Path(__file__).with_name('prove_cuda_paired_cohorts_host.cc')),rows=rows)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--header',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.source,a.header,a.output)))
