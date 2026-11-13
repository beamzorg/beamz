"""Freeze a compiled research module under explicit, bounded spill budgets.

This does not relax the existing strict build gate. It is a separate opt-in
experiment contract; numerical validation and timing remain mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.check_cuda_moving_build import is_moving_kernel, ninja_contract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(candidate, expected, budget):
    candidate=Path(candidate).resolve()
    staged=json.loads((candidate/'staged-source.json').read_text())
    if any(sha(candidate/'source'/name)!=value for name,value in staged['source_hashes'].items()):
        raise ValueError('Staged sources changed during compilation')
    build=candidate/'build'
    module=build/'_cuda_native.cpython-311-x86_64-linux-gnu.so'
    contract=ninja_contract((build/'build.ninja').read_text(),(build/'CMakeFiles/rules.ninja').read_text())
    toolkit=Path('/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/cuobjdump')
    resource=subprocess.check_output([str(toolkit),'--dump-resource-usage',str(module)],text=True)
    rows=[]
    for name,fields in re.findall(r' Function ([^\n]+):\n([^\n]+)',resource):
        if not is_moving_kernel(name):continue
        values={k:int(v) for k,v in re.findall(r'\b(REG|STACK|SHARED|LOCAL):(\d+)',fields)}
        if set(values)!={'REG','STACK','SHARED','LOCAL'}:raise ValueError('Incomplete resource row')
        rows.append(dict(name=name,**values))
    ptxas=[];name=''
    for line in (candidate/'build.log').read_text().splitlines():
        if 'Compiling entry function' in line:name=line.split("'")[1]
        match=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',line)
        if match and is_moving_kernel(name):
            ptxas.append(dict(name=name,stack=int(match[1]),spill_stores=int(match[2]),spill_loads=int(match[3])))
    if any(len(v)!=expected or len({r['name'] for r in v})!=expected for v in (rows,ptxas)):
        raise ValueError(f'Kernel count expected{expected}, actual{len(rows)}/{len(ptxas)}')
    if budget<0 or budget>256:raise ValueError('Explicit exploratory budget must be0..256 bytes')
    failures=[r for r in rows if r['LOCAL']>budget or r['STACK']>budget]
    failures += [r for r in ptxas if max(r[k] for k in ('stack','spill_stores','spill_loads'))>budget]
    record=dict(schema='beamz.exploratory-all-moving-resources.v1',module=str(module),module_sha256=sha(module),
                kernel_count=expected,compile_contract=contract,kernels=rows,ptxas=ptxas,
                explicit_stack_local_static_spill_budget_bytes=budget,passed=not failures,
                failures=failures,gpu_executed=False,scope='EXPLORATORY; resource admission is not correctness or performance acceptance')
    (candidate/'exploratory-all-moving-contract.json').write_text(json.dumps(record,indent=2)+'\n')
    if failures:raise ValueError(f'Explicit budget exceeded in{len(failures)}records')
    frozen=candidate.parent/'frozen-execution'/candidate.name/sha(module)/module.name
    frozen.parent.mkdir(parents=True,exist_ok=True)
    with frozen.open('xb') as f:f.write(module.read_bytes())
    frozen.chmod(0o444)
    manifest=dict(module=str(frozen),sha256=sha(frozen),kernel_count=expected,
                  all_moving_zero_stack_local_spill=not any(r[k] for r in ptxas for k in ('stack','spill_stores','spill_loads')),
                  explicit_all_moving_budget_bytes=budget,resource_contract_sha256=sha(candidate/'exploratory-all-moving-contract.json'),
                  source_hashes=staged['source_hashes'],gpu_executed=False)
    (candidate/'frozen-execution-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_hashes'}),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--expected-kernels',type=int,required=True)
    p.add_argument('--budget',type=int,required=True)
    a=p.parse_args();freeze(a.candidate,a.expected_kernels,a.budget)
