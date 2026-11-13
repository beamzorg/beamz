"""Freeze an explicitly bounded-spill diamond trial without changing strict gates.

Only the newly introduced diamond bulk/face kernels may spill. Existing kernels
must retain zero stack/local/spills. This is research admission, not production
selection, and preserves the original strict-gate failure as evidence.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from scripts.check_cuda_moving_build import is_moving_kernel,ninja_contract

CUOBJDUMP='/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/cuobjdump'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allowed(name):return bool(re.search(r'moving_native\d+AsymmetricDiamond(?:UniformBulk|NoIOFace)',name))


def admit(source_candidate,output,expected=82,stack_budget=512,spill_budget=8192):
    source_candidate,output=Path(source_candidate).resolve(),Path(output).resolve()
    if output.exists() or stack_budget!=512 or spill_budget!=8192:raise ValueError('New explicit bounded-spill trial required')
    staged=json.loads((source_candidate/'staged-source.json').read_text())
    for name,sha in staged['source_hashes'].items():
        if digest(source_candidate/'source'/name)!=sha:raise ValueError('Compiled source changed:'+name)
    build=source_candidate/'build';module=build/'_cuda_native.cpython-311-x86_64-linux-gnu.so'
    compile_contract=ninja_contract((build/'build.ninja').read_text(),(build/'CMakeFiles/rules.ninja').read_text())
    usage=subprocess.run([CUOBJDUMP,'--dump-resource-usage',str(module)],capture_output=True,text=True,check=True).stdout
    resources=[]
    for name,fields in re.findall(r' Function ([^\n]+):\n([^\n]+)',usage):
        if not is_moving_kernel(name):continue
        values={k:int(v) for k,v in re.findall(r'\b(REG|STACK|SHARED|LOCAL):(\d+)',fields)}
        if set(values)!={'REG','STACK','SHARED','LOCAL'}:raise ValueError('Incomplete resource row')
        if values['LOCAL'] or values['STACK']>(stack_budget if allowed(name) else 0):raise ValueError('Unapproved stack/local resources:'+name)
        resources.append(dict(name=name,**values))
    ptxas=[];name=''
    for line in (source_candidate/'build.log').read_text().splitlines():
        if 'Compiling entry function' in line:name=line.split("'")[1]
        match=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',line)
        if match and is_moving_kernel(name):
            row=dict(name=name,stack=int(match[1]),spill_stores=int(match[2]),spill_loads=int(match[3]))
            if row['stack']>(stack_budget if allowed(name) else 0) or any(row[k]>(spill_budget if allowed(name) else 0) for k in ('spill_stores','spill_loads')):
                raise ValueError('Unapproved ptxas spill resources:'+name)
            ptxas.append(row)
    if any(len(rows)!=expected or len({r['name'] for r in rows})!=expected for rows in (resources,ptxas)):
        raise ValueError('Incomplete exact moving kernel inventory')
    if not any(r['STACK'] for r in resources):raise ValueError('Use ordinary strict admission for a zero-stack candidate')
    output.mkdir();sha=digest(module);target=output.parent/'frozen-execution'/output.name/sha/module.name;target.parent.mkdir(parents=True,exist_ok=False);shutil.copy2(module,target)
    if digest(target)!=sha:raise ValueError('Frozen copy identity mismatch')
    report=dict(schema='beamz.exploratory-diamond-resource-admission.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                strict_gate_rejected=True,strict_gate_rejection_retained=str(source_candidate),source_candidate=str(source_candidate),
                source_hashes=staged['source_hashes'],source_manifest_sha256=digest(source_candidate/'staged-source.json'),
                module=str(target),sha256=sha,kernel_count=expected,all_moving_zero_stack_local_spill=False,
                moving_exploratory_stack_budget_bytes=stack_budget,static_spill_load_store_budget_bytes=spill_budget,
                allowed_spill_kernels='Only AsymmetricDiamondUniformBulk and AsymmetricDiamondNoIOFace',
                resources=resources,ptxas=ptxas,compile_contract=compile_contract,gpu_executed=False,
                scope='Bounded-spill architectural trial; complete physics and measured performance required; not a production selection',producer_sha256=digest(__file__))
    (output/'exploratory-admission.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest={k:report[k] for k in ('module','sha256','kernel_count','all_moving_zero_stack_local_spill','moving_exploratory_stack_budget_bytes','gpu_executed','scope')}
    (output/'frozen-execution-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(admit(a.source_candidate,a.output)))
