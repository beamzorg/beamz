"""Validate current emitted cyclic source facts and ordered source arithmetic."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.patch_cuda_v12_coincident_source_geometry import ORIGINAL
from scripts.patch_cuda_cyclic_source import FACT10
from scripts.validate_cuda_cyclic_source import HARNESS


def validate(candidate):
    candidate=Path(candidate).resolve();source=candidate/'source'
    header=(source/'src/native_moving.h').read_text();consumer=(source/'src/asymmetric_native.cuh').read_text()
    if header.count(FACT10)!=1:raise ValueError('Actual mask10 topology predicate missing')
    start=consumer.index('    if constexpr(PlanePair){');end=consumer.index('\n   }\n   if constexpr(HasObserve)',start)
    callback=consumer[start:end]
    if callback.count(ORIGINAL)!=1 or 'if constexpr(SourceMask==10)' not in callback:raise ValueError('Ordered generic source oracle or cyclic callback changed')
    harness=HARNESS.replace('int SourceMask=10>', 'int SourceMask=10,int Policy=5>')
    # Policy5 explicitly covers points outside the transverse source support;
    # the generic fallback and specialized plane callback must both skip them.
    harness=harness.replace('x=12+random_word()%(nx-24);', 'x=12+random_word()%(nx-24);if(sample%4==0)y=11;if(sample%4==1)x=nx-12;')
    program=harness.replace('@ORIGINAL@',ORIGINAL).replace('@CALLBACK@',callback)
    directory=candidate/'host-cyclic-source-proof';directory.mkdir(exist_ok=False)
    rows=[]
    # Only alter the specialized branch's product order, preserving its two
    # coefficient/waveform pairings. Generic reference remains independent.
    needle0='value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+offset),__ldg(s.waveforms+time)));'
    needle1='value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+s.ny*s.nx+offset),__ldg(s.waveforms+s.ntime+time)));'
    if callback.count(needle0)<1 or callback.count(needle1)!=1:raise ValueError('Two ordered cyclic additions missing')
    wrong=callback.replace(needle0,'@FIRST_CYCLIC_ADD@',1).replace(needle1,needle0,1).replace('@FIRST_CYCLIC_ADD@',needle1,1)
    for name,code in (('actual',program),('wrong-add-order',harness.replace('@ORIGINAL@',ORIGINAL).replace('@CALLBACK@',wrong))):
        path=directory/(name+'.cc');binary=directory/name;path.write_text(code)
        command=['c++','-std=c++20','-O2','-ffp-contract=off','-fno-fast-math','-I'+str(source/'src'),str(path),'-o',str(binary)]
        subprocess.run(command,capture_output=True,text=True,check=True)
        run=subprocess.run([str(binary)],capture_output=True,text=True);result=json.loads(run.stdout)
        if (result['errors']==0)!=(name=='actual'):raise RuntimeError('Cyclic callback check or negative witness failed:'+run.stdout)
        rows.append(dict(name=name,returncode=run.returncode,result=result,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    proof=dict(passed=True,rows=rows,actual_consumer_sha256=hashlib.sha256(consumer.encode()).hexdigest(),
               scope='Actual emitted mask10 host topology and scalar RN source callback, clipped transverse support, independent generic oracle and addition-order negative; CUDA validation separate')
    (directory/'result.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args();print(json.dumps(validate(a.candidate)))
