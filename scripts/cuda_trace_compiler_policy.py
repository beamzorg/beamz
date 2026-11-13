"""Isolate backend compiler storage/scheduling policies on frozen CUDA sources.

Only the non-RDC moving translation unit receives the selected option. The
ordinary reference, source trace, floating-point options and host runtime retain
their original settings. Resource gates and actual timings remain required.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha


def stage(source,output,policy):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied research source required')
    consumer=(source/'src/asymmetric_native.cuh').read_bytes()
    if sha(consumer)!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    flags={'reg0':'-Xptxas=--register-usage-level=0','reg2':'-Xptxas=--register-usage-level=2',
           'reg8':'-Xptxas=--register-usage-level=8','reg10':'-Xptxas=--register-usage-level=10',
           'cheap':'-Xptxas=--allow-expensive-optimizations=false','ptxO2':'-Xptxas=-O2',
           'vector':'--extra-device-vectorization'}
    if policy not in flags:raise ValueError('Explicit compiler policy required')
    cmake=(source/'CMakeLists.txt').read_text()
    cmake+='\nif(TARGET beamz_cuda_moving)\n  target_compile_options(beamz_cuda_moving PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:'+flags[policy]+'>)\nendif()\n'
    shutil.copytree(source,output);(output/'CMakeLists.txt').write_text(cmake)
    proof=dict(schema='beamz.moving-backend-compiler-policy.v1',policy=policy,moving_flag=flags[policy],
               scalar_consumer_sha256=PARENT_CONSUMER,cmake_sha256=sha(cmake.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               scope='Only moving TU compiler; no ordinary-reference/FP-mode/frontend/host/field changes',expected_kernels=66,GPU_executed=False)
    (output.parent/'compiler-policy-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',choices=('reg0','reg2','reg8','reg10','cheap','ptxO2','vector'),required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.policy)))
