"""Bind generated diagonal register/shared/IO routes to the version model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.build_cuda_diagonal_logical import component_region
from scripts.cuda_space_time_ir import FIELDS,yee_operator
from scripts.cuda_trace_phase_cycle import block


def validate(candidate,parent,model):
    candidate,parent=Path(candidate).resolve(),Path(parent).resolve();src=candidate/'source/src';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proof=json.loads((candidate/'diagonal-logical-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();old=(parent/'src/asymmetric_native.cuh').read_text();body=(src/'native_diagonal_logical.cuh').read_text()
    if sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256'] or sha(src/'native_extension.cc')!=proof['extension_sha256']:raise ValueError('Changed bound diagonal consumer')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Changed diagonal helper:'+name)
    for name in ('native_moving.cc','native_moving.h','asymmetric_native_cell_generated.cuh','yee_cell_generated.cuh','yee_primitives.cuh'):
        if (src/name).read_bytes()!=(parent/'src'/name).read_bytes():raise ValueError('Changed numerical/owner/certificate source:'+name)
    for signature in ('struct Environment {','struct FixedCubeCachedEnvironment:','float Finish(','FieldPacket ImportPacket(','void Observe(','void FixedCubeCachedBody(','void SourcePendingPairedBody(','inline cudaError_t LaunchNoIO(','inline cudaError_t LaunchSourcePending('):
        if block(c,signature)[2]!=block(old,signature)[2]:raise ValueError('Changed frozen fallback/physics/dispatch:'+signature)
    legacy=block(c,'void DiagonalLegacyUniformBody(')[2].replace('void DiagonalLegacyUniformBody(','void UniformBody(',1)
    if legacy!=block(old,'void UniformBody(')[2] or legacy!=proof['legacy_uniform_body']:raise ValueError('Frozen uniform fallback inverse failed')
    if block(c,'void UniformBody(')[2]!=proof['uniform_wrapper']:raise ValueError('Changed certified new scope')
    m=json.loads(Path(model).read_text())
    if not m.get('passed') or m['source_sha256']!=sha(Path(__file__).with_name('prove_cuda_diagonal_private_cells.py')) or len(m['negative_controls'])!=7:raise ValueError('Current full version/WAR model required')
    regions=[[],[]]
    for b in proof['component_bindings']:
        emitted,binding=component_region(b['phase'],b['component'])
        if emitted!=b['emitted'] or json.loads(json.dumps(binding))!={k:v for k,v in b.items() if k!='emitted'} or body.count(emitted)!=1:raise ValueError('Changed canonical generated read routes')
        regions[b['phase']].append(emitted)
    loop=block(body,'for(int n=first;n<=last;++n)')[2]
    if loop.count('__syncthreads();')!=4:raise ValueError('Four shared retirement/publication joins required')
    functions=[]
    for signature,prefix in (('DiagonalPoint DiagonalComputePoint(','template<int T>'),('int DiagonalSlot(',''),('int DiagonalBoundarySlot(',''),('float DiagonalShuffle(',''),('DiagonalPoint DiagonalGhostLogical(',''),('DiagonalPoint DiagonalSurfaceLogical(',''),('void DiagonalStoreFamily(','template<int T>')):
        functions.append(prefix+block(body,signature)[2])
    functions.insert(0,'struct DiagonalPoint{int z,y,x;};');functions.append(block(old,'int Surface(')[2])
    records=[]
    for u in yee_operator(cpml_axes=()).updates:
        if u.output.field not in FIELDS:continue
        f=FIELDS.index(u.output.field)
        for r in sorted(u.reads):
            if r.field in FIELDS:records.append((f//3,f,FIELDS.index(r.field),r.version,*r.offset))
    reads='static const Read reads[]={'+','.join('{'+','.join(map(str,row))+'}' for row in records)+'};'
    code=Path(__file__).with_name('prove_cuda_diagonal_storage_host.cc').read_text().replace('// ACTUAL_HELPERS','\n'.join(functions)).replace('// CANONICAL_READS',reads).replace('// ACTUAL_H_REGIONS','\n'.join(regions[0])).replace('// ACTUAL_E_REGIONS','\n'.join(regions[1]))
    out=candidate/'host-diagonal-logical-proof';out.mkdir(exist_ok=False);rows=[]
    for name in ('actual','wrong-shuffle-lane','wrong-Z-boundary','conflicted-shared-banks'):
        folder=out/name;folder.mkdir();text=code
        if name=='wrong-shuffle-lane':text=text.replace('(sx+31)&31','(sx+1)&31')
        if name=='wrong-Z-boundary':text=text.replace('DiagonalBoundarySlot(sy-1,sx-1)','DiagonalBoundarySlot(sy,sx)')
        if name=='conflicted-shared-banks':text=text.replace('((x+2*y+z)&31)','((x+y+z)&31)')
        cc=folder/'proof.cc';cc.write_text(text);binary=folder/'proof';built=subprocess.run(['g++','-std=c++20','-O2','-fsanitize=address,undefined',str(cc),'-o',str(binary)],capture_output=True,text=True);(folder/'build.log').write_text(built.stdout+built.stderr)
        if built.returncode:raise RuntimeError('Actual diagonal proof compile failed:'+str(folder/'build.log'))
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180);(folder/'run.log').write_text(run.stdout+run.stderr)
        if (run.returncode!=0 if name=='actual' else run.returncode==0):raise RuntimeError('Unexpected diagonal proof:'+name+run.stdout+run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr[-1000:]))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],model_source_sha256=m['source_sha256'],canonical_generated_regions=True,exact_frozen_fallback_inverse=True,actual_host_helpers=rows,scope='Actual emitted private/shared/shuffle/boundary operands, geometry/inverse, ghost/owned publication, native surface IO and bank addresses. Host warp shuffle is token based; version/WAR interleaving separately modeled. CUDA memory visibility and full FP32 simulation require GPU checks.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate,a.parent,a.model)))
