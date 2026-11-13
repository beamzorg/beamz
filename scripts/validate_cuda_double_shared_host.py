"""Audit two shared in-place phases and actual a/b profile/geometry lowerings."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.build_cuda_phase_rebinding import H_OLD,E_OLD
from scripts.cuda_trace_phase_cycle import block
from scripts.validate_cuda_phase_rebinding_host import PROVIDERS,run_cpp


def validate(candidate,model):
    candidate=Path(candidate).resolve();src=candidate/'source/src';proof=json.loads((candidate/'double-shared-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not proof.get('double_shared') or proof.get('double_barriers_per_step')!=2 or proof.get('double_shared_bytes')!=100960 or sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256']:raise ValueError('Bound two-phase shared compiler required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Changed double-family helper:'+name)
    model=json.loads(Path(model).read_text())
    if not model.get('passed') or model['source_sha256']!=sha(Path(__file__).with_name('prove_cuda_double_shared.py')) or len(model['negative_controls'])!=4 or not all(c['same_family_center_only_updates']==6 and c['persistent_private_field_values']==0 for c in model['cases']):raise ValueError('Actual canonical in-place dependency/version proof required')
    c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_double_shared.cuh').read_text();geometry=(src/'native_double_geometry.cuh').read_text();native=block(c,'void UniformBody(')[2];position=0
    for phase,r in enumerate(proof['double_region_bindings']):
        marker='{int step;asm volatile("mov.b32 %0,%1;":';a=native.index(marker,position);_,b,region=block(native[a:],marker);position=a+b
        if region!=r['original']:raise ValueError('Actual canonical H/E loop ancestry required')
        expected=once(region,H_OLD if phase==0 else E_OLD,'return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];')
        expected=once(expected,'h[c][j]=value;' if phase==0 else 'e[c][j]=value;',f'frame[(c+{phase*3})*PAD+Slot(zs,sy,sx)]=value;');expected=re.sub(r'\bSlot\(','PhaseSlot(',expected)
        generated='\n'.join(('if(!(flags&62u))' if i==0 else 'else if(!(flags&46u))' if i==1 else 'else')+'{'+f'DoubleSharedEnvironment<CubeN,{policy},{uniform}>env{{p,flags,profiles}};'+expected[1:-1]+'}' for i,(policy,uniform) in enumerate(((2,'false'),(4,'false'),(5,'true'))))
        if expected!=r['provider_lowered'] or generated!=r['emitted'] or body.count(generated)!=1:raise ValueError('Only canonical shared providers/output placement may change')
    loop=block(body,'for(int n=first;n<=last;++n)')[2]
    if loop.count('__syncthreads();')!=1 or loop.count('closed_pipeline::Publish(n);')!=1:raise ValueError('One H join plus original release-publication join required')
    if re.search(r'float (?:h|e|state)\[3\]\[Slots\]',body):raise ValueError('No persistent private field arrays allowed')
    out=candidate/'host-double-shared-proof';out.mkdir(exist_ok=False)
    constants='\n'.join(line for line in c.splitlines() if line.startswith('constexpr int BZ=') or line.startswith('constexpr int PAD='))
    functions=block(c,'int Surface(')[2]+'\n'+block(geometry,'int PhaseSlot(')[2]+'\n'+block(geometry,'struct PhasePoint{')[2]+';\nconstexpr int PhaseGhostPoints=PAD-BZ*BY*BX;\n'
    functions+='\n'.join(('template<int T>\n' if 'Private' in s else '')+block(geometry,s)[2] for s in ('PhasePoint PhaseGhostLogical(','PhasePoint PhasePrivateLogical(','PhasePoint PhaseGhostPhysicalSlot('))
    cold=block((src/'native_closed_pipeline_control.cuh').read_text(),'struct Cold {')[2]+';'
    provider=re.sub(r'\bSlot\(','PhaseSlot(',PROVIDERS).replace('c<3','c<6').replace('3*PAD','6*PAD').replace('49280','100736').replace('49504','100960').replace('single_bank_slots','shared_family_slots')
    provider=provider.replace('6*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)<=100736','6*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)+576*sizeof(float)<=100736')
    provider=provider.replace('// ACTUAL_DEFINITIONS',constants+'\n'+functions+'\n'+cold);providers=run_cpp(out,'actual-providers',provider,src)
    primitives=(src/'yee_primitives.cuh').read_text();parts=[]
    for signature in ('constexpr int CpmlAxis(','float ScaleYeeDifference(','float PackedMaterialSource('):parts.append(('template<class Index>\n' if 'PackedMaterial' in signature else '')+block(primitives,signature)[2])
    environment=block(c,'struct Environment {')[2]+';';a=c.index('struct Environment {');environment=c[c.rfind('template<',0,a):a]+environment
    fixed=block(c,'struct FixedCubeCachedEnvironment:')[2]+';';a=c.index('struct FixedCubeCachedEnvironment:');fixed=c[c.rfind('template<',0,a):a]+fixed
    profiles='\n'.join(block(c,s)[2] for s in ('int IndependentProfileCacheIndex(','float ReadIndependentProfileCache('))
    stage='struct CPUThreadIndex{int x=0;};static CPUThreadIndex threadIdx;\ntemplate<int T>\n'+block(body,'void DoubleStageProfiles(')[2]
    methods=Path(__file__).with_name('prove_cuda_effect_regions_host.cc').read_text().replace('// BEAMZ_EFFECT_CANONICAL_PRIMITIVES','\n'.join(parts)).replace('// BEAMZ_EFFECT_ACTUAL_ENVIRONMENTS',profiles+'\n'+environment+'\n'+fixed+'\n'+stage)
    physics=run_cpp(out,'actual-physics-methods',methods,src,moving=True,defines=['-DBEAMZ_DOUBLE_SHARED_ENGINE=1'])
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],canonical_region_inverse=True,same_family_center_only_model=True,actual_providers=providers,actual_physics_methods=physics,model_sha256=model['source_sha256'],scope='Actual shared providers/ghost layout, independent a/b staging with overrun guards, actual physics methods and canonical H/E region inverse, in-place/version/auxiliary model. CUDA shared visibility/full30 FP32 parity/performance remain separate.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate,a.model)))
