"""Reuse terminal field outputs as the final cross-tile communication cut.

The traced finite-horizon graph has no E_depth consumers among evolving state.
H_depth consumers are current E/psi updates. Existing terminal field stores
already publish exactly those sourced H values; later topological ranks can
read the final field allocation instead of a duplicate trail publication.
Raw monitor staging and its ordered DFT are unchanged. This changes copied
research sources only and requires subsequent numerical/GPU validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from scripts.cuda_space_time_ir import FIELDS, yee_operator
from scripts.cuda_trace_phase_cycle import ROOT, block, sha
from scripts.cuda_trace_column_schedule import PARENT, PARENT_SHA


def terminal_cut(depth, *, operator=None, consumer_horizon=None):
    if type(depth) is not int or depth < 2:
        raise ValueError('Temporal horizon must contain at least two versions')
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    consumers = {field:set() for field in FIELDS}
    horizon=depth if consumer_horizon is None else consumer_horizon
    if type(horizon) is not int or horizon<depth:
        raise ValueError('Consumer horizon cannot end before the terminal version')
    for version in range(1, horizon+1):
        for update in operator.updates:
            for read in update.reads:
                if version-1+read.version == depth and read.field in consumers:
                    consumers[read.field].add(update.output.field)
    if any(consumers[field] for field in FIELDS if field.startswith('E')):
        raise ValueError('Terminal E has a live evolving-state consumer')
    if any(not consumers[field] for field in FIELDS if field.startswith('H')):
        raise ValueError('Expected terminal H consumers absent')
    return dict(depth=depth, terminal_consumers={k:sorted(v) for k,v in consumers.items()},
                terminal_H_provider='existing unique FinalField H store; upstream rank completes first',
                terminal_E_provider='terminal output and raw observer only; no ghost import',
                duplicate_trail_writes_removed='all six fields at terminal version only',
                required_invariants=['Canonical H-before-E graph', 'No deferred reader of terminal E trails',
                    'Every valid terminal H point has exactly one final output writer',
                    'Kernel/rank dependencies publish final output before downstream import',
                    'Retain raw observations and chronological DFT staging'])


def transform(original, *, layout1_only=True):
    text = original
    if layout1_only:
        old='template<int Depth,bool Fast>\n__device__ __forceinline__ FieldPacket ImportPacket('
        new='template<int Depth,bool Fast,bool ReuseTerminal=true>\n__device__ __forceinline__ FieldPacket ImportPacket('
        if text.count(old)!=1:
            raise ValueError('Exact packet template declaration required')
        text=text.replace(old,new)
    start, end, imported = block(text, 'FieldPacket ImportPacket(')
    anchor = ' const float*point=p.trails+'
    if imported.count(anchor) != 1:
        raise ValueError('Exact packet read site required')
    terminal = '''
 if(version==Depth){
  const int z=oz-1+dz-version,y=oy-1+dy-version,x=ox-1+dx-version;
  const unsigned terminal_mask=NativeSupportMask(p,z,y,x)&7u;
  const auto&a=FinalField(p,0);const auto&b=FinalField(p,1);const auto&c=FinalField(p,2);
  return {terminal_mask&1?__ldcg(a.data+z*a.plane+y*a.row+x):0.f,
          terminal_mask&2?__ldcg(b.data+z*b.plane+y*b.row+x):0.f,
          terminal_mask&4?__ldcg(c.data+z*c.plane+y*c.row+x):0.f,0.f,0.f,0.f};
 }
'''
    if layout1_only:
        terminal=' if constexpr(ReuseTerminal){\n'+terminal+' }\n'
    imported = imported.replace(anchor, terminal+anchor)
    text = text[:start]+imported+text[end:]
    sites = text.count('if(surface>=0){')
    if sites < 4:
        raise ValueError('Complete native/source publication sites required')
    text = text.replace('if(surface>=0){', 'if(surface>=0&&(Layout!=1||n<Depth)){' if layout1_only else 'if(surface>=0&&n<Depth){')
    if layout1_only:
        # Layout is uniform for a complete native invocation. Preserve the old
        # communication cut for Layout2, whose register budget is already full.
        old='template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageRegisterLow('
        text=text.replace(old,'template<int Depth,bool Safe,int T,bool ReuseTerminal=true>\n__device__ __forceinline__ void StageRegisterLow(')
        begin,end,low=block(text,'void StageRegisterLow(')
        low=low.replace('ImportPacket<Depth,Safe>', 'ImportPacket<Depth,Safe,ReuseTerminal>')
        text=text[:begin]+low+text[end:]
        for signature in ('void UniformBody(','void FixedCubeCachedBody('):
            begin,end,body=block(text,signature)
            body=body.replace('ImportPacket<Depth,false>', 'ImportPacket<Depth,false,(Layout==1)>')
            body=body.replace('ImportPacket<Depth,SupportSafe>', 'ImportPacket<Depth,SupportSafe,(Layout==1)>')
            body=body.replace('StageRegisterLow<Depth,SupportSafe,T>', 'StageRegisterLow<Depth,SupportSafe,T,(Layout==1)>')
            text=text[:begin]+body+text[end:]
    return text, sites


def stage(source, output, expected):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output.exists() or output == ROOT/'cuda' or ROOT/'cuda' in output.parents
            or source in output.parents or output in source.parents):
        raise ValueError('New disjoint copied source required')
    original = (source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode()) != expected:
        raise ValueError('Explicit immutable parent hash required')
    text, sites = transform(original)
    shutil.copytree(source, output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    proof = dict(schema='beamz.trace-terminal-final-reuse.v1',
                 horizons=[terminal_cut(d) for d in (4,16)], publication_sites=sites,
                 parent_source=str(source), parent_consumer_sha256=expected,
                 consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
                 rank_dependencies_unchanged=True, scalar_FP_source_observer_DFT_unchanged=True,
                 admission='Invocation-uniform Layout1; Layout2 retains original terminal trails/imports',
                 expected_kernels=66, GPU_executed=False)
    (output.parent/'terminal-reuse-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PARENT)
    p.add_argument('--expected-consumer-sha256',default=PARENT_SHA)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
