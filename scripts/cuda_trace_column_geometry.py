"""Retarget the traced column hierarchy to another admitted power-of-two cache.

The old shape-specific source pipeline is replaced by the existing generic
source-capable CUDA path, so whole-workload validation is mandatory. This is a
separate candidate, not an implicit change to any installed backend.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

from scripts.cuda_trace_phase_cycle import block, sha
from scripts.cuda_trace_box_search import stage as stage_box
from scripts.cuda_trace_systolic_compiler import (ThreadMap,compile_column,prove_routes,
    prove_compact_providers,prove_z_providers,prove_deduplicated_z)


def stage(source,output,cache,z_groups,*,min_blocks=1):
    source,output=Path(source).resolve(),Path(output).resolve()
    parent=json.loads((source.parent/'column-schedule-proof.json').read_text())
    if not parent.get('deduplicated_frontiers') or parent['z_groups']!=2:
        raise ValueError('A compact deduplicated two-group parent is required')
    mapping=ThreadMap(cache=tuple(cache),threads=cache[1]*32*z_groups,z_groups=z_groups)
    mapping.validate()
    proof=stage_box(source,output,tuple(cache),512,expected_consumer=parent['consumer_sha256'])
    text=(output/'src/asymmetric_native.cuh').read_text()
    text=text.replace('ColumnZGroups=2;',f'ColumnZGroups={z_groups};')
    text=text.replace('constexpr int T=1024,Slots=CZ/ColumnZGroups,ZStep=1;',
                      f'constexpr int T={mapping.threads},Slots=CZ/ColumnZGroups,ZStep=1;')
    # Layout2's packed-material fallback has longer address lifetimes. Give
    # its Fast instantiations one XY plane instead of imposing the main path's
    # register cap; its actual mapping and launch must change together.
    text=text.replace('constexpr int T=ResearchThreads,Slots=PAD/T,ZStep=T/(CY*CX);',
                      'constexpr int T=(Fast&&Layout==2)?(CY*CX):ResearchThreads,Slots=PAD/T,ZStep=T/(CY*CX);')
    # StageBox changes the fallback mapping; restore the independently selected
    # column mapping only in entrypoints which actually call the generated body.
    column_entries=[]
    for signature in ('void AsymmetricNoIOFixedCube(','void AsymmetricNoIOUniform1024('):
        start,end,entry=block(text,signature)
        if 'TraceColumnBody<' not in entry:
            continue
        before=text[:start]
        bounds=list(re.finditer(r'__launch_bounds__\([^\n]*?\)',before))[-1]
        before=before[:bounds.start()]+f'__launch_bounds__({mapping.threads},{min_blocks})'+before[bounds.end():]
        text=before+entry+text[end:]
        column_entries.append(signature)
    if column_entries:
        start,end,launch=block(text,'inline cudaError_t LaunchNoIO(')
        launch=re.sub(r'<<<uniform_count,(?:1024|ResearchThreads),ColumnSharedBytes,',
                      f'<<<uniform_count,{mapping.threads},ColumnSharedBytes,',launch)
        text=text[:start]+launch+text[end:]
    start,end,entry=block(text,'void Asymmetric(')
    if 'TraceColumnBody<' in entry:
        before=text[:start]
        bounds=list(re.finditer(r'__launch_bounds__\([^\n]*?\)',before))[-1]
        # The fallback source/observer kernels retain their original512 budget.
        eligible='(Fast&&Layout==1&&IsotropicH)'
        fallback='((Fast&&Layout==2)?(CY*CX):ResearchThreads)'
        before=before[:bounds.start()]+f'__launch_bounds__(({eligible}?{mapping.threads}:{fallback}),({eligible}?{min_blocks}:1))'+before[bounds.end():]
        text=before+entry+text[end:]
        start,end,launch=block(text,'inline cudaError_t Launch(')
        launch=launch.replace('<<<count,ResearchThreads,ResearchSharedBytes,',
                              f'<<<count,({eligible}?{mapping.threads}:{fallback}),ResearchSharedBytes,')
        launch=launch.replace('ResearchSharedBytes=Fast?ColumnBaseBytes:SharedBytes',
                              f'ResearchSharedBytes={eligible}?ColumnBaseBytes:SharedBytes')
        text=text[:start]+launch+text[end:]
    # Physical coarse observation masks must never follow the moving cache.
    _,_,old_observe=block((source/'src/asymmetric_native.cuh').read_text(),'void Observe(')
    _,_,new_observe=block(text,'void Observe(')
    if old_observe!=new_observe:
        raise ValueError('Observation masks changed with cache geometry')
    (output/'src/asymmetric_native.cuh').write_text(text)
    owned=[n-1 for n in cache]
    y_words=owned[0]*owned[1]*owned[2]
    z_family=z_groups*owned[1]*owned[2]+owned[1]*owned[2]+z_groups*owned[2]
    field_banks=4 if parent['schedule']['separate_families'] else 2
    z_families=2 if field_banks==4 else 1
    shared=(field_banks*y_words+z_families*z_family)*4+76+(3456 if parent['cache_profiles'] else 0)
    result=dict(parent, schema='beamz.trace-column-geometry.v1',parent_source=str(source),
                parent_consumer_sha256=parent['consumer_sha256'],consumer_sha256=sha(text.encode()),
                producer_sha256=sha(Path(__file__).read_bytes()),schedule=compile_column(mapping=mapping,
                    separate_families=field_banks==4),routing_proof=prove_routes(mapping),
                compact_provider_proof=prove_compact_providers(mapping),
                z_provider_proof=prove_z_providers(mapping),deduplication_proof=prove_deduplicated_z(mapping),
                column_shared_bytes=shared,z_groups=z_groups,min_blocks=min_blocks,
                fallback_threads=512,source_paired_storage_disabled=True,
                source_observer_fallback_paths_unchanged=False,expected_kernels=56,GPU_executed=False)
    (output.parent/'column-schedule-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=int,nargs=3,required=True)
    p.add_argument('--z-groups',type=int,required=True)
    p.add_argument('--min-blocks',type=int,default=1)
    a=p.parse_args()
    print(json.dumps(stage(a.source,a.output,tuple(a.cache),a.z_groups,min_blocks=a.min_blocks)))
