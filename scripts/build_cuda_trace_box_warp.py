"""Bind the larger rectangular warp tile to every native physics class."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.build_cuda_trace_tile import stage as scaffold,replace_body
from scripts.cuda_trace_box_warp_storage import BoxWarpStorage
from scripts.cuda_trace_box_warp_codegen import generate
from scripts.cuda_trace_phase_cycle import block


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique box-warp binding required:'+old[:100])
    return text.replace(old,new,1)


def resource_metadata(extension,shared):
    extension,count=re.subn(r'(result\["moving_(?:source_pending|original)_shared_bytes"\]=)\d+',lambda m:m[1]+str(shared),extension)
    if count!=2:raise ValueError('Two real frame metadata fields required')
    return extension


def stage(source,output,expected,*,shared_field='Hx',late_addresses=False,volatile_addresses=False,private_z=4,private_y=4,cache=(16,16,32),provider_home=False,slot_major=False,group_publications=False,bounded_warp=False,terminal_epilogue=False,legacy_source_cpml=False):
    if shared_field=='none':shared_field=None
    if isinstance(shared_field,str) and ',' in shared_field:shared_field=tuple(shared_field.split(','))
    output=Path(output).resolve();t=BoxWarpStorage(tuple(cache),shared_field=shared_field,private_z=private_z,private_y=private_y,provider_home=provider_home,slot_major=slot_major)
    frame=t.shared_words*4+76+(t.threads//32*4 if volatile_addresses else 0)
    profile_slots=min(12,(101376-frame)//288)
    if profile_slots<1:raise ValueError('No room for this traced field/profile placement')
    t=BoxWarpStorage(tuple(cache),shared_field=shared_field,profile_slots=profile_slots,private_z=private_z,private_y=private_y,provider_home=provider_home,slot_major=slot_major)
    generated,proof=generate(t,late_addresses=late_addresses,volatile_addresses=volatile_addresses,group_publications=group_publications,bounded_warp=bounded_warp,terminal_epilogue=terminal_epilogue)
    # Reuse only the verified ABI/launch binding scaffold. Its temporary cache
    # is replaced before any build; its numerical generator is not executed.
    scaffold(source,output,expected,cache=(16,16,16),bulk_groups=2,base_groups=2)
    (output.parent/'tile-compiler-proof.json').rename(output.parent/'box-scaffold-binding.json')
    text=(output/'src/asymmetric_native.cuh').read_text()
    cz,cy,cx=t.cache
    text=once(text,'BZ=15,BY=15,BX=15,CZ=16,CY=16,CX=16',f'BZ={cz-1},BY={cy-1},BX={cx-1},CZ={cz},CY={cy},CX={cx}')
    start=text.index('constexpr size_t SharedBytes=');end=text.index('struct Source {',start)
    shared=proof['shared_bytes'];cached=proof['cached_shared_bytes']
    text=text[:start]+f'''constexpr size_t SharedBytes={shared},ProfileCachedSharedBytes={cached},FixedCubeSharedBytes={cached},TraceSourceFrameBytes={shared};
static_assert(BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE==0,"Generated rectangular warp storage owns indexing");
'''+text[end:]
    if terminal_epilogue:
        terminal_start=text.index(' if constexpr(ReuseTerminal){',text.index('FieldPacket ImportPacket('))
        terminal_end=text.index(' const float*point=p.trails',terminal_start)
        text=text[:terminal_start]+''' // Terminal H values have an explicit final trail row. E is dead here.
 if constexpr(ReuseTerminal){if(version==Depth)mask&=7u;}
'''+text[terminal_end:]
    if cached>101376:raise ValueError('Actual frame plus independent profiles exceeds opt-in block capacity')
    if t.profile_slots<12:
        text=once(text,'template<int Policy>constexpr size_t ResearchSourceSharedBytes=SourcePendingSharedBytes+(Policy==5?864ULL*sizeof(float):0);',
                  f'template<int Policy>constexpr size_t ResearchSourceSharedBytes=SourcePendingSharedBytes+(Policy==5?{t.profile_slots*72}ULL*sizeof(float):0);')
        text=once(text,'i<864;i+=blockDim.x',f'i<{t.profile_slots*72};i+=blockDim.x')
        text=once(text,'if constexpr(CachedProfiles)return ReadIndependentProfileCache(profiles,index,kind,packed);',
                  f'if constexpr(CachedProfiles){{if(index<{t.profile_slots})return ReadIndependentProfileCache(profiles,index,kind,packed);}}')
        text=once(text,'else{const auto&psi=p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}',
                  '{const auto&psi=p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}')
        text=once(text,'if(kind>=2)return ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x));',
                  f'if(kind>=2&&index<{t.profile_slots})return ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x));')
    text=replace_body(text,'void UniformBody(','''
 extern __shared__ float frame[];
 const int tile=tiles[blockIdx.x];
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env{p,flags,CachedProfiles?frame+SharedBytes/sizeof(float):nullptr};
 RunBoxWarp<Depth,Layout,CachedProfiles>(p,tiles,env);
''')
    text=replace_body(text,'void FixedCubeCachedBody(','''
 extern __shared__ float frame[];const auto flags=p.flags[tiles[blockIdx.x]];
 FixedCubeCachedEnvironment<N,ActiveAxes>env{p,flags,frame+SharedBytes/sizeof(float)};
 RunBoxWarp<Depth,1,true>(p,tiles,env);
''')
    text=replace_body(text,'void SourcePendingPairedBody(','''
 extern __shared__ float frame[];const auto flags=p.flags[tiles[blockIdx.x]];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>env{p,flags,CachedProfiles?frame+SharedBytes/sizeof(float):nullptr};
 RunBoxWarp<Depth,Layout,CachedProfiles>(p,tiles,env);
''')
    if legacy_source_cpml:
        if t.cache!=(8,16,32) or t.threads!=1024 or not terminal_epilogue or t.profile_slots!=12:
            raise ValueError('Legacy CPML source mapping requires original geometry,1024threads and terminal H trails')
        original=(Path(source)/'src/asymmetric_native.cuh').read_text()
        old=block(original,'void SourcePendingPairedBody(')[2]
        old=old[old.index('{')+1:-1]
        if old.count('if(surface>=0&&(Layout!=1||n<Depth))')!=2:
            raise ValueError('Two complete frozen source publication sites required')
        old=old.replace('if(surface>=0&&(Layout!=1||n<Depth))','if(surface>=0)')
        old=old.replace('p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;',
                        'if(Layout!=1||n<Depth)p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;')
        current=block(text,'void SourcePendingPairedBody(')[2]
        current=current[current.index('{')+1:-1]
        text=replace_body(text,'void SourcePendingPairedBody(','if constexpr(Policy==5){'+old+'}else{'+current+'}')
        text=once(text,'SourcePendingSharedBytes+(Policy==5?864ULL*sizeof(float):0)',
                  '(Policy==5?62136ULL:SourcePendingSharedBytes)')
    if any(a in text for a in ('TraceBulkGroups','TraceBaseGroups','TraceSourceGroups','TraceGenericSourceGroups','RunTraceTile<')):
        raise ValueError('Temporary topology escaped final binding')
    (output/'src/asymmetric_native.cuh').write_text(text)
    (output/'src/trace_tile_generated.cuh').write_text(generated)
    header=(output/'src/native_moving.h').read_text()
    for old,new in [('kOwnedZ=15,kOwnedY=15,kOwnedX=15',f'kOwnedZ={cz-1},kOwnedY={cy-1},kOwnedX={cx-1}'),
                    ('owned[3]{15,15,15}',f'owned[3]{{{cz-1},{cy-1},{cx-1}}}'),('owned{15,15,15}',f'owned{{{cz-1},{cy-1},{cx-1}}}')]:
        header=once(header,old,new)
    (output/'src/native_moving.h').write_text(header)
    extension=(output/'src/native_extension.cc').read_text()
    extension=once(extension,'module.attr("research_cache_shape")=nb::make_tuple(16,16,16);',
                   f'module.attr("research_cache_shape")=nb::make_tuple({cz},{cy},{cx});')
    extension=resource_metadata(extension,shared)
    if legacy_source_cpml:
        extension=extension.replace('result["moving_source_pending_shared_bytes"]='+str(shared),'result["moving_source_pending_shared_bytes"]=58680')
        extension=extension.replace('result["moving_source_pending_barriers"]=2;','result["moving_source_pending_barriers"]=5;')
        extension=extension.replace('module.attr("moving_source_pending_barriers")=2;','module.attr("moving_source_pending_barriers")=5;')
        extension=once(extension,'module.attr("moving_trace_generated")=true;',
                       'module.attr("moving_trace_generated")=true;module.attr("moving_source_cpml_legacy_fallback")=true;')
    for name in ('moving_noio_uniform_threads','moving_source_pending_threads','moving_source_plane_pair_threads'):
        extension=extension.replace(f'module.attr("{name}")=512;',f'module.attr("{name}")={t.threads};')
    proof.update(parent_consumer_sha256=expected,consumer_sha256=hashlib.sha256(text.encode()).hexdigest(),
                 extension_sha256=hashlib.sha256(extension.encode()).hexdigest(),host_header_sha256=hashlib.sha256(header.encode()).hexdigest(),
                 expected_kernels=68,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 legacy_source_cpml=legacy_source_cpml,barriers_per_step_by_class={'generated':2,'source_cpml':5 if legacy_source_cpml else 2},
                 complete_physics_scope=f'Native physics uses the generated{t.threads}-thread private Z/Y rectangle'+(' except Policy5source, which retains frozen paired storage with terminal H trails' if legacy_source_cpml else '')+'; physical observation bins remain8x16')
    (output/'src/native_extension.cc').write_text(extension)
    (output.parent/'box-warp-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--shared-field',default='Hx');p.add_argument('--late-addresses',action='store_true');p.add_argument('--volatile-addresses',action='store_true');p.add_argument('--private-z',type=int,default=4);p.add_argument('--private-y',type=int,default=4);p.add_argument('--cache',type=int,nargs=3,default=(16,16,32));p.add_argument('--provider-home',action='store_true');p.add_argument('--slot-major',action='store_true');p.add_argument('--group-publications',action='store_true');p.add_argument('--bounded-warp',action='store_true');p.add_argument('--terminal-epilogue',action='store_true');p.add_argument('--legacy-source-cpml',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,shared_field=a.shared_field,late_addresses=a.late_addresses,volatile_addresses=a.volatile_addresses,private_z=a.private_z,private_y=a.private_y,cache=tuple(a.cache),provider_home=a.provider_home,slot_major=a.slot_major,group_publications=a.group_publications,bounded_warp=a.bounded_warp,terminal_epilogue=a.terminal_epilogue,legacy_source_cpml=a.legacy_source_cpml)))
