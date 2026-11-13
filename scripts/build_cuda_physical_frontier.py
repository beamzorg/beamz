"""Lower moving-tile cross-task operands onto the physical working field bank.

Research staging only. Keep the original trail allocation/metadata for the first
trial, but never access its payload. A full field copy at each macro start is
deliberately included in solver timing; allocation/alias changes are separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_phase_cycle import block

PARENT_SHA = "a7104fb39695c2a3d8df0bac3db20d647957851e89c655a81125f601742d25c0"


def stage(source, output, *, in_place=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError("New disjoint source required")
    original=(source/"src/asymmetric_native.cuh").read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=PARENT_SHA:
        raise ValueError("Frozen bounded-terminal-cache parent required")
    text=original
    old="const auto&f=InitialField(p,field);return Contains(f,z,y,x)?f.data[z*f.plane+y*f.row+x]:0.f;"
    new="const auto&f=FinalField(p,field);return Contains(f,z,y,x)?__ldcg(f.data+z*f.plane+y*f.row+x):0.f;"
    if text.count(old)!=1:raise ValueError("Reviewed startup read changed")
    text=text.replace(old,new,1)
    start,end,_=block(text,"FieldPacket ImportPacket(")
    text=text[:start]+'''FieldPacket ImportPacket(const Parameters&p,const size_t*bases,int version,
 int oz,int oy,int ox,int dz,int dy,int dx){
 (void)bases;
 const int z=oz-1+dz-version,y=oy-1+dy-version,x=ox-1+dx-version;
 unsigned mask=Fast?63u:NativeSupportMask(p,z,y,x);
 if constexpr(ReuseTerminal)if(version==Depth)mask&=7u;
 const auto&a=FinalField(p,0);const auto&b=FinalField(p,1);const auto&c=FinalField(p,2);
 const auto&d=FinalField(p,3);const auto&e=FinalField(p,4);const auto&f=FinalField(p,5);
 return {mask&1?__ldcg(a.data+z*a.plane+y*a.row+x):0.f,
         mask&2?__ldcg(b.data+z*b.plane+y*b.row+x):0.f,
         mask&4?__ldcg(c.data+z*c.plane+y*c.row+x):0.f,
         mask&8?__ldcg(d.data+z*d.plane+y*d.row+x):0.f,
         mask&16?__ldcg(e.data+z*e.plane+y*e.row+x):0.f,
         mask&32?__ldcg(f.data+z*f.plane+y*f.row+x):0.f};
}'''+text[end:]
    # Retain each original body and numerical computation; replace its payload
    # publication only. Paired source bodies have two distinct local E packets.
    replaced=0
    while "if(surface>=0&&(Layout!=1||n<Depth))" in text:
        begin,finish,_=block(text,"if(surface>=0&&(Layout!=1||n<Depth))")
        text=text[:begin]+text[finish:]
        anchor="if(n==Depth)"
        start=text.index(anchor,begin)
        text=text[:start]+"if(surface>=0||n==Depth)"+text[start+len(anchor):]
        replaced+=1
    if replaced!=4 or "p.trails[" in text:raise ValueError("Incomplete physical publication lowering")
    anchor=" BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;"
    if text.count(anchor)!=1:raise ValueError("Reviewed enqueue anchor changed")
    initialization=''' // Initialize the mutable physical frontier from the immutable macro input.
 // These copies are captured with the solve graph and included in its timing.
 for(int field=0;field<6;++field){
  const auto&a=InitialField(p,field);const auto&b=FinalField(p,field);
  if(a.z!=b.z||a.y!=b.y||a.x!=b.x||a.row!=a.x||b.row!=b.x||a.plane!=a.y*a.x||b.plane!=b.y*b.x)return cudaErrorInvalidValue;
  const auto copy=cudaMemcpyAsync(b.data,a.data,size_t(a.z)*a.y*a.x*sizeof(float),cudaMemcpyDeviceToDevice,stream);
  if(copy!=cudaSuccess)return copy;
 }
'''
    if in_place:
        initialization=''' // Internal NativeSession working state is the physical time frontier.
 // The other allocated bank remains disjoint and unused in this first trial.
 for(int field=0;field<6;++field)p.phase[field/3].output[field%3]=InitialField(p,field);
'''
    text=text.replace(anchor,initialization+anchor,1)
    shutil.copytree(source,output)
    (output/"src/asymmetric_native.cuh").write_text(text)
    if in_place:
        extension=(output/"src/native_extension.cc").read_text()
        begin,finish,body=block(extension,"nb::dict RunMoving(")
        if body.count("std::swap(current_,other_);")!=2:
            raise ValueError("Reviewed macro bank accounting changed")
        parity="if((count/depth)&1)std::swap(current_,other_);"
        if body.count(parity)!=1:raise ValueError("Reviewed graph replay parity changed")
        body=body.replace(parity,"/* Physical frontier needs no graph-replay bank parity change. */",1)
        body=body.replace("std::swap(current_,other_);","/* Physical frontier remains in current_; no descriptor bank swap. */")
        body=body.replace('"native-asymmetric-moving-v1"','"native-asymmetric-physical-frontier-v1"')
        extension=extension[:begin]+body+extension[finish:]
        # The marker exposes the architectural contract to research harnesses.
        marker="NB_MODULE(_cuda_native, module) {"
        if extension.count(marker)!=1:raise ValueError("Module entry changed")
        extension=extension.replace(marker,marker+'\n module.attr("moving_physical_frontier_inplace")=true;',1)
        (output/"src/native_extension.cc").write_text(extension)
    proof=dict(schema="beamz.physical-frontier-lowering.v1",parent_consumer_sha256=PARENT_SHA,
               consumer_sha256=hashlib.sha256(text.encode()).hexdigest(),
               publication_bodies=replaced,trail_payload_accesses=0,
               bootstrap="Internal working input is the physical frontier" if in_place else "Macro input copied to output frontier; coherent initial reads from frontier",
               lifetime="Outgoing upper-surface values and final owned volume published at original points in tile program order; full rank joins unchanged",
               native_allocation_unchanged=True,macro_copy_inside_timing=not in_place,
               internal_in_place_frontier=in_place,caller_input_capture_and_preservation_unchanged=True,
               requires="Physical-frontier RAW/WAR proof plus complete GPU validation before benchmarking",
               generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output.parent/"physical-frontier-compiler-proof.json").write_text(json.dumps(proof,indent=2)+"\n")
    return proof


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--in-place",action="store_true");a=p.parse_args()
    print(json.dumps(stage(a.source,a.output,in_place=a.in_place)))
