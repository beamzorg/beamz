"""Exact temporal interval facts for a whole owned Yee box inside CPML12.

Only point math may use the fast class. Native windows, imports, low stages,
all four barriers, owner publication, raw observations and final scatter retain
their original semantics. This module does not select or run a GPU kernel.
"""

from __future__ import annotations


def interior_interval(minimum, origin, owned, depth):
    """Every owned point is valid, off walls, and all12 CPML guards are false."""
    low = max(
        1, *(o + b - n + 12 for o, b, n in zip(origin, owned, minimum, strict=True))
    )
    high = min(depth, *(o - 12 for o in origin))
    return low, high


INVENTORY_CPP = r"""
nb::dict NoIOStepInventory(const BeamzNativeMovingPlan&p,const std::vector<BeamzBuffer>&fields){
 using namespace beamz::cuda::moving;
 uint64_t tiles=0,live_tiles=0,window_steps=0,fast_steps=0,valid_fields=0,fast_fields=0,support_steps=0,support_fields=0,boundary_free_tiles=0;
 uint64_t by_mask_tiles[8]{},by_mask_windows[8]{},by_mask_fast[8]{};
 int minimum[3]={INT32_MAX,INT32_MAX,INT32_MAX};
 for(int c=0;c<6;++c)for(int a=0;a<3;++a)minimum[a]=std::min(minimum[a],int(fields[c].dims[a]));
 for(size_t tile=0;tile<p.tile_ids.size();++tile){
  if(Policy(p.tile_flags[tile])!=1)continue;++tiles;const int mask=(p.tile_flags[tile]&14)>>1;++by_mask_tiles[mask];
  int q[3]={int(tile/(p.counts[1]*p.counts[2])),int((tile/p.counts[2])%p.counts[1]),int(tile%p.counts[2])},origin[3];
  int first=1,last=p.depth,fast_first=1,fast_last=p.depth,support_first=1,support_last=p.depth;
  for(int a=0;a<3;++a){origin[a]=q[a]*p.owned[a];first=std::max(first,origin[a]-p.extent[a]+1);last=std::min(last,origin[a]+p.owned[a]-1);
   fast_first=std::max(fast_first,origin[a]+p.owned[a]-minimum[a]+12);fast_last=std::min(fast_last,origin[a]-12);
   support_first=std::max(support_first,origin[a]+p.owned[a]-minimum[a]);support_last=std::min(support_last,origin[a]-1);
  }
  if(!(p.tile_flags[tile]&kNativeBoundary))++boundary_free_tiles;
  if(first>last)continue;++live_tiles;const uint64_t count=last-first+1;
  window_steps+=count;by_mask_windows[mask]+=count;
  const int begin=std::max(first,fast_first),end=std::min(last,fast_last);const uint64_t fast=std::max(0,end-begin+1);
  fast_steps+=fast;by_mask_fast[mask]+=fast;fast_fields+=fast*6*p.owned[0]*p.owned[1]*p.owned[2];
  const uint64_t safe=std::max(0,std::min(last,support_last)-std::max(first,support_first)+1);support_steps+=safe;support_fields+=safe*6*p.owned[0]*p.owned[1]*p.owned[2];
  for(int n=first;n<=last;++n)for(int c=0;c<6;++c){uint64_t points=1;for(int a=0;a<3;++a){const int lo=std::max(origin[a]-n,0),hi=std::min(origin[a]+p.owned[a]-n,int(fields[c].dims[a]));points*=std::max(0,hi-lo);}valid_fields+=points;}
 }
 nb::dict r;r["noio_tiles"]=tiles;r["live_noio_tiles"]=live_tiles;r["window_steps"]=window_steps;r["interior_steps"]=fast_steps;
 r["noio_boundary_free_tiles"]=boundary_free_tiles;r["support_safe_steps"]=support_steps;r["support_safe_component_updates"]=support_fields;r["support_safe_window_fraction"]=window_steps?double(support_steps)/window_steps:0.;r["support_safe_component_fraction"]=valid_fields?double(support_fields)/valid_fields:0.;
 r["valid_component_updates"]=valid_fields;r["interior_component_updates"]=fast_fields;
 r["interior_window_fraction"]=window_steps?double(fast_steps)/window_steps:0.;r["interior_component_fraction"]=valid_fields?double(fast_fields)/valid_fields:0.;
 nb::list rows;for(int mask=0;mask<8;++mask){nb::dict row;row["axis_mask"]=mask;row["tiles"]=by_mask_tiles[mask];row["window_steps"]=by_mask_windows[mask];row["interior_steps"]=by_mask_fast[mask];rows.append(row);}r["by_axis_mask"]=rows;return r;
}
"""
