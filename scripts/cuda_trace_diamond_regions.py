"""Physical bounds and exact integer feasibility for retimed diamond owners."""
from __future__ import annotations


def physical_bounds(origin,owned,depth,bias_y,retiming=None,*,orientation='xy'):
    if orientation=='yz':
        lo,hi=physical_bounds((origin[2],origin[0],origin[1]),(owned[2],owned[0],owned[1]),depth,bias_y,retiming)
        return (lo[2],lo[1],lo[0]),(hi[2],hi[1],hi[0])
    if orientation!='xy':raise ValueError('Unknown diamond orientation')
    oz,op,om=origin;bz,bp,bm=owned
    rmin,rmax=(0,1) if retiming is None else (retiming,retiming)
    low=(oz-depth,(op-om+1-bm+bias_y+rmin)//2,
         (op+om-2*depth-bias_y-rmax)//2)
    high=(oz+bz-2,(op-om+bp-1+bias_y+rmax+1)//2,
          (op+om+bp+bm-4-bias_y-rmin+1)//2)
    return low,high


def has_update(origin,owned,depth,bias_y,shape,retiming,*,orientation='xy'):
    if orientation=='yz':
        return has_update((origin[2],origin[0],origin[1]),(owned[2],owned[0],owned[1]),depth,bias_y,(shape[2],shape[1],shape[0]),retiming)
    if orientation!='xy':raise ValueError('Unknown diamond orientation')
    """Exact integer feasibility by eliminating y from its three lower/upper bounds."""
    oz,op,om=origin;bz,bp,bm=owned;nz,ny,nx=shape
    for n in range(max(1,oz-nz+1),min(depth,oz+bz-1)+1):
        pl,ph=op-n,op+bp-1-n
        ml,mh=om-bias_y-retiming-n,om+bm-1-bias_y-retiming-n
        xlow=max(0,ml,pl-(ny-1),(pl+ml+1)//2)
        xhigh=min(nx-1,ph,mh+ny-1,(ph+mh)//2)
        if xlow<=xhigh:return True
    return False


def cpp_helpers(orientation='xy'):
    source= '''
inline int64_t BeamzDiamondFloorHalf(int64_t a){return a/2-((a<0)&&(a%2));}
inline int64_t BeamzDiamondCeilHalf(int64_t a){return a/2+((a>0)&&(a%2));}
inline void BeamzNativeDiamondBounds(const int*o,const int*b,int depth,int bias,int64_t*lo,int64_t*hi,int r=-1){
 const int rmin=r<0?0:r,rmax=r<0?1:r;
 lo[0]=int64_t(o[0])-depth;hi[0]=int64_t(o[0])+b[0]-2;
 lo[1]=BeamzDiamondFloorHalf(int64_t(o[1])-o[2]+1-b[2]+bias+rmin);
 hi[1]=BeamzDiamondCeilHalf(int64_t(o[1])-o[2]+b[1]-1+bias+rmax);
 lo[2]=BeamzDiamondFloorHalf(int64_t(o[1])+o[2]-2*depth-bias-rmax);
 hi[2]=BeamzDiamondCeilHalf(int64_t(o[1])+o[2]+b[1]+b[2]-4-bias-rmin);
}
inline bool BeamzNativeDiamondHasUpdate(const int*o,const int*b,int depth,int bias,const int*shape,int r){
 int first=1,last=depth;
 if(o[0]-shape[0]+1>first)first=o[0]-shape[0]+1;
 if(o[0]+b[0]-1<last)last=o[0]+b[0]-1;
 for(int n=first;n<=last;++n){
  const int64_t pl=int64_t(o[1])-n,ph=int64_t(o[1])+b[1]-1-n;
  const int64_t ml=int64_t(o[2])-bias-r-n,mh=int64_t(o[2])+b[2]-1-bias-r-n;
  int64_t low=0,high=shape[2]-1,value;
  if(ml>low)low=ml;if((value=pl-(shape[1]-1))>low)low=value;
  if((value=BeamzDiamondCeilHalf(pl+ml))>low)low=value;
  if(ph<high)high=ph;if((value=mh+shape[1]-1)<high)high=value;
  if((value=BeamzDiamondFloorHalf(ph+mh))<high)high=value;
  if(low<=high)return true;
 }
 return false;
}
'''

    if orientation=='yz':
        import re
        logical={0:2,1:0,2:1};physical={0:2,1:1,2:0}
        source=re.sub(r'\b(o|b|shape|lo|hi)\[([012])\]',
                      lambda m:f'{m[1]}[{(logical if m[1] in ("o","b") else physical)[int(m[2])]}]',source)
    elif orientation!='xy':raise ValueError('Unknown diamond orientation')
    return source
