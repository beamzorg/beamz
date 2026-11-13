// Exhaust the independent high-boundary residues of native7x15x31 ownership.
// Stronger than terminalH alone: imports may contain ANY of the6 Yee fields.
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
using Point=std::array<int,3>;
static constexpr Point B{7,15,31};
static bool active(Point q,Point n,int step,int depth){
 int first=1,last=depth;
 for(int a=0;a<3;++a){first=std::max(first,q[a]*B[a]-n[a]);last=std::min(last,q[a]*B[a]+B[a]-1);}
 return first<=step&&step<=last;
}
static bool has_packet(Point q,Point n,int mask,int step){
 bool below_high=false;
 for(int a=0;a<3;++a){const bool ghost=mask&(1<<a);const int origin=q[a]*B[a];
  int lo=ghost?origin-step-1:origin-step,hi=ghost?lo:origin-step+B[a]-1;
  lo=std::max(lo,0);hi=std::min(hi,n[a]);if(lo>hi)return false;below_high|=lo<n[a];
 }
 // Union of Ex/Ey/Ez support is the inclusive cube except its(Nz,Ny,Nx)
 // corner. Each H support is contained in this union.
 return below_high;
}
static bool reachable(Point q,Point n,int step,int depth,int from,int target){
 if(from==target)return true;Point node=q;
 for(int a=0;a<3;++a)node[a]-=bool(from&(1<<a));
 if(!active(node,n,step,depth))return false;
 for(int a=0;a<3;++a)if((target&(1<<a))&&!(from&(1<<a)))if(reachable(q,n,step,depth,from|(1<<a),target))return true;
 return false;
}
int main(){
 size_t domains=0,active_steps=0,packet_dependencies=0;
 for(int rz=0;rz<7;++rz)for(int ry=0;ry<15;++ry)for(int rx=0;rx<31;++rx)
 for(int depth:{4,16,32}){
  // Full independent residues, with enough physicalextent to include both
  // low-side startup and clipped high-side windows in the same small domain.
  Point n{28+rz,30+ry,31+rx},counts;
  for(int a=0;a<3;++a)counts[a]=(n[a]+1+depth+B[a]-1)/B[a];
  for(int z=0;z<counts[0];++z)for(int y=0;y<counts[1];++y)for(int x=0;x<counts[2];++x){Point q{z,y,x};
   for(int step=1;step<=depth;++step){if(!active(q,n,step,depth))continue;++active_steps;
    for(int target=1;target<8;++target){bool exists=true;for(int a=0;a<3;++a)if((target&(1<<a))&&!q[a])exists=false;
     if(!exists||!has_packet(q,n,target,step))continue;bool reached=false;
     for(int a=0;a<3;++a)if(target&(1<<a))reached|=reachable(q,n,step,depth,1<<a,target);
     if(!reached)throw std::runtime_error("Nearest3 failed on an active full-step ghost packet");++packet_dependencies;
    }
   }
  }
  ++domains;
 }
 const Point q{4,2,1},n{15,17,47};const int step=12,target=7;
 bool reachable3=false;for(int a=0;a<3;++a)reachable3|=reachable(q,n,step,16,1<<a,target);
 if(!has_packet(q,n,target,step)||reachable3)throw std::runtime_error("Late clipped-corner negative ineffective");
 std::cout<<"{\"passed\":true,\"domains\":"<<domains<<",\"active_steps\":"<<active_steps<<",\"packet_dependencies\":"<<packet_dependencies<<",\"nearest3_late_startup_rejected\":true,\"scope\":\"Complete7x15x31 independent boundaryresidues, depths4/16/32, full6field supportunion and fixed-step readinesscausality. All7late-startup directwait required. CUDAmemoryorder separate\"}\n";
}
