"""Bind individual traced field reads to proven tile storage expressions.

Every call receives a typed versioned read, not a runtime field ID or neighbor
selector. Shared-provider conditions already implied by the route are erased.
The generic native cell functor remains unchanged for independent comparison.
"""
from __future__ import annotations
from dataclasses import dataclass

from scripts.cuda_space_time_ir import IllegalSchedule
from scripts.cuda_trace_tile_topology import TileTopology,allocate_providers
from scripts.generate_cuda_native_cell import generate_native_cell
from scripts.cuda_trace_phase_cycle import block


@dataclass(frozen=True)
class TileReadBinding:
    cache: tuple[int,int,int]=(16,16,16)
    shared_e: bool=False

    def regions(self,field):
        return [(i,r) for i,r in enumerate(allocate_providers(TileTopology(self.cache,2))[0]) if r.field==field]

    def shared_y(self,field,z,y,x):
        region=next((i,r) for i,r in self.regions(field) if r.axis==1)
        i,_=region;_,cy,cx=self.cache
        return f'frame[Storage::O{i}+(({z})*({cy}/Rows)+({y})/Rows)*{cx}+({x})]'

    def shared_z(self,field,z,y,x):
        cz,cy,cx=self.cache;r=32//cx
        regions=self.regions(field);zregion=next(((i,q) for i,q in regions if q.axis==0),None)
        yregion=next(((i,q) for i,q in regions if q.axis==1),None)
        if zregion is None:
            if r==1 and yregion:return self.shared_y(field,z,y,x)
            raise IllegalSchedule('No traced Z provider')
        i,region=zregion
        if region.excluded_y_residue is None:
            return f'frame[Storage::O{i}+((({z})/Slots)*{cy}+({y}))*{cx}+({x})]'
        electric=field.startswith('E')
        packed=f'(({y})-({y})/Rows'+('-1)' if electric else ')')
        other=f'frame[Storage::O{i}+((({z})/Slots)*({cy}-{cy}/Rows)+{packed})*{cx}+({x})]'
        primary=self.shared_y(field,z,y,x)
        predicate=f'({y})%Rows=='+('0' if electric else 'Rows-1')
        return f'(({predicate})?({primary}):({other}))'

    def __call__(self,output,read):
        phase=int(output.startswith('E'));source_phase=int(read.field.startswith('E'))
        component='xyz'.index(read.field[-1]);dz,dy,dx=read.offset
        if read.version!=(0 if source_phase else phase):
            raise IllegalSchedule('Field version incompatible with H-before-E tile schedule')
        if source_phase==phase and any(read.offset):
            raise IllegalSchedule('Same-family neighbor requires a different stage schedule')
        if self.shared_e and source_phase:
            return f'frame[Storage::Words/2+{component}*PAD+Slot(zs+({dz}),sy+({dy}),sx+({dx}))]'
        family='e' if source_phase else 'h'
        if not any(read.offset):return f'{family}[{component}][j]'
        if sum(abs(d) for d in read.offset)!=1:raise IllegalSchedule('Nearest axial read required')
        if dx:
            if component not in (1,2):raise IllegalSchedule('Unexpected normal-X neighbor')
            return f'wx{component}'
        if dy:
            if component not in (0,2):raise IllegalSchedule('Unexpected normal-Y neighbor')
            shared=self.shared_y(read.field,'zs',f'((sy+({dy}))&(CY-1))','sx')
            if self.cache[2]==32:return shared
            internal='sy%Rows!=Rows-1' if dy>0 else 'sy%Rows!=0'
            return f'(({internal})?wy{component}:({shared}))'
        z=f'((zs+({dz}))&(CZ-1))'
        shared=self.shared_z(read.field,z,'sy','sx')
        return f'((j+({dz})>=0&&j+({dz})<Slots)?{family}[{component}][j+({dz})]:({shared}))'


def inline_point_section(code,electric,cache=(16,16,16),*,shared_e=False):
    """Replace a compiler-owned component event with traced scalar SSA blocks."""
    binding=TileReadBinding(cache,shared_e)
    blocks=generate_native_cell(read_binding=binding)['point_blocks']
    start=code.index('    const auto read=')
    end=code.index('    #pragma unroll',start)
    code=code[:start]+code[end:]
    start,end,_=block(code,'for(int c=0;c<3;++c)')
    start=code.rfind('    #pragma unroll',0,start)
    family='E' if electric else 'H';parts=[]
    for c in range(3):
        field=family+'xyz'[c]
        target=(f'frame[Storage::Words/2+{c}*PAD+Slot(zs,sy,sx)]' if electric and shared_e
                else f'{family.lower()}[{c}][j]')
        parts+=['    {float value=0.f;',f'     if(env.Valid({3*int(electric)+c},z,y,x))'+'{',
                blocks[field],'     value=result;}',f'     {target}=value;','    }']
    return code[:start]+'\n'.join(parts)+code[end:]
