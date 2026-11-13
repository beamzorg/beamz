"""CPU-only actual planner/material certificate binding for a frozen probe.

No native DSO import or GPU use. Reconstructs the exact worker packet, then
compiles the actual host planner, material certificate, compactions and deep
observation builder against the selected source headers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prove(fixture_path, worker_path, source, output):
    spec = importlib.util.spec_from_file_location("frozen_positive_probe", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    fixture, data, original = worker.load_fixture(fixture_path)
    packet, attrs = worker.make_packet(original, fixture["attributes"], "direct-source-DFT", 17, True)
    output.mkdir(parents=True, exist_ok=True)
    minimum = [min(a.shape[d] for a in packet[:6]) for d in range(3)]
    scalar_h = all(packet[i].shape == () and packet[i].dtype == np.float32 for i in range(6, 12))
    hbits = [int(packet[i].view(np.uint32)) for i in range(6, 12)] if scalar_h else []
    isotropic = scalar_h and hbits[:3] == [0x3F800000] * 3 and len(set(hbits[3:])) == 1 and all(np.isfinite(packet[i]) for i in range(6, 12))
    indices = [37, 38, 39, 40, 41, 42, 113] + [88 + 3 * g for g in range(9)]
    for i in indices:
        (output / f"input-{i}.bin").write_bytes(np.ascontiguousarray(packet[i]).tobytes())
    declarations = []
    for i, array in enumerate(packet):
        kind = {np.dtype("float32"): "kBeamzF32", np.dtype("int32"): "kBeamzS32"}.get(array.dtype)
        if kind is None or array.ndim > 4:
            raise ValueError("This admitted FP32 probe requires ordinary F32/S32 arrays")
        declarations.append(f"in[{i}].rank={array.ndim};in[{i}].element_type={kind};")
        for a, n in enumerate(array.shape):
            declarations.append(f"in[{i}].dims[{a}]={n};")
        if array.ndim == 3:
            declarations.append(f"in[{i}].row_stride={array.shape[2]};in[{i}].plane_stride={array.shape[1]*array.shape[2]};")
    loads = []
    for i in indices:
        ctype = "float" if packet[i].dtype == np.float32 else "int32_t"
        loads.append(f"auto v{i}=load<{ctype}>({json.dumps(str((output/f'input-{i}.bin').resolve()))});in[{i}].data=v{i}.data();")
    cpp = r'''#include "native_moving.h"
#include "native_deep.h"
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <vector>
template<class T>std::vector<T>load(const char*path){std::ifstream f(path,std::ios::binary|std::ios::ate);auto n=f.tellg();if(n<0||size_t(n)%sizeof(T))throw std::runtime_error("inputbytes");std::vector<T>v(size_t(n)/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),n);return v;}
int main(){constexpr int depth=16;BeamzBuffer in[134]{};for(auto&b:in)for(int a=0;a<4;++a)b.dims[a]=1;
@DECLARATIONS@
@LOADS@
BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};uint32_t presence=0;
for(int g=0;g<9;++g){groups[g].coefficients=in[86+3*g];groups[g].waveforms=in[87+3*g];groups[g].starts=in[88+3*g];groups[g].coincident=(@COINCIDENT@>>g)&1;groups[g].disjoint=(@DISJOINT@>>g)&1;starts[g]=static_cast<const int32_t*>(in[88+3*g].data);if(g>=3&&in[86+3*g].dims[0])presence|=1u<<(g-3);}
auto obs=BeamzBuildNativeDeepPlan(in,in[113],static_cast<const int32_t*>(in[113].data));std::vector<std::array<int,3>>points;int component=0;
for(int64_t key:obs.keys){while(component<5&&key>=obs.field_base[component+1])++component;const int64_t local=key-obs.field_base[component];const auto&f=in[component];points.push_back({int(local/(f.dims[1]*f.dims[2])),int((local/f.dims[2])%f.dims[1]),int(local%f.dims[2])});}
auto p=BeamzBuildNativeMovingPlan(in,depth,points,groups,starts,12,true);
const void*tables[3]={in[37].data,in[38].data,in[39].data},*codes[3]={in[40].data,in[41].data,in[42].data};
auto cert=BeamzCertifyNativeMovingUniformE(in,in+37,in+40,tables,codes,p);constexpr bool isotropic=@ISOTROPIC@;
int certified=0;for(size_t tile=0;tile<p.tile_flags.size();++tile){if(!cert.supported||!cert.admitted[tile])continue;uint8_t flags=p.tile_flags[tile];if(!flags){p.tile_flags[tile]|=64;continue;}int policy=beamz::cuda::moving::Policy(flags);if(!isotropic||(policy!=1&&policy!=2))continue;
int q[3]={int(tile/(p.counts[1]*p.counts[2])),int((tile/p.counts[2])%p.counts[1]),int(tile%p.counts[2])};bool productive=false;
for(int c=0;c<3;++c){int64_t first=1,last=depth;const auto&f=in[3+c];for(int a=0;a<3;++a){int64_t origin=int64_t(q[a])*p.owned[a];first=std::max(first,origin-f.dims[a]+1);last=std::min(last,origin+p.owned[a]-1);}productive|=first<=last;}
if(productive){p.tile_flags[tile]|=64;if(policy==1)++certified;}}
BeamzCompactNativeMovingUniformNoIO(p,isotropic);BeamzCompactNativeMovingUniformObservers(p,isotropic);
long long onehot[3]{},active[2][3]{},policies[6]{};int witness[3][8]{};for(auto&w:witness)w[0]=-1;int counted=0;
for(size_t rank=0;rank<p.bulk_counts.size();++rank){for(int policy=0;policy<6;++policy)policies[policy]+=p.policy_offsets[7*rank+policy+1]-p.policy_offsets[7*rank+policy];
int first=p.policy_offsets[7*rank+1],count=p.uniform_noio_counts[rank];counted+=count;
for(int i=0;i<count;++i){int tile=p.tile_ids[first+i];uint8_t flags=p.tile_flags[tile];int mask=flags&14;if(beamz::cuda::moving::Policy(flags)!=1||!(flags&64))throw std::runtime_error("badcertprefix");if(mask!=2&&mask!=4&&mask!=8)continue;int axis=mask==2?0:mask==4?1:2;++onehot[axis];
int q[3]={tile/(p.counts[1]*p.counts[2]),(tile/p.counts[2])%p.counts[1],tile%p.counts[2]};constexpr int axes[6]={1,0,0,2,2,1};
for(int family=0;family<2;++family){bool hit=false;for(int term=0;term<6;++term){if(axes[term]!=axis)continue;const auto&f=in[3*family+term/2];for(int side=0;side<2;++side){int64_t begin=1,end=depth,lo[3]{},hi[3]{},origin[3]{};
for(int a=0;a<3;++a){origin[a]=q[a]*p.owned[a];hi[a]=f.dims[a];if(a==axis){if(side)lo[a]=f.dims[a]-12;else hi[a]=12;}begin=std::max(begin,origin[a]-hi[a]+1);end=std::min(end,origin[a]+p.owned[a]-lo[a]-1);}
if(begin<=end){hit=true;if(witness[axis][0]<0){witness[axis][0]=tile;witness[axis][1]=family;witness[axis][2]=term;witness[axis][3]=int(begin);for(int a=0;a<3;++a)witness[axis][4+a]=int(std::max(lo[a],origin[a]-begin));witness[axis][7]=flags;}}}}
active[family][axis]+=hit;}
}}
if(counted!=certified)throw std::runtime_error("certcountmismatch");int minimum[3]={@MINIMUM@};bool plane=BeamzNativeSourcePlanePairFact(groups,9,starts,minimum,presence);
printf("{\"certified_noio\":%d,\"onehot_noio_zyx\":[%lld,%lld,%lld],\"active_H_zyx\":[%lld,%lld,%lld],\"active_E_zyx\":[%lld,%lld,%lld],\"source_plane_fact\":%s,\"source_policies34\":[%lld,%lld],\"policy_counts\":[%lld,%lld,%lld,%lld,%lld,%lld],\"observation_keys\":%zu,\"witnesses\":[",certified,onehot[0],onehot[1],onehot[2],active[0][0],active[0][1],active[0][2],active[1][0],active[1][1],active[1][2],plane?"true":"false",policies[3],policies[4],policies[0],policies[1],policies[2],policies[3],policies[4],policies[5],obs.keys.size());
for(int a=0;a<3;++a){if(a)printf(",");printf("[");for(int j=0;j<8;++j){if(j)printf(",");printf("%d",witness[a][j]);}printf("]");}printf("]}");return 0;}
'''.replace("@DECLARATIONS@", "\n".join(declarations)).replace("@LOADS@", "\n".join(loads)).replace("@ISOTROPIC@", "true" if isotropic else "false").replace("@MINIMUM@", ",".join(map(str, minimum))).replace("@COINCIDENT@",str(attrs["coincident_source_group_mask"])).replace("@DISJOINT@",str(attrs["disjoint_source_group_mask"]))
    cfile, binary = output / "positive-plan.cc", output / "positive-plan"
    cfile.write_text(cpp)
    command = ["c++", "-std=c++17", "-O2", "-I" + str(source / "src"), str(cfile), str(source / "src/native_moving.cc"), str(source / "src/native_deep.cc"), "-o", str(binary)]
    build = subprocess.run(command, text=True, capture_output=True)
    if build.returncode:
        raise RuntimeError(build.stderr)
    result = json.loads(subprocess.check_output([str(binary)], text=True))
    result.update({"fixture_sha256": sha(fixture_path), "worker_sha256": sha(worker_path),
                   "reconstruction": "make_packet(inputs, attrs, 'direct-source-DFT',17,True)",
                   "shape_zyx": minimum, "attributes": attrs, "input_hashes": [worker.digest_array(a) for a in packet],
                   "host_isotropic_class": bool(isotropic),
                   "source_hashes": {n: sha(source / "src" / n) for n in ("native_moving.cc", "native_moving.h", "native_deep.cc", "native_extension.cc")},
                   "oracle_cpp_sha256": sha(cfile), "compile_command": command,
                   "scope": "Actual host planner/material certificate/deep observation builder/compactions plus exact copied E-active admission; productive scalar recurrence witnesses, not GPU execution or timings"})
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = prove(a.fixture, a.worker, a.source, a.output)
    print(json.dumps({k: r[k] for k in ("shape_zyx", "certified_noio", "onehot_noio_zyx", "active_H_zyx", "active_E_zyx", "source_plane_fact", "source_policies34")}))


if __name__ == "__main__":
    main()
