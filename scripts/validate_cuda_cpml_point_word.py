"""Host UBSan oracle for the actual one-word cubic CPML point context.

Large-N cases exercise packing alone. Actual address arithmetic remains under
the existing NativeSession signed-int field and slab admission limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_cpml_point_word import helper
from scripts.patch_cuda_shared_profiles import ORIGINAL_COEFFICIENT
from scripts.validate_cuda_cpml_point_context import HARNESS, method


PACKING = r'''
size_t packing_checks=0,packing_errors=0,wrong_unsigned_decode=0;
template<int Phase,int Axis>void packing(){
 for(int n:{1,24,1289,1<<20}){
  Parameters p{};for(int a=0;a<3;++a)p.physical_n[a]=n;
  OriginalEnvironment<Axis> old{p};
  for(int coordinate:{-32,-16,-1,0,11,12,n/2,n-13,n-12,n-1,n,n+1,n+64}){
   int point[3]={3,5,7};point[Axis]=coordinate;
   const int z=point[0],y=point[1],x=point[2];
   CubicCpmlPointContext<Phase,Axis,OriginalEnvironment<Axis>>ctx(old,z,y,x);
   for(int term=0;term<6;++term){
    if(beamz::cuda::yee::CpmlAxis(term)!=Axis)continue;
    const int slot=Phase*6+term,want=old.Packed(slot,z,y,x);
    packing_errors+=ctx.PackedIndex()!=want;
    packing_errors+=ctx.LocalActive(slot,z,y,x)!=old.LocalActive(slot,z,y,x);
    wrong_unsigned_decode+=int(ctx.packed_active>>1)!=want;
    ++packing_checks;
   }
  }
 }
}
'''


def harness(context, methods):
    text = HARNESS.replace("@METHODS@", methods).replace("@COEFFICIENT@", ORIGINAL_COEFFICIENT).replace("@CONTEXT@", context)
    text = text.replace("ctx.Offset(slot)", "ctx.Offset(slot,z,y,x)")
    old = "int swapped=ctx.point_base+(slot%6!=ctx.ExtraTerm?ctx.point_delta:0);negative_stride+=want!=swapped;"
    new = "int other=0;while(beamz::cuda::yee::CpmlAxis(other)!=Axis||other==slot%6)++other;int swapped=ctx.Offset((slot/6)*6+other,z,y,x);negative_stride+=want!=swapped;"
    if text.count(old) != 1:
        raise ValueError("Reviewed four-field stride negative control changed")
    text = text.replace(old, new)
    marker = "int main(){run<0,0>();"
    if text.count(marker) != 1:
        raise ValueError("Reviewed oracle main changed")
    text = text.replace(marker, PACKING + "\nint main(){packing<0,0>();packing<0,1>();packing<0,2>();packing<1,0>();packing<1,1>();packing<1,2>();run<0,0>();")
    text = text.replace("errors+=negative_stride==0||negative_profile==0;", "errors+=packing_errors;errors+=negative_stride==0||negative_profile==0||wrong_unsigned_decode==0;")
    text = text.replace(r'\"errors\":%zu}', r'\"packing_checks\":%zu,\"wrong_unsigned_decode_counterexamples\":%zu,\"errors\":%zu}')
    text = text.replace("negative_stride,negative_profile,errors);", "negative_stride,negative_profile,packing_checks,wrong_unsigned_decode,errors);")
    if "point_base" in text or "point_delta" in text or "ctx.Offset(slot)" in text:
        raise ValueError("An old context member or call remains in the oracle")
    return text


def validate(source, output):
    source, output = Path(source), Path(output)
    consumer = (source / "src/asymmetric_native.cuh").read_text()
    context = helper()
    if consumer.count(context) != 1:
        raise ValueError("Actual one-word point context differs")
    methods = "\n".join(method(consumer, name) for name in (
        "int Packed(", "bool LocalActive(", "int PsiOffset(", "float LoadLocal(", "void StoreLocal("))
    program = harness(context, methods)
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "point-word.cc", output / "point-word"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    result.update(actual_consumer_sha256=hashlib.sha256(consumer.encode()).hexdigest(),
                  context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                  oracle_sha256=hashlib.sha256(program.encode()).hexdigest(), command=command,
                  scope="Actual original Environment vs one-word Phase/Axis context; six classes, addresses N24..1289, packing-only N1..2**20; support/CPML/outside boundaries, signed-zero and unequal independent profiles/state; UBSan; no CUDA resource or speed claim")
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(validate(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
