"""Audit bounded macro buffers against the actual planner and version model.

The existing planner harness supplies its exhaustive encoded-task, slot,
certificate and neighbor checks. This adds raw-ring reuse and drain boundaries
and independently checks that the numerical bodies are unchanged.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block
from scripts import prove_cuda_native_macro_pipeline as model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(candidate, parent):
    candidate, parent = Path(candidate).resolve(), Path(parent).resolve()
    src = candidate / 'source/src'
    proof = json.loads((candidate / 'macro-pipeline-compiler-proof.json').read_text())
    for file, key in [('asymmetric_native.cuh', 'consumer'), ('native_extension.cc', 'extension'),
                      ('native_moving.h', 'header'), ('native_macro_slab.h', 'planner')]:
        assert digest(src / file) == proof[key + '_sha256'], key
    assert digest(parent / 'src/asymmetric_native.cuh') == proof['parent_consumer_sha256']
    for name in ('asymmetric_native_cell_generated.cuh', 'yee_cell_generated.cuh', 'native_moving.cc', 'native_launch.cc'):
        assert (src / name).read_bytes() == (parent / 'src' / name).read_bytes(), name
    c = (src / 'asymmetric_native.cuh').read_text()
    old = (parent / 'src/asymmetric_native.cuh').read_text()
    assert block(c, 'float Finish(')[2].replace('+raw_step_offset', '+step_offset') == block(old, 'float Finish(')[2]
    for signature in ('bool Valid(', 'int Packed(', 'bool LocalActive(', 'int PsiOffset(',
                      'float LoadLocal(', 'void StoreLocal(', 'float Coefficient(', 'float Difference(',
                      'void Observe(', 'FieldPacket ImportPacket(', 'float Initial(', 'const MstepCompactField& MacroFinalField('):
        assert block(c, signature)[2] == block(old, signature)[2], signature
    raw_slots = proof['raw_ring_slots']
    extra = f'env.raw_step_offset=(MacroStepOffset<1>(p,tiles[blockIdx.x])%{raw_slots})*Depth;'
    assert c.count(extra) == 3
    for signature in ('void UniformBody(', 'void FixedCubeCachedBody(', 'void SourcePendingPairedBody('):
        assert block(c, signature)[2].replace(extra, '') == block(old, signature)[2], signature
    e = (src / 'native_extension.cc').read_text()
    for text in ('level_begin=level_end;', 'level_begin,level_end,m==0',
                 f'offset+m*depth,(m%{raw_slots})*depth', 'DeferScratchForMoving();',
                 'if(mstep_)EnsureScratchResources();'):
        assert text in e, text
    entry = block(c, 'int BeamzEnqueueNativeMacroSlab(')[2]
    assert 'if(begin_epoch)' in entry and 'slab.policy_offsets_host+=size_t(level_begin)*7' in entry
    planner = (src / 'native_macro_slab.h').read_text()
    safety = f' stride=std::max(stride,int((ranks+{raw_slots}-1)/{raw_slots}));\n'
    assert planner.index(safety) < planner.index('if(ranks+size_t(macros-1)*stride>')
    harness_path = Path(__file__).with_name('validate_cuda_macro_pipeline_host.py')
    tree = ast.parse(harness_path.read_text())
    code = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'code' for t in n.targets))
    extra_checks = r'''
  // Actual affine producer levels for representative raw observation points.
  // Emulate every physical raw write and every chronological macro drain.
  if(3*slab.macro_level_stride<int(p.bulk_counts.size()))return 21;
  struct Effect{int level,macro,version,key;};std::vector<Effect>effects;
  for(int m=0;m<macros;++m)for(int n=1;n<=depth;++n)for(int key=0;key<3;++key){
   int rank=0;for(int a=0;a<3;++a){int point=key==0?0:key==1?dims[a]/2:dims[a]-1;rank+=(point+n)/p.owned[a];}
   effects.push_back({rank+m*slab.macro_level_stride,m,n,key});
  }
  std::stable_sort(effects.begin(),effects.end(),[](const Effect&a,const Effect&b){return a.level<b.level;});
  std::vector<int>raw(size_t(3)*depth*3,-1);size_t next=0;int begin=0,tasks_seen=0;
  for(int m=0;m<macros;++m){int end=int(p.bulk_counts.size())+m*slab.macro_level_stride;
   if(begin<0||end<=begin||end>slab.level_count)return 22;
   for(int level=begin;level<end;++level)tasks_seen+=slab.policy_offsets[level*7+6]-slab.policy_offsets[level*7];
   while(next<effects.size()&&effects[next].level<end){const auto&x=effects[next++];raw[((x.macro%3)*depth+x.version-1)*3+x.key]=x.macro*depth+x.version;}
   for(int n=1;n<=depth;++n)for(int key=0;key<3;++key)if(raw[((m%3)*depth+n-1)*3+key]!=m*depth+n)return 23;
   begin=end;
  }
  if(begin!=slab.level_count||tasks_seen!=tiles*macros||next!=effects.size())return 24;
  if(dims[0]==512&&dims[1]==512&&dims[2]==512&&depth==16&&macros==8)
   std::cerr<<"512 stride="<<slab.macro_level_stride<<" frontier="<<slab.max_frontier<<" levels="<<slab.level_count<<"\n";
'''.replace('3*slab.macro_level_stride', f'{raw_slots}*slab.macro_level_stride').replace('size_t(3)*depth*3', f'size_t({raw_slots})*depth*3').replace('macro%3', f'macro%{raw_slots}').replace('m%3', f'm%{raw_slots}')
    code = code.replace('  ++cases;', extra_checks + '\n  ++cases;')
    directory = candidate / 'host-macro-pipeline-proof'
    directory.mkdir(exist_ok=False)
    rows = []
    variants = [('actual', planner, code),
                ('unsafe-raw-reuse', planner.replace(safety, ''), code.replace(f'  if({raw_slots}*slab.macro_level_stride<int(p.bulk_counts.size()))return 21;', '')),
                ('wrong-raw-drain-slot', planner, code.replace(f'raw[((m%{raw_slots})*depth+n-1)', f'raw[(((m+1)%{raw_slots})*depth+n-1)')),
                ('duplicate-range', planner, code.replace('   begin=end;', '   begin=0;')),
                ('wrong-macro-encoding', planner.replace('size_t(m)*tiles+tile', 'size_t(m)*(tiles-1)+tile'), code)]
    for name, header, test_code in variants:
        folder = directory / name
        folder.mkdir()
        (folder / 'native_macro_slab.h').write_text(header)
        cc, binary = folder / 'proof.cc', folder / 'proof'
        cc.write_text(test_code)
        subprocess.run(['g++', '-std=c++17', '-O2', '-fsanitize=address,undefined',
                        '-I' + str(folder), '-I' + str(src), str(cc), str(src / 'native_moving.cc'),
                        '-o', str(binary)], capture_output=True, text=True, check=True)
        r = subprocess.run([str(binary)], capture_output=True, text=True)
        assert (r.returncode == 0) == (name == 'actual'), (name, r.returncode, r.stdout, r.stderr)
        # A negative must fail a deliberate semantic check, not crash.
        if name != 'actual':
            assert r.returncode in (3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24), (name, r.stderr)
        rows.append(dict(name=name, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
    version_source = inspect.getsource(model.token_pipeline)
    bounded = version_source.replace('stride=1+sum(shift)',
                                    f'stride=max(1+sum(shift),(sum(counts)-2+{raw_slots}-1)//{raw_slots})')
    assert bounded != version_source
    namespace = dict(vars(model))
    exec(compile(bounded, '<bounded-stride-token-model>', 'exec'), namespace)
    version_cases = [namespace['token_pipeline'](minimum=(40,13,17), owned=(3,5,7), depth=4, macros=8, seed=s)
                     for s in (13,37)]
    result = dict(passed=True, consumer_sha256=proof['consumer_sha256'], planner_sha256=proof['planner_sha256'],
                  rows=rows, unchanged_helpers=True, canonical_headers_byte_identical=True, exact_Finish_inverse=True,
                  version_cases=version_cases, version_model_sha256=digest(Path(model.__file__)),
                  bounded_version_model_sha256=hashlib.sha256(bounded.encode()).hexdigest(),
                  planner_harness_sha256=digest(harness_path), raw_slots=raw_slots,
                  scope='Actual planner, all task/slot/certificate/dependency checks, chronological raw ring drains and four negative controls, ASan/UBSan; wider-stride native field/history token model. GPU full-output and method-switch validation separate.')
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(validate(a.candidate, a.parent)))
