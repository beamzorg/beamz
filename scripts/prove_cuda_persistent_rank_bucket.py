"""Actual C++ bucket-priority plans versus an independent Python DAG oracle."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_persistent_wavefront_dag import build, tile_rank
from scripts.patch_cuda_persistent_rank_bucket import PARENT, transformed


HARNESS = r'''
#include "native_persistent.h"
#include <iostream>
#include <numeric>
#include <random>
void array(const std::vector<int32_t>&x){std::cout<<'[';for(size_t i=0;i<x.size();++i){if(i)std::cout<<',';std::cout<<x[i];}std::cout<<']';}
BeamzNativeMovingPlan moving(std::array<int,3>dims,int seed){
 BeamzNativeMovingPlan p;p.counts=dims;p.depth=16;p.epochs=4;
 const int count=dims[0]*dims[1]*dims[2],ranks=dims[0]+dims[1]+dims[2]-2;
 p.bulk_counts.assign(ranks,0);p.slots.assign(count,0);p.rank_offsets={0};std::mt19937 rng(seed+19);
 for(int rank=0;rank<ranks;++rank){std::vector<int32_t>nodes,perm;for(int t=0;t<count;++t)if(BeamzPersistentTileRank(t,p)==rank)nodes.push_back(t);
  perm.resize(nodes.size());std::iota(perm.begin(),perm.end(),0);
  if(dims==std::array<int,3>{2,2,2}){int index=rank==1?seed%6:rank==2?(seed/6)%6:0;while(index--)std::next_permutation(perm.begin(),perm.end());}
  else std::shuffle(perm.begin(),perm.end(),rng);
  for(size_t i=0;i<nodes.size();++i)p.slots[nodes[i]]=perm[i];
  // Slot IDs are deliberately independent of input within-rank job order.
  std::shuffle(nodes.begin(),nodes.end(),rng);p.tile_ids.insert(p.tile_ids.end(),nodes.begin(),nodes.end());
  p.max_frontier=std::max(p.max_frontier,int(nodes.size()));p.rank_offsets.push_back(p.tile_ids.size());
 }
 return p;
}
int main(){size_t cases=0,rejected=0;
 try{
  for(auto dims:std::vector<std::array<int,3>>{{1,1,1},{2,2,2},{2,3,4},{8,4,3},{12,9,6},{76,36,18}}){
   int seeds=dims==std::array<int,3>{2,2,2}?36:dims==std::array<int,3>{76,36,18}?1:4;
   for(int seed=0;seed<seeds;++seed){auto p=moving(dims,seed);auto rank=BeamzBuildNativePersistentPlan(p,false),locality=BeamzBuildNativePersistentPlan(p,true);
    if(rank.job_ids!=p.tile_ids||rank.previous_holder!=locality.previous_holder)throw std::logic_error("rank input or slot holders changed");
    std::cout<<"{\"type\":\"case\",\"counts\":["<<dims[0]<<','<<dims[1]<<','<<dims[2]<<"],\"seed\":"<<seed<<",\"frontier\":"<<p.max_frontier<<",\"slots\":";array(p.slots);
    std::cout<<",\"rank_jobs\":";array(rank.job_ids);std::cout<<",\"locality_jobs\":";array(locality.job_ids);
    std::cout<<",\"previous_holder\":";array(locality.previous_holder);std::cout<<",\"job_index\":";array(locality.job_index);std::cout<<"}\n";++cases;
   }
  }
  auto good=moving({2,3,4},0);
  for(int mode=0;mode<6;++mode){auto bad=good;
   if(mode==0){int r=1;bad.slots[bad.tile_ids[bad.rank_offsets[r]+1]]=bad.slots[bad.tile_ids[bad.rank_offsets[r]]];}
   if(mode==1)bad.tile_ids[1]=bad.tile_ids[0];
   if(mode==2)bad.epochs=3;
   if(mode==3)bad.counts[0]+=1;
   if(mode==4)bad.slots[0]=bad.max_frontier;
   if(mode==5)std::swap(bad.tile_ids[0],bad.tile_ids.back());
   try{BeamzBuildNativePersistentPlan(bad,true);throw std::logic_error("bad plan accepted");}catch(const std::invalid_argument&){++rejected;}
  }
  std::cout<<"{\"type\":\"summary\",\"cases\":"<<cases<<",\"invalid_plan_rejections\":"<<rejected<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}
'''


def morton(tile, counts):
    _, ny, nx = counts
    z, y, x = tile // (ny * nx), (tile // nx) % ny, tile % nx
    return sum(((x >> bit) & 1) << (3 * bit) | ((y >> bit) & 1) << (3 * bit + 1) | ((z >> bit) & 1) << (3 * bit + 2) for bit in range(21))


def oracle(dag, window):
    remaining = [len(p) for p in dag["predecessors"]]
    key = lambda t: (dag["ranks"][t] // window, morton(t, dag["counts"]), t)
    ready = [key(t) for t, count in enumerate(remaining) if not count]
    heapq.heapify(ready)
    jobs = []
    while ready:
        _, _, tile = heapq.heappop(ready)
        jobs.append(tile)
        for child in dag["successors"][tile]:
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(ready, key(child))
    if len(jobs) != len(remaining):
        raise AssertionError("Independent DAG oracle did not cover all tiles")
    return jobs


def verify_order(dag, jobs):
    if sorted(jobs) != list(range(len(dag["ranks"]))):
        raise AssertionError("Not a full tile permutation")
    done = set()
    for tile in jobs:
        if not dag["predecessors"][tile] <= done:
            raise AssertionError("RAW or old reader/holder dependency missing")
        done.add(tile)


def validate(source, output, window):
    source, output = Path(source), Path(output)
    original_path = Path(__file__).resolve().parents[1] / ".cache/cuda-five-hour-2026-11-13-1059/persistent-p5-g4-floor-global-v1/source/src/native_persistent.h"
    header = (source / "src/native_persistent.h").read_bytes()
    if header.decode() != transformed(original_path.read_text(), window):
        raise ValueError("Actual copied header differs from bound bucket transformation")
    if hashlib.sha256((source / "src/native_persistent.cuh").read_bytes()).hexdigest() != PARENT["native_persistent.cuh"]:
        raise ValueError("Actual GPU kernel changed")
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "plan.cc", output / "plan"
    cpp.write_text(HARNESS)
    command = ["c++", "-std=c++20", "-O2", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", "-I" + str(source / "src"), str(cpp), "-o", str(binary)]
    compilation = subprocess.run(command, capture_output=True, text=True)
    (output / "compile.log").write_text(compilation.stdout + compilation.stderr)
    compilation.check_returncode()
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    (output / "actual-plans.jsonl").write_text(run.stdout)
    cases, summary = [], None
    priority_checks = dependency_checks = raw_negative = war_negative = 0
    full_cube_slot_permutations = set()
    for line in run.stdout.splitlines():
        case = json.loads(line)
        if case["type"] == "summary":
            summary = case
            continue
        counts, slots = tuple(case["counts"]), case["slots"]
        dag = build(counts, slots, case["frontier"])
        expected = oracle(dag, window)
        if case["locality_jobs"] != expected:
            raise AssertionError("Actual C++ plan violates min-available bucket/Morton/tile priority")
        verify_order(dag, case["rank_jobs"])
        verify_order(dag, expected)
        previous = [dag["previous_slot_holder"].get(t, -1) for t in range(len(slots))]
        if previous != case["previous_holder"]:
            raise AssertionError("Actual C++ slot owners differ from independent Python DAG")
        if any(case["job_index"][t] != i for i, t in enumerate(expected)):
            raise AssertionError("Actual C++ inverse job index differs")
        priority_checks += len(expected)
        dependency_checks += dag["all_edges"]
        if counts == (2, 2, 2):
            full_cube_slot_permutations.add(tuple(slots))
        # Full permutations with an inverted actual edge must fail the oracle.
        for kind, edge in (("RAW", next(((p, t) for p, children in enumerate(dag["raw_successors"]) for t in children), None)),
                           ("WAR", next(((reader, t) for t, old in dag["previous_slot_holder"].items() for reader in dag["raw_successors"][old] if t not in dag["raw_successors"][reader]), None))):
            if edge is None:
                continue
            bad = expected.copy();a,b = edge
            ia,ib = case["job_index"][a],case["job_index"][b]
            bad[ia],bad[ib] = bad[ib],bad[ia]
            try:
                verify_order(dag,bad)
            except AssertionError:
                if kind == "RAW":raw_negative += 1
                else:war_negative += 1
            else:
                raise AssertionError("Inverted dependency was accepted:" + kind)
        cases.append({"counts": counts, "seed": case["seed"], "tiles": len(slots), "all_edges": dag["all_edges"],
                      "locality_job_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest()})
    if summary is None or summary["invalid_plan_rejections"] != 6 or len(full_cube_slot_permutations) != 36 or not war_negative or not raw_negative:
        raise AssertionError("Incomplete actual plan admission/slot permutation/negative coverage")
    report = {"schema": "beamz.rank-bucket-Morton-actual-plan-proof.v1", "window": window,
              "header_sha256": hashlib.sha256(header).hexdigest(), "actual_CXX_summary": summary,
              "min_available_bucket_priority_checks": priority_checks, "RAW_WAR_edge_checks": dependency_checks,
              "all_small_cube_slot_permutations": len(full_cube_slot_permutations),
              "inverted_RAW_negative_rejections": raw_negative, "inverted_WAR_negative_rejections": war_negative,
              "cases": cases, "command": command, "harness_sha256": hashlib.sha256(HARNESS.encode()).hexdigest(),
              "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "GPU_executed": False,
              "scope": "Actual copied C++ header plans vs independent Python RAW+slot-reuse DAG and minimum available(rank//window,Morton,tile) Kahn priority; actual within-rank slots permuted independently of rank input job order; no hard bound on in-flight GPU ranks or speed claim"}
    (output / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--window", required=True, type=int, choices=(2, 4))
    a = p.parse_args()
    print(json.dumps(validate(a.source, a.output, a.window), indent=2))
