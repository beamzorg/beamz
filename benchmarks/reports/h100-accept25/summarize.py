"""Summarize raw H100 acceptance records; no extrapolated throughput."""

import json
import re
import statistics as s
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw = ROOT / "raw"
groups = {}
for p in raw.glob("paired-*.json"):
    m = re.fullmatch(
        r"paired-(\d+)-(\d+)-([xz])-(jax|cuda_streamed)-(baseline|fixed)", p.stem
    )
    if not m:
        continue
    rep, count, axis, backend, rev = m.groups()
    groups.setdefault((count, axis, backend), {}).setdefault(int(rep), {})[rev] = (
        json.loads(p.read_text())
    )
rows = []
for (count, axis, backend), reps in sorted(groups.items()):
    pairs = []
    for rep, versions in sorted(reps.items()):
        if set(versions) != {"baseline", "fixed"}:
            continue
        b, f = versions["baseline"], versions["fixed"]
        bt = s.median(b["warm_runtime_samples_s"])
        ft = s.median(f["warm_runtime_samples_s"])
        pairs.append(
            dict(
                repetition=rep,
                baseline_s=bt,
                fixed_s=ft,
                slowdown_percent=100 * (ft / bt - 1),
                baseline_gcups=b["kernel_gcups"],
                fixed_gcups=f["kernel_gcups"],
                baseline_compile_s=b["compile_s"],
                fixed_compile_s=f["compile_s"],
                baseline_cv=b["runtime_cv"],
                fixed_cv=f["runtime_cv"],
                fixed_capacity=f.get("cuda_schedule"),
            )
        )
    if pairs:
        rows.append(
            dict(
                devices=int(count),
                axis=axis,
                backend=backend,
                pairs=pairs,
                median_slowdown_percent=s.median(p["slowdown_percent"] for p in pairs),
                all_pairs_under_five_percent=len(pairs) == 3
                and all(p["slowdown_percent"] < 5 for p in pairs),
            )
        )
capacity = []
for p in raw.glob("*.json"):
    d = json.loads(p.read_text())
    if not isinstance(d, dict) or "stages" not in d:
        continue
    times = [
        stage["seconds"] for stage in d["stages"] if stage["stage"].startswith("run_")
    ]
    capacity.append(
        dict(
            name=p.stem,
            status=d["status"],
            failure_stage=d.get("failure_stage"),
            cells=d.get("cells"),
            devices=d.get("device_count"),
            schedule=d.get("cuda_schedule"),
            input_gib=d.get("input_bytes", 0) / 2**30,
            executable_bytes=d.get("executable_bytes"),
            final_step=d.get("final_step"),
            warm_gcups=d["cells"] * d["args"]["steps"] / s.median(times[1:]) / 1e9
            if len(times) > 1
            else None,
            stages=[
                dict(
                    stage=t["stage"],
                    seconds=t["seconds"],
                    host_peak_gib=t["max_rss_bytes"] / 2**30,
                    gpus=[
                        dict(
                            id=g["id"],
                            live_gib=g["stats"]["bytes_in_use"] / 2**30,
                            peak_gib=g["stats"]["peak_bytes_in_use"] / 2**30,
                            limit_gib=g["stats"]["bytes_limit"] / 2**30,
                        )
                        for g in t["gpus"]
                    ],
                )
                for t in d["stages"]
            ],
        )
    )
summary = dict(performance=rows, capacity=capacity)
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
