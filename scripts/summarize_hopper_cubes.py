"""Summarize completed cube trials without confusing allocator pools with live bytes."""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def timed_telemetry(path, data):
    csv_path = path.with_name(f"cube-{data['side']}-telemetry.csv")
    selected = []
    if csv_path.exists():
        with csv_path.open() as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                try:
                    stamp = (
                        datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                    )
                    if any(
                        start <= stamp <= end
                        for start, end in data.get("sample_windows_unix", [])
                    ):
                        selected.append([float(v.strip().split()[0]) for v in row[1:]])
                except (ValueError, IndexError):
                    continue
    return dict(
        timed_telemetry_samples=len(selected),
        timed_nvml_memory_max_gib=max(r[0] for r in selected) / 1024
        if selected
        else None,
        timed_gpu_busy_median_percent=statistics.median(r[2] for r in selected)
        if selected
        else None,
        timed_memory_busy_median_percent=statistics.median(r[3] for r in selected)
        if selected
        else None,
        timed_sm_clock_min_mhz=min(r[4] for r in selected) if selected else None,
        timed_sm_clock_max_mhz=max(r[4] for r in selected) if selected else None,
        timed_power_mean_w=statistics.mean(r[6] for r in selected)
        if selected
        else None,
        timed_temp_max_c=max(r[7] for r in selected) if selected else None,
    )


def summarize(root):
    rows = []
    for path in sorted(root.rglob("cube-*.json")):
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            # A live download may catch a record between writes; retry next sync.
            continue
        if d.get("status") != "complete":
            continue
        total_mib = float(d["stages"][0]["gpu"].split(",")[1])
        total = total_mib * 2**20
        peak = max(s["memory_stats"]["peak_bytes_in_use"] for s in d["stages"])
        warm = [s for s in d["stages"] if s["stage"].startswith("iteration_")]
        held = max(s["memory_stats"]["bytes_in_use"] for s in warm)
        temp = d["executable_memory"]["temp_size_in_bytes"]
        r = dict(
            run="finish" if "finish" in path.relative_to(root).parts else "initial",
            record=str(path.relative_to(root)),
            allocator=d["allocator_env"].get("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc"),
            side=d["side"],
            cells=d["cells"],
            gcups=d["gcups"],
            min_gcups=d["cells"] * d["steps"] / max(d["samples_s"]) / 1e9,
            max_gcups=d["cells"] * d["steps"] / min(d["samples_s"]) / 1e9,
            median_sample_s=statistics.median(d["samples_s"]),
            setup_s=d["setup_s"],
            compile_s=d["compile_s"],
            peak_live_gib=peak / 2**30,
            peak_live_percent=peak / total * 100,
            post_execution_live_gib=held / 2**30,
            runtime_live_plus_temp_estimate_gib=(held + temp) / 2**30,
            peak_pool_gib=(
                max(s["memory_stats"].get("pool_bytes", 0) for s in d["stages"]) / 2**30
                if any("pool_bytes" in s["memory_stats"] for s in d["stages"])
                else None
            ),
            nvml_post_execution_gib=max(float(s["gpu"].split(",")[0]) for s in warm)
            / 1024,
            host_peak_gib=d["host_peak_rss_bytes"] / 2**30,
            finite=d["finite_complete_state"],
            cpml_shell_fraction=1 - (1 - 24 / d["side"]) ** 3,
        )
        r.update(timed_telemetry(path, d))
        rows.append(r)
    rows.sort(key=lambda r: (r["allocator"], r["side"]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    rows = summarize(a.directory)
    (a.directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        return
    with (a.directory / "summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['allocator']} {r['side']}³  {r['cells'] / 1e6:.1f}M cells  {r['gcups']:.3f} GCUPS  peak live {r['peak_live_gib']:.2f} GiB ({r['peak_live_percent']:.1f}%)  pool {r['peak_pool_gib']} GiB"
        )
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.25)
        colors = {"bfc": "#167c80", "cuda_async": "#7853a5"}
        for allocator in sorted({r["allocator"] for r in rows}):
            group = [r for r in rows if r["allocator"] == allocator]
            x = [r["peak_live_percent"] for r in group]
            y = [r["gcups"] for r in group]
            ax.errorbar(
                x,
                y,
                yerr=[
                    [r["gcups"] - r["min_gcups"] for r in group],
                    [r["max_gcups"] - r["gcups"] for r in group],
                ],
                fmt="o-",
                capsize=3,
                color=colors[allocator],
                label=allocator,
            )
            for r in group:
                ax.annotate(
                    f"{r['side']}³",
                    (r["peak_live_percent"], r["gcups"]),
                    xytext=(0, 10 if allocator == "bfc" else -18),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        ax.axhline(30, color="#ad4934", ls="--", label="30 GCUPS target")
        ax.set(
            xlabel="Peak live GPU allocation across setup + execution (% of VRAM)",
            ylabel="Warm throughput (GCUPS)",
            title="H100 SXM: cube size sweep, FP32, CPML12",
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        ax.set_ylim(bottom=20, top=31)
        fig.text(
            0.01,
            0.02,
            "One mode source; one mode monitor, 3 frequencies; 256 steps; 7 samples. Bars: sample range.\nMemory is the process peak, not warm working set or allocator reservation.",
            fontsize=9,
        )
        fig.savefig(a.directory / "gcups-vs-memory.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
