"""Compare a selected checkout using the common realistic workload builder.

Invoke in a fresh process per revision. First-result timing includes setup and
compilation; execution timing includes internal storage conversion. Imports,
validation and result hashing are outside both timings.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


class DeviceMemorySampler:
    """Include native CUDA allocations that JAX allocator statistics omit."""

    def __init__(self, count):
        self.stop = threading.Event()
        self.error = None
        self.samples = 0
        self.peaks = []
        try:
            import pynvml

            pynvml.nvmlInit()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            indices = list(map(int, visible)) if visible != [""] else range(count)
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
            self.peaks = [0] * len(self.handles)
            self.totals = [
                pynvml.nvmlDeviceGetMemoryInfo(h).total for h in self.handles
            ]

            def poll():
                while not self.stop.is_set():
                    try:
                        for i, handle in enumerate(self.handles):
                            self.peaks[i] = max(
                                self.peaks[i],
                                pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                            )
                        self.samples += 1
                    except Exception as exc:
                        self.error = str(exc)
                        break
                    self.stop.wait(0.05)

            self.thread = threading.Thread(target=poll, daemon=True)
            self.thread.start()
        except Exception as exc:
            self.error = str(exc)

    def finish(self):
        self.stop.set()
        if hasattr(self, "thread"):
            self.thread.join()
        return dict(
            peak_used_bytes=self.peaks,
            total_bytes=getattr(self, "totals", []),
            samples=self.samples,
            error=self.error,
            interval_s=0.05,
        )


def enable_parallel_host_placement(workers):
    """Parallelize bounded host reads for capacity benchmarks only."""
    # Capacity-only setup option: evaluate the unchanged bounded region
    # reader on several host threads. GPU arrays and solver code are unchanged.
    from concurrent.futures import ThreadPoolExecutor

    import jax

    from beamz._region_array import RegionArray
    from beamz.simulation import sharding as sharding_runtime

    original_place = sharding_runtime.place_region_array

    def parallel_place(value, target):
        def place(item):
            device, index = item
            bounds = [key.indices(n) for key, n in zip(index, value.shape, strict=True)]
            shape = tuple(stop - start for start, stop, _ in bounds)

            def read(region):
                return value[
                    tuple(
                        slice(start + key.start, start + key.stop)
                        for (start, _, _), key in zip(bounds, region, strict=True)
                    )
                ]

            local = RegionArray(shape, value.dtype, read)
            return original_place(local, jax.sharding.SingleDeviceSharding(device))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            arrays = list(
                pool.map(
                    place, target.addressable_devices_indices_map(value.shape).items()
                )
            )
        return jax.make_array_from_single_device_arrays(value.shape, target, arrays)

    sharding_runtime.place_region_array = parallel_place


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shape", nargs=3, type=int, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--backend", default="cuda_streamed")
    parser.add_argument("--schedule", choices=("auto", "streamed", "mstep"))
    parser.add_argument("--boundary", choices=("all", "right", "pec"), default="all")
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--warmup-seconds", type=float, default=0)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--material", choices=("binary", "smooth"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--axes", choices=("012", "120", "201"), default="012")
    parser.add_argument("--fusion", choices=("auto", "0"), default="auto")
    parser.add_argument("--tuning-cache", type=Path)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--shard-axis", choices=("auto", "x", "y", "z"), default="auto")
    parser.add_argument("--host-setup", action="store_true")
    parser.add_argument("--host-placement-workers", type=int, default=1)
    parser.add_argument("--monitors", type=int, default=2)
    parser.add_argument("--frequencies", type=int, default=3)
    parser.add_argument("--monitor-offset-cells", type=int)
    parser.add_argument("--transverse-offset-cells", type=int, default=0)
    parser.add_argument(
        "--transverse-anchor", choices=("center", "low"), default="center"
    )
    parser.add_argument("--public-samples", type=int, default=0)
    parser.add_argument("--validation", choices=("full", "sampled"), default="full")
    parser.add_argument("--sample-points", type=int, default=2048)
    parser.add_argument("--mstep-tile", choices=("16x32", "8x32", "8x16", "16x16"))
    parser.add_argument("--mstep-segment", choices=("whole", "32", "64", "128", "256"))
    args = parser.parse_args()
    if args.samples < 3 or args.devices < 1 or args.warmups < 1:
        parser.error("need >=3 samples, positive devices and warmups")
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    for key in list(os.environ):
        if key.startswith("BEAMZ_CUDA_"):
            del os.environ[key]
    if args.axes != "012":
        os.environ["BEAMZ_CUDA_STORAGE_AXES"] = args.axes
    if args.fusion != "auto":
        os.environ["BEAMZ_CUDA_CPML_CORE_FUSION"] = args.fusion
    if args.tuning_cache is not None:
        os.environ["BEAMZ_CUDA_TUNING_CACHE"] = str(args.tuning_cache.resolve())
    if args.mstep_tile:
        os.environ["BEAMZ_CUDA_MSTEP_TILE"] = args.mstep_tile
    if args.mstep_segment:
        os.environ["BEAMZ_CUDA_MSTEP_SEGMENT"] = args.mstep_segment

    sampler = DeviceMemorySampler(args.devices)
    import beamz._cuda as extension
    import jax
    import numpy as np

    import beamz
    from beamz.simulation import observe
    from beamz.simulation import sharding as sharding_runtime
    from beamz.simulation.execute import build_scan, initial_program_state

    if args.host_placement_workers > 1:
        enable_parallel_host_placement(args.host_placement_workers)

    assert Path(beamz.__file__).resolve().parent.parent == root
    # Local development may symlink the extension into this checkout's build tree.
    assert Path(extension.__file__).resolve().is_relative_to(root)
    workload_path = Path(__file__).with_name("benchmark_cuda_realistic.py")
    spec = importlib.util.spec_from_file_location("comparison_workload", workload_path)
    workload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workload)
    settings = SimpleNamespace(
        shape=args.shape,
        steps=args.steps,
        presteps=0,
        pml=12,
        monitors=args.monitors,
        frequencies=args.frequencies,
        material=args.material,
        source="mode",
        monitor_type="mode",
        boundary=args.boundary,
        monitor_offset_cells=args.monitor_offset_cells,
        transverse_offset_cells=args.transverse_offset_cells,
        transverse_anchor=args.transverse_anchor,
    )
    devices = jax.devices("gpu")
    assert len(devices) == args.devices, (devices, args.devices)
    sharding = (
        dict(axis=args.shard_axis, num_devices=args.devices, backend="gpu")
        if args.devices > 1
        else None
    )
    start = time.perf_counter()
    options = {"schedule": args.schedule} if args.schedule is not None else {}
    placement = (
        jax.default_device(jax.devices("cpu")[0]) if args.host_setup else nullcontext()
    )
    with placement:
        sim = workload.build_simulation(settings)
        program = sim.compile(
            num_steps=args.steps, backend=args.backend, sharding=sharding, **options
        )
        state = initial_program_state(
            program, t=0, current_step=0, monitor_steps=args.steps
        )
    state = sharding_runtime.prepare_state(
        program,
        state,
        replicated_fields=(*observe.MONITOR_FIELDS, "t", "current_step"),
    )
    coefficients = sharding_runtime.place_tree(program, program.coefficients)
    if args.host_setup and args.devices == 1:
        state = jax.device_put(state, devices[0])
        coefficients = jax.device_put(coefficients, devices[0])
    jax.block_until_ready(state)
    assert len(state.ex.devices()) == args.devices
    if args.devices > 1:
        assert not state.ex.sharding.is_fully_replicated
    setup_s = time.perf_counter() - start
    print(json.dumps(dict(stage="setup", seconds=setup_s)), flush=True)
    start = time.perf_counter()
    scan = build_scan(program, donate_state=False)
    decision = getattr(scan, "cuda_schedule_decision", None)
    executable = scan.lower(state, coefficients).compile()
    compile_s = time.perf_counter() - start
    print(json.dumps(dict(stage="compiled", seconds=compile_s)), flush=True)
    start = time.perf_counter()
    jax.block_until_ready(executable(state, coefficients))
    first_run_s = time.perf_counter() - start
    for _ in range(max(0, args.warmups - 1)):
        jax.block_until_ready(executable(state, coefficients))
    warmup_start = time.perf_counter()
    additional_warmups = 0
    while time.perf_counter() - warmup_start < args.warmup_seconds:
        jax.block_until_ready(executable(state, coefficients))
        additional_warmups += 1
    samples = []
    result = None
    for _ in range(args.samples):
        # Release the previous output before allocating the next one.
        del result
        start = time.perf_counter()
        result = jax.block_until_ready(executable(state, coefficients))
        samples.append(time.perf_counter() - start)
    kernel_memory = [d.memory_stats() for d in devices]
    state_cpml_dtypes = sorted(
        {str(x.dtype) for x in (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)}
    )
    assert all(dtype == "float32" for dtype in state_cpml_dtypes)
    print(json.dumps(dict(stage="timed", samples_s=samples)), flush=True)

    def host_values(leaf):
        if args.validation == "full" or leaf.size <= args.sample_points:
            return np.asarray(jax.device_get(leaf))
        shards = getattr(leaf, "addressable_shards", None)
        if shards is None:
            flat = np.asarray(leaf).reshape(-1)
            return flat[
                np.linspace(0, flat.size - 1, args.sample_points, dtype=np.int64)
            ]
        if leaf.sharding.is_fully_replicated:
            shards = shards[:1]
        values = []
        for shard in shards:
            flat = shard.data.reshape(-1)
            indices = np.linspace(
                0, flat.size - 1, min(flat.size, args.sample_points), dtype=np.int64
            )
            values.append(np.asarray(jax.device_get(flat[indices])))
            # Uniform samples can miss the pulse in a multi-billion-cell grid.
            # Also retain the source neighbourhood of full-volume Yee leaves.
            if leaf.ndim == 3 and all(
                n <= m <= n + 1 for n, m in zip(args.shape, leaf.shape, strict=True)
            ):
                nz, ny, nx = args.shape
                centers = [
                    nz // 2 + args.transverse_offset_cells,
                    ny // 2 + args.transverse_offset_cells,
                    int(0.3 * nx),
                ]
                if args.transverse_anchor == "low":
                    centers[:2] = [min(n // 2 + 1, settings.pml + 12) for n in (nz, ny)]
                local = []
                for size, key, center, radius in zip(
                    leaf.shape, shard.index, centers, (12, 12, 48), strict=True
                ):
                    start, stop, _ = key.indices(size)
                    lo, hi = max(start, center - radius), min(stop, center + radius)
                    local.append(slice(max(0, lo - start), max(0, hi - start)))
                if all(key.stop > key.start for key in local):
                    values.append(
                        np.asarray(jax.device_get(shard.data[tuple(local)])).ravel()
                    )
        return np.concatenate(values)

    def fingerprints(value):
        return [
            dict(
                shape=list(a.shape),
                dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            )
            for leaf in jax.tree.leaves(value)
            for a in [host_values(leaf)]
        ]

    memory_before_validation = sampler.finish()
    peak_host_rss_before_validation = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    reference = args.reference
    monitor_signal_max = max(
        float(np.max(np.abs(np.asarray(jax.device_get(value))), initial=0))
        for value in (result.dft_vec_re, result.dft_vec_im)
    )
    monitor_weights = np.asarray(jax.device_get(result.dft_weight_sum)).tolist()
    leaves = jax.tree.leaves(result)
    if args.save_reference:
        reference.mkdir(parents=True, exist_ok=False)
    else:
        assert len(list(reference.glob("*.npy"))) == len(leaves)
    validation = []
    for index, leaf in enumerate(leaves):
        actual = host_values(leaf)
        path = reference / f"{index:03}.npy"
        if args.save_reference:
            np.save(path, actual)
            expected = actual
        else:
            expected = np.load(path, mmap_mode="r")
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert np.isfinite(actual).all()
        exact = np.array_equal(actual, expected)
        scale = float(np.max(np.abs(expected), initial=0))
        max_error = float(np.max(np.abs(actual - expected), initial=0))
        tolerance_pass = True
        mismatch_count = 0
        if np.issubdtype(actual.dtype, np.inexact):
            mismatch_count = int(
                np.count_nonzero(
                    np.abs(actual - expected)
                    > max(3e-6, 1e-6 * scale) + 3e-5 * np.abs(expected)
                )
            )
            tolerance_pass = mismatch_count == 0
        else:
            np.testing.assert_array_equal(actual, expected)
        validation.append(
            dict(
                index=index,
                shape=list(actual.shape),
                logical_shape=list(leaf.shape),
                dtype=str(actual.dtype),
                exact=exact,
                max_abs_error=max_error,
                reference_scale=scale,
                tolerance_pass=tolerance_pass,
                mismatch_count=mismatch_count,
                sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            )
        )
    input_fingerprints = fingerprints(state)
    public_samples = []
    public_memory = None
    if args.public_samples:
        del result, leaves, leaf, state, coefficients, executable, scan
        public_sampler = DeviceMemorySampler(args.devices)
        for sample in range(args.public_samples + 1):
            start = time.perf_counter()
            result = sim.advance(
                num_steps=args.steps,
                backend=args.backend,
                sharding=sharding,
                performance=False,
                **options,
            ).state
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - start
            if sample:
                public_samples.append(elapsed)
            del result
        public_memory = public_sampler.finish()
    revision_file = root / ".revision"
    data = dict(
        workload=vars(settings),
        root=str(root),
        commit=revision_file.read_text().strip()
        if revision_file.exists()
        else subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        extension_version=extension.__version__,
        extension_abi=extension.__abi_version__,
        extension_sha256=hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        workload_sha256=hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        jax_version=jax.__version__,
        device=str(jax.devices()[0]),
        devices=[dict(id=d.id, kind=d.device_kind) for d in devices],
        device_count=args.devices,
        shard_axis=program.sharding.layout.axis_name,
        host_setup=args.host_setup,
        host_placement_workers=args.host_placement_workers,
        kernel_device_memory=kernel_memory,
        nvml_memory=memory_before_validation,
        public_nvml_memory=public_memory,
        peak_host_rss_before_validation=peak_host_rss_before_validation,
        validation_scope=args.validation,
        sampled_source_neighbourhood=True,
        monitor_signal_max=monitor_signal_max,
        monitor_weights=monitor_weights,
        public_samples_s=public_samples,
        requested_axes=args.axes,
        actual_axes=getattr(program.config, "cuda_storage_axes", (0, 1, 2)),
        fusion=args.fusion,
        cuda_flags=program.config.cuda_flags,
        setup_s=setup_s,
        compile_s=compile_s,
        backend=args.backend,
        schedule=args.schedule,
        schedule_decision=decision,
        first_run_s=first_run_s,
        first_result_s=setup_s + compile_s + first_run_s,
        warmups=args.warmups,
        additional_warmups=additional_warmups,
        warmup_seconds=args.warmup_seconds,
        samples_s=samples,
        median_gcups=int(np.prod(args.shape))
        * args.steps
        / statistics.median(samples)
        / 1e9,
        initial_state=input_fingerprints,
        coefficients=fingerprints(program.coefficients),
        sources=[
            dict(
                component=s.component,
                timing=s.timing,
                slab_starts=s.slab_starts,
                slab_sizes=s.slab_sizes,
                arrays=fingerprints((s.coeff, s.waveform)),
            )
            for s in program.sources
        ],
        final_state_validation=validation,
        source_specs=len(program.sources),
        cpml_dtypes=state_cpml_dtypes,
        validation_rtol=3e-5,
        validation_atol="max(3e-6, 1e-6 * reference_leaf_max_abs)",
    )
    assert all(dtype == "float32" for dtype in data["cpml_dtypes"])
    if (root / "beamz/simulation/cuda/tuning.py").exists():
        from beamz.simulation.cuda.tuning import tuning_report

        data["tuning"] = tuning_report(program)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {k: data[k] for k in ("root", "median_gcups", "actual_axes", "cuda_flags")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
