"""Explicit notebook-only bridge and instrumentation for native CUDA trials.

The public frontend is imported from the caller's pinned source tree. The
experimental branch packs exactly its Mstep ABI, runs the immutable native
module, and uses that same frontend's result decoding and normalization.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import json
import os
import resource
import threading
import time
from pathlib import Path

import numpy as np


class MemorySampler:
    def __init__(self, output):
        self.output = Path(output)
        self.rows = []
        self.phase = "notebook"
        self.stop_event = threading.Event()
        self.lib = ct.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2():
            raise RuntimeError("NVML initialization failed")
        self.handle = ct.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ct.byref(self.handle)):
            raise RuntimeError("NVML GPU handle failed")

        class Process(ct.Structure):
            _fields_ = [
                ("pid", ct.c_uint),
                ("used", ct.c_ulonglong),
                ("gpu_instance", ct.c_uint),
                ("compute_instance", ct.c_uint),
            ]

        self.Process = Process
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        import psutil

        process = psutil.Process()
        while not self.stop_event.is_set():
            count = ct.c_uint(64)
            rows = (self.Process * 64)()
            status = self.lib.nvmlDeviceGetComputeRunningProcesses_v2(
                self.handle, ct.byref(count), rows
            )
            if status:
                raise RuntimeError(
                    "NVML per-process memory query failed: " + str(status)
                )
            gpu = next(
                (int(x.used) for x in rows[: count.value] if x.pid == os.getpid()), 0
            )
            row = dict(
                t=time.perf_counter(),
                phase=self.phase,
                gpu_process_bytes=gpu,
                rss_bytes=process.memory_info().rss,
            )
            self.rows.append(row)
            if len(self.rows) % 20 == 0:
                with (self.output / "memory-live.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
            available = psutil.virtual_memory().available
            if available < 2 * (1 << 30):
                (self.output / "host-memory-guard.json").write_text(
                    json.dumps(
                        dict(
                            available_bytes=available,
                            last_sample=row,
                            reason="Abort own trial before another system host-RAM OOM; leave unrelated applications untouched",
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                os._exit(75)
            self.stop_event.wait(0.05)

    def summary(self):
        if not self.rows or not any(x["gpu_process_bytes"] for x in self.rows):
            raise RuntimeError("No positive process GPU memory samples")
        phases = {}
        for name in sorted({x["phase"] for x in self.rows}):
            rows = [x for x in self.rows if x["phase"] == name]
            phases[name] = dict(
                peak_gpu_process_bytes=max(x["gpu_process_bytes"] for x in rows),
                peak_host_rss_bytes=max(x["rss_bytes"] for x in rows),
                samples=len(rows),
            )
        return dict(
            peak_gpu_process_bytes=max(x["gpu_process_bytes"] for x in self.rows),
            peak_host_rss_bytes=max(x["rss_bytes"] for x in self.rows),
            host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            sampling_interval_s=0.05,
            phases=phases,
        )


class Trial:
    def __init__(self, variant, output, module_path=None):
        import beamz as bz
        from beamz.simulation import execute

        self.variant, self.output = variant, Path(output)
        self.host_reset = (
            variant == "bounded" and os.getenv("BEAMZ_NATIVE_HOST_RESET") == "1"
        )
        if self.host_reset:
            import jax

            jax.config.update("jax_default_device", jax.devices("cpu")[0])
        self.started = time.perf_counter()
        self.memory = MemorySampler(self.output)
        self.runs = []
        self.last_state = None
        self.session = None
        self.module_path = module_path
        self.original_run = bz.Simulation.run
        original_timed = execute._run_program_state_timed
        self.decode_samples = []
        self.result_build_samples = []
        from beamz.simulation.results import SimulationResults

        original_build = SimulationResults.from_run

        def build(cls, *args, **kwargs):
            start = time.perf_counter()
            try:
                return original_build(*args, **kwargs)
            finally:
                self.result_build_samples.append(time.perf_counter() - start)

        SimulationResults.from_run = classmethod(build)
        original_decode = execute._decode_monitor_results

        def decode(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original_decode(*args, **kwargs)
            finally:
                self.decode_samples.append(time.perf_counter() - start)

        execute._decode_monitor_results = decode

        def timed(*args, **kwargs):
            previous = self.memory.phase
            self.memory.phase = "engine"
            try:
                state, elapsed = original_timed(*args, **kwargs)
                self.last_state = state
                return state, elapsed
            finally:
                self.memory.phase = previous

        execute._run_program_state_timed = timed

        def run(simulation, *args, **kwargs):
            self.last_state = None
            previous = self.memory.phase
            self.memory.phase = "public_run"
            start = time.perf_counter()
            decode_start = len(self.decode_samples)
            build_start = len(self.result_build_samples)
            try:
                if variant == "main":
                    result = self.original_run(simulation, *args, **kwargs)
                    native = None
                else:
                    if args:
                        raise ValueError(
                            "Notebook bridge accepts the default complete fresh run only"
                        )
                    result, native = self.native_run(simulation, kwargs)
                elapsed = time.perf_counter() - start
                self.runs.append(
                    dict(
                        public_wall_s=elapsed,
                        engine_s=result.performance.runtime_s,
                        gcups=result.performance.gcups,
                        native=native,
                        monitor_decode_s=sum(self.decode_samples[decode_start:]),
                        result_build_s=sum(self.result_build_samples[build_start:]),
                    )
                )
                return result
            finally:
                self.memory.phase = previous

        bz.Simulation.run = run

    def native_run(self, simulation, kwargs):
        import jax

        from beamz.simulation import execute
        from beamz.simulation.cuda import runtime as rt
        from beamz.simulation.results import SimulationResults
        from scripts.export_cuda_native_fixture import capture_mstep, initial_native_state

        prepare_start = time.perf_counter()

        def stage(name, **data):
            self.memory.phase = name
            import psutil

            with (self.output / "native-preparation.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            stage=name,
                            elapsed_s=time.perf_counter() - prepare_start,
                            rss_bytes=psutil.Process().memory_info().rss,
                            **data,
                        )
                    )
                    + "\n"
                )

        if self.session is None:
            stage("module_load")
            spec = importlib.util.spec_from_file_location(
                "_cuda_native", self.module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.module = module
            # Main prepares progress-sized programs, while source waveforms
            # retain request.run.total_steps. Match that preparation horizon;
            # the native schedule below still executes every physical step.
            prepared_steps = execute._progress_chunk_lengths(simulation.num_steps)[0]
            stage(
                "frontend_compile",
                prepared_steps=prepared_steps,
                total_steps=simulation.num_steps,
            )
            # The runner selects CPU preparation for host-reset before the
            # simulation is constructed. Keep that policy during packet capture.
            prepared_simulation = simulation
            stage(
                "preparation_placement",
                original=simulation.setup_device_resolved,
                selected=prepared_simulation.setup_device_resolved,
            )
            self.program = prepared_simulation.compile(
                num_steps=prepared_steps, backend="cuda_streamed", progress=False
            )
            if self.program.config.cuda_storage_axes != (0, 1, 2):
                raise ValueError("Native comparison requires canonical storage axes")
            stage("initial_state")
            self.initial = initial_native_state(
                self.program,
                t=float(simulation.time[0]),
                current_step=0,
                monitor_steps=simulation.num_steps,
            )
            if self.initial.polarization:
                raise ValueError("Native temporal comparison does not support dispersion state")
            ctx = execute.build_step_context(self.program)
            if self.program.coefficients.e_inverse_offdiagonal.size:
                raise ValueError(
                    "Native notebook bridge does not support tensor corrections; refusing to omit them"
                )
            uniform_cpml = rt._uniform_cpml_thickness(ctx)
            stage(
                "capture_packet",
                metric_kind=self.program.config.metric_kind,
                uniform_cpml=uniform_cpml,
            )
            self.moving = (
                self.program.config.metric_kind == "isotropic_uniform"
                and uniform_cpml == 12
            )
            if not self.moving:
                raise ValueError(
                    "This comparison requires the temporal engine; use an isotropic regular grid with CPML12"
                )
            self.fallback_reason = None
            packet = capture_mstep(
                self.program,
                self.initial,
                min(prepared_steps, execute.CUDA_GRAPH_MAX_STEPS),
                materialize_workspaces=False,
            )
            attributes = packet["attributes"]
            self.packet_attributes = {
                k: v.item() if hasattr(v, "item") else v
                for k, v in packet["attributes"].items()
            }
            stage("stream_host_inputs", input_count=len(packet["arguments"]))
            from scripts.cuda_fixture_store import ArrayReader, ArrayWriter, drop_pages, host_chunks

            directory = self.output / "prepared-inputs"
            writer = ArrayWriter(directory)
            for index, value in enumerate(packet["arguments"]):
                stage(
                    "input_transfer",
                    index=index,
                    bytes=int(value.size * value.dtype.itemsize),
                )
                host = np.asarray(jax.device_get(value))
                # A bit-exact zero fact permits sparse staging without writing
                # domain-sized zero field/scratch buffers to disk. Signed zero
                # and every nonzero byte retain the ordinary capture path.
                positive_zero = (
                    not np.any(np.asarray([host.flat[0]], dtype=host.dtype).view(np.uint8))
                    if host.size and all(stride == 0 for stride in host.strides)
                    else all(not np.any(np.ascontiguousarray(block).view(np.uint8))
                             for _, block in host_chunks(host))
                )
                if positive_zero:
                    writer.put(str(index), shape=host.shape, dtype=host.dtype, zero=True)
                else:
                    writer.put(str(index), host)
                del host
            writer.finish()
            reader = ArrayReader(directory)
            self.inputs = [reader[str(i)] for i in range(len(packet["arguments"]))]
            self.input_fingerprints = [
                writer.arrays[str(i)]["sha256"] for i in range(len(self.inputs))
            ]
            for value in self.inputs:
                drop_pages(value)
            if self.host_reset:
                # The native session will own runtime storage. Retain only
                # mapped initial values for metadata/validation, not a second
                # complete JAX field, CPML and DFT state plus trace workspaces.
                import gc

                from scripts.export_cuda_native_fixture import FIELDS_NATIVE

                changes = {name: self.inputs[i] for i, name in enumerate(FIELDS_NATIVE)}
                changes.update(
                    cpml_psi_h_terms=tuple(self.inputs[31:37]),
                    cpml_psi_e_terms=tuple(self.inputs[62:68]),
                    dft_vec_re=self.inputs[120],
                    dft_vec_im=self.inputs[121],
                    dft_weight_sum=self.inputs[122],
                )
                self.initial = self.initial._replace(**changes)
                del packet
                gc.collect()
                stage("release_trace_storage")
            if any(
                self.inputs[86 + 3 * g].shape[0]
                and self.inputs[87 + 3 * g].shape[1] < simulation.num_steps
                for g in range(9)
            ):
                raise ValueError(
                    "Prepared source waveform does not cover the full unchanged notebook run"
                )
            self.resource_plan = dict(
                module.moving_resource_plan(self.inputs, depth=16)
            )
            import subprocess

            free = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            ) * (1 << 20)
            monitor = 113
            persistent = (
                *range(6),
                *range(31, 37),
                *range(62, 68),
                *range(monitor + 7, monitor + 10),
                monitor + 14,
                monitor + 15,
            )
            payload = (
                sum(v.nbytes for v in self.inputs)
                + (
                    0
                    if self.host_reset
                    else sum(self.inputs[i].nbytes for i in persistent)
                )
                + 4
            )
            if payload + 2 * (1 << 30) > free:
                raise MemoryError(
                    f"Native upload admission: payload={payload}, free={free}, reserve=2GiB"
                )
            stage("native_upload")
            self.session = module.NativeSession(
                self.inputs,
                attributes,
                preserve_inputs=not self.host_reset,
                kind="mstep",
                initialize_scratch=False,
            )
            for value in self.inputs:
                drop_pages(value)
            stage("ready")
        prepare_s = time.perf_counter() - prepare_start
        steps = simulation.num_steps
        outer = (
            list(execute._progress_chunk_lengths(steps))
            if kwargs.get("progress", False)
            else [steps]
        )
        limit = execute.CUDA_GRAPH_MAX_STEPS
        # Match main's bounded internal graph calls. Capturing an entire
        # 1,525-step progress block would manufacture a large startup penalty.
        chunks = [
            n
            for length in outer
            for n in (
                [limit] * (length // limit)
                + ([length % limit] if length % limit else [])
            )
        ]
        reset_s = 0.0
        if self.host_reset:
            self.memory.phase = "host_reset"
            reset_started = time.perf_counter()
            self.session.reset()
            reset_s = time.perf_counter() - reset_started
        self.memory.phase = "engine"
        start = time.perf_counter()
        timing = self.session.run_moving(chunks, 16)
        metadata = self.session.metadata()
        if (
            timing.get("algorithm") != "asymmetric_moving"
            or timing.get("moving_macro_count", 0) <= 0
            or metadata.get("moving_depth") != 16
        ):
            raise RuntimeError("The temporal schedule did not execute")
        engine_wall_s = time.perf_counter() - start
        self.memory.phase = "result_decode"
        download_start = time.perf_counter()
        clocks = self.session.clocks()
        dft = 30
        monitor = 113
        changes = {}
        for offset, name in enumerate(("dft_vec_re", "dft_vec_im", "dft_weight_sum")):
            template = self.inputs[monitor + 7 + offset]
            value = np.empty(template.shape, template.dtype)
            self.session.download_into(dft + offset, value)
            changes[name] = value
        changes.update(
            t=np.asarray(clocks["time"], np.float32),
            current_step=np.asarray(clocks["current_step"], np.int32),
        )
        # Monitor decoding consumes DFT buffers only. Read full field/CPML
        # state later, one leaf at a time, solely for numerical validation.
        self.last_state = self.initial._replace(**changes)
        download_s = time.perf_counter() - download_start
        # Compare synchronized host-wall execution against main's performance
        # timer. Keep CUDA-event time separately rather than mixing clock types.
        engine_s = engine_wall_s
        result = SimulationResults.from_run(
            simulation,
            runtime_fields=self.program.grid,
            monitor_results=execute._decode_monitor_results(
                simulation, self.program, self.last_state
            ),
            completed_steps=int(self.last_state.current_step),
            store_full_materials=False,
            source_launch_powers=execute._compiled_source_launch_powers(
                self.program, len(simulation.sources)
            ),
            performance=execute._performance_for(
                self.program, runtime_s=engine_s, steps=steps
            ),
        )
        return result, dict(
            prepare_s=prepare_s,
            host_reset_s=reset_s,
            download_s=download_s,
            engine_wall_s=engine_wall_s,
            device_event_s=timing["cuda_ms"] / 1000,
            timing=timing,
            metadata=metadata,
            chunks=chunks,
            selected_method="moving16",
            fallback_reason=self.fallback_reason,
        )

    def export(self, namespace, finish=False):
        import jax

        from scripts.export_cuda_native_fixture import flatten_state

        if not hasattr(self, "notebook_wall_s"):
            self.notebook_wall_s = time.perf_counter() - self.started
            self.cold_memory = self.memory.summary()
            self.memory.phase = "validation_export"
            raw = namespace["sim_data"].renormalize(None)
            if "benchmark_observables" in namespace:
                arrays = {k: np.asarray(v) for k, v in namespace["benchmark_observables"].items()}
            else:
                arrays = dict(
                    raw_flux_through=np.asarray(raw["flux_through"].flux),
                    raw_flux_cross=np.asarray(raw["flux_cross"].flux),
                    through=np.asarray(namespace["T_through"]),
                    cross=np.asarray(namespace["T_cross"]),
                    neffs=np.asarray(namespace["modes"].neffs),
                    freqs=np.asarray(namespace["freqs"]),
                    launched_power=np.asarray(namespace["source_power"]),
                )
                arrays.update({"raw_field_" + k: np.asarray(v)
                               for k, v in raw["field"].dft_fields.items()})
            assert all(np.isfinite(v).all() for v in arrays.values())
            np.savez_compressed(self.output / "arrays.npz", **arrays)
            from scripts.cuda_fixture_store import ArrayWriter, drop_pages
            from scripts.export_cuda_native_fixture import state_output_mapping

            writer = ArrayWriter(self.output / "state")
            for name, value in flatten_state(self.last_state):
                mapping = state_output_mapping(name)
                if (
                    self.session
                    and mapping["kind"] in ("field", "output")
                    and not name.startswith("dft_")
                ):
                    index = mapping.get("index", mapping.get("component"))
                    temporary = self.output / "validation-leaf.npy"
                    host = np.lib.format.open_memmap(
                        temporary, mode="w+", dtype=value.dtype, shape=value.shape
                    )
                    self.session.download_into(index, host)
                    writer.put(name, host, require_finite=True)
                    drop_pages(host)
                    del host
                    temporary.unlink()
                else:
                    host = np.asarray(jax.device_get(value))
                    writer.put(name, host, require_finite=True)
                    del host
            writer.finish()
            self.memory.phase = "notebook"

            self.last_state = None
            simulation = namespace["sim"]
            self.info = dict(
                grid_shape=list(simulation.grid.shape),
                steps=simulation.num_steps,
                metric_kind=simulation.grid.metric_kind,
                frequencies=len(namespace["freqs"]),
                jax_version=jax.__version__,
                devices=[d.device_kind for d in jax.devices()],
            )
        result = dict(
            variant=self.variant,
            notebook_wall_s=self.notebook_wall_s,
            runs=self.runs,
            memory=self.memory.summary(),
            cold_notebook_memory=self.cold_memory,
            **self.info,
        )
        measured = [x for x in self.memory.rows if x["phase"] != "validation_export"]
        result["memory"]["peak_host_rss_bytes"] = max(x["rss_bytes"] for x in measured)
        result["memory"]["peak_gpu_process_bytes"] = max(
            x["gpu_process_bytes"] for x in measured
        )
        result["memory"]["excludes_validation_export"] = True
        if self.variant != "main":
            result.update(
                native_module=self.module_path,
                native_module_sha256=hashlib.sha256(
                    Path(self.module_path).read_bytes()
                ).hexdigest(),
                packet_attributes=self.packet_attributes,
                resource_plan=self.resource_plan,
                input_fingerprints=self.input_fingerprints,
                temporal_active=self.moving,
                fallback_reason=self.fallback_reason,
                preparation_device_policy="CPU"
                if self.host_reset
                else "notebook default (GPU)",
                preparation_placement_changed=self.host_reset,
                memory_policy="host-reset" if self.host_reset else "preserve",
            )
        (self.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        if finish:
            self.memory.stop_event.set()
            self.memory.thread.join()
            (self.output / "memory-samples.json").write_text(
                json.dumps(self.memory.rows) + "\n"
            )
            if self.session:
                self.session.close()
        return result
