"""One CUDA controller, with a frozen plan and owned native state per simulation."""
from __future__ import annotations

import gc
import os
import threading
import time
import weakref
from contextlib import ExitStack, suppress
from dataclasses import replace
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from beamz.simulation._cuda_abi import CUDA_GRAPH_CACHE

from ._capabilities import _Unsupported
from ._contracts import (
    _CPML,
    _DFT,
    _FIELDS,
    _YEE,
    _Packet,
    _StateContract,
)
from ._plan import _choose, _Plan
from ._prepare import _capture, _compile, _diagnostic_inputs

if TYPE_CHECKING:
    from beamz.simulation.cuda.runtime.native_sharding import _ShardStepper

_SESSIONS: weakref.WeakKeyDictionary[Any, _Session | _ShardStepper] = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()
_DEVICE_LOCKS: dict[str, Any] = {}

def _summarize_diagnostics(diagnostics, domain_measure):
    from beamz.const import EPS_0, MU_0

    energy = np.float32(0)
    for index in (3, 4, 5, 0, 1, 2):
        constant = EPS_0 if index >= 3 else MU_0
        energy = np.float32(energy + np.float32(0.5 * constant) * np.float32(diagnostics["sums"][index]))
    for cross in diagnostics["sums"][6:]:
        energy = np.float32(energy + np.float32(EPS_0) * np.float32(cross))
    return float(energy) * domain_measure, float(np.max(diagnostics["max_fields"])), bool(np.all(diagnostics["finite"]))


class _Session:
    def __init__(self, simulation: Any = "", module: Any = "", *, requested: str = "auto", budget: int = 0, memory_policy: str = "auto", diagnostics: bool = False, compact_monitors: bool = False) -> None:
        import jax

        from beamz.simulation import execute
        from beamz.simulation.model import SimulationState

        self.state_contract = _StateContract(total_steps=simulation.num_steps, steps=simulation.num_steps)
        self.key = (requested, budget, memory_policy); self.lock = threading.RLock()
        self.has_diagnostics = diagnostics
        self._checkpoint_refs = ()
        self._checkpoint_tree = None
        self._completed = 0
        self._temporal_blocks = self._ordinary_tail_steps = 0
        self._on_grid = True
        self._compact_monitors = False
        self._monitor_check_bytes = -1
        if diagnostics and not hasattr(module.NativeSession, "configure_diagnostics"):
            raise _Unsupported("Installed CUDA component lacks native termination diagnostics; rebuild it")
        with jax.default_device(jax.devices("cpu")[0]):
            device = module.device_info()
            self.device_uuid = str(device["uuid"])
            self.device_uuids = (self.device_uuid,)
            effective_budget = min(budget or int(device["total_memory_bytes"]*.9), int(device["free_memory_bytes"]))
            prepared_steps = min(execute._progress_chunk_lengths(simulation.num_steps)[0], 256)
            if budget and not getattr(module,"compiled_coefficient_staging_bytes",0):
                raise _Unsupported("Installed component lacks bounded native coefficient preparation")
            self.program = _compile(simulation, module, steps=prepared_steps, memory_budget=effective_budget)
            config = self.program.config
            self.cuda_options = (
                config.cuda_flags & CUDA_GRAPH_CACHE,
                config.cuda_graph_cache_capacity,
            )
            psi_zero = np.zeros((), dtype=np.float32)
            if compact_monitors and getattr(module, "compiled_compact_dft_components", False):
                self.program = self._pack_monitor_components(self.program)
                self._compact_monitors = any(m.dft_packed_components is not None for m in self.program.monitors)
            fields = {name: np.asarray(getattr(self.program.grid, name.capitalize())) for name in _FIELDS}
            self.initial = SimulationState(**fields,
                cpml_psi_h_terms=tuple(np.broadcast_to(psi_zero, t.slab.shape) for t in self.program.boundary.cpml.h_terms),
                cpml_psi_e_terms=tuple(np.broadcast_to(psi_zero, t.slab.shape) for t in self.program.boundary.cpml.e_terms),
                polarization=(), **execute.monitor_runtime.empty_monitor_values(self.program, num_steps=simulation.num_steps, host_recorders=True, virtual_dft=True),
                t=np.asarray(simulation.time[0], np.float32), current_step=np.asarray(0, np.int32))
            temporal_recorders = getattr(module, "compiled_temporal_recorders", False)
            temporal_tensors = getattr(module, "compiled_temporal_tensor", False)
            depth = 4 if temporal_tensors and self.program.coefficients.e_inverse_offdiagonal.size else 16
            packet = _capture(self.program, self.initial, prepared_steps,
                native_effects=hasattr(module.NativeSession, "configure_effects"),
                clamp_source_tails=getattr(module, "compiled_clamped_source_tails", False),
                ragged_gather=getattr(module, "compiled_ragged_dft_gather", False),
                minimum_recorder_chunk=depth if requested=="temporal" and temporal_recorders else 1)
            tensor, scatters, recorders = packet["effects"]
            effects_bytes = packet["effects_bytes"]
            self._chunk_steps = packet["chunk_steps"]
            self._recorder_specs = tuple(m for m in self.program.monitors if m.recorder_index >= 0)
            self._recording_prefix = ()
            self._shared_recordings = False
            layout = _CPML if len(packet["arguments"]) == _CPML.input_count else _YEE
            values = []
            for i, value in enumerate(packet["arguments"]):
                host = np.asarray(value)
                zero = i in layout.zeroable and host.ndim and (not host.size or
                    (not any(host.strides) and not np.any(np.asarray([host.flat[0]], dtype=host.dtype).view(np.uint8))) or
                    not np.any(np.ascontiguousarray(host).view(np.uint8)))
                values.append(module.ZeroBuffer(list(host.shape), host.dtype.name) if zero else np.ascontiguousarray(host) if host.ndim else host)
            self.packet = _Packet(tuple(values), {k: v.item() if hasattr(v, "item") else v for k,v in packet["attributes"].items()})
            device = module.device_info()
            temporal_reason = ""
            if not getattr(module, "compiled_temporal_boundary_widths", False):
                if self.packet.attributes["boundary_code"] != (12 << 8):
                    temporal_reason = "temporal execution requires symmetric CPML12 without PEC"
                elif min(min(self.packet._buffer(name).shape) for name in _FIELDS) <= 24:
                    temporal_reason = "temporal execution requires a positive interior beyond CPML12"
            groups = tuple(self.packet.arguments[i].shape[0] for i in range(layout.source,layout.monitor,3))
            coincident = int(self.packet.attributes["coincident_source_group_mask"])
            temporal_sources = getattr(module, "compiled_temporal_sources", False)
            if not temporal_sources and any(count>1 and not (coincident>>i)&1 for i,count in enumerate(groups)):
                temporal_reason = "noncoincident sources require ordinary execution"
            if tensor and not temporal_tensors:
                temporal_reason = "off-diagonal electric tensors require ordinary execution"
            elif scatters and not temporal_sources:
                temporal_reason = "irregular sources require ordinary execution"
            if recorders and not temporal_recorders:
                temporal_reason = "installed CUDA component requires ordinary execution for field recorders"
            elif recorders and self._chunk_steps < depth:
                temporal_reason = f"field recorder staging cannot fit a {depth}-step temporal block"
            if diagnostics and tensor and not getattr(module, "compiled_tensor_diagnostics", False):
                raise _Unsupported("Installed CUDA component lacks native tensor energy diagnostics")
            diagnostic_inputs = _diagnostic_inputs(self.program) if diagnostics else None
            diagnostic_bytes = diagnostic_inputs[3] if diagnostic_inputs is not None else 0
            self.plan = _choose(
                self.packet.arguments, module, device, steps=simulation.num_steps,
                requested=requested, budget=budget, temporal_reason=temporal_reason, memory_policy=memory_policy,
                extra_payload_bytes=diagnostic_bytes + effects_bytes,
                recorder_staging_bytes=effects_bytes if recorders and not (tensor or scatters) else 0,
                scatters=scatters, recorders=recorders, tensor=bool(tensor), attributes=self.packet.attributes,
            )
            if self.plan.direct_volume_recording:
                recorders = ()
                effects_bytes = 0
            elif self.plan.boundary_gather_bytes:
                effects_bytes = self.plan.boundary_gather_bytes
            if self.plan.ordinary_general and not getattr(module,"compiled_general_ordinary",False):
                raise _Unsupported("Installed component lacks general native ordinary execution: " + self.plan.reason)
            if self.plan.payload_bytes + self.plan.reserve_bytes > int(device["free_memory_bytes"]):
                raise MemoryError("The frozen CUDA plan does not fit currently free GPU memory")
            self.initial = self.initial._replace(**self.packet._state_inputs())
            del packet, fields, values
            gc.collect()
            native_attributes = dict(self.packet.attributes)
            if self.plan.schedule == "temporal" and getattr(module, "compiled_rolling_tiles", False):
                native_attributes.update(moving_macros=self.plan.macro_slab_count, rolling_band=self.plan.rolling_band)
            self.native = module.NativeSession(self.packet.arguments, native_attributes, preserve_inputs=False,
                kind="mstep" if self.plan.schedule=="temporal" else "generated", initialize_scratch=False)
            try:
                if tensor or scatters or recorders:
                    tensor_inputs = tuple(
                        module.ZeroBuffer(a.shape, "float32") if i >= 5
                        else np.ascontiguousarray(a) if a.ndim else np.asarray(a)
                        for i, a in enumerate(tensor)
                    )
                    acquisition = {"at_boundaries": True} if self.plan.boundary_gather_bytes else {}
                    allocated = self.native.configure_effects(tensor_inputs, scatters, recorders, self._chunk_steps, **acquisition)
                    if allocated != effects_bytes:
                        raise RuntimeError("Native CUDA effect allocations differ from the admitted plan")
                if diagnostic_inputs is not None:
                    self._domain_measure = diagnostic_inputs[2]
                    report = self.native.configure_diagnostics(*diagnostic_inputs[:3])
                    self.plan = replace(self.plan, payload_bytes=self.plan.payload_bytes - diagnostic_bytes + int(report["gpu_bytes"]))
            except BaseException:
                self.native.close()
                raise

    @staticmethod
    def _pack_monitor_components(program):
        spectral = tuple(m for m in program.monitors if m.dft_enabled and m.freq_count and m.dft_point_count)
        components = {
            m.monitor_index: tuple(int(c) for c in np.flatnonzero(np.asarray(m.dft_component_mask) > 0))
            for m in spectral
        }
        if not all(components.values()) or all(len(c) == 6 for c in components.values()):
            return program
        offset = 0
        monitors = []
        for monitor in program.monitors:
            packed = components.get(monitor.monitor_index)
            if packed is not None:
                monitor = replace(monitor, dft_value_offset=offset, dft_packed_components=packed)
                offset += len(packed) * monitor.freq_count * monitor.dft_point_count
            monitors.append(monitor)
        return replace(program, monitors=tuple(monitors))

    def _advance(self, steps, *, snapshot=True):
        """Advance owned state in place; download only cumulative observations."""
        start = time.perf_counter()
        if self.plan.schedule == "temporal":
            # Direct full frames reuse the staggered field at its acquisition
            # boundary. Native batching owns the internal graph slabs.
            options = {"scheduling_sms": self.plan.scheduling_sms} if self.plan.scheduling_sms else {}
            interval = min(s.record_interval for s in self._recorder_specs) if self.plan.direct_volume_recording else 0
            step, end = self._completed, self._completed + steps
            while step < end:
                length = end - step
                if interval:
                    length = min(length, interval - step % interval)
                report = self.native.run_moving([length], self.plan.depth, self.plan.raw_slots, **options)
                if report is not None:
                    self._temporal_blocks += int(report["moving_macro_count"])
                    self._ordinary_tail_steps += int(report["ordinary_tail_steps"])
                step += length
                if interval and step % interval == 0:
                    self._download_volume_frames(step)
        else:
            chunk_steps = getattr(self, "_chunk_steps", 256)
            complete, tail = divmod(steps, chunk_steps)
            chunks = [chunk_steps] * complete + ([tail] if tail else [])
            self.native.run_schedule(chunks)
        elapsed = time.perf_counter() - start
        self._completed += steps
        if not snapshot:
            return None, elapsed
        clocks = self.native.clocks()
        if int(clocks["current_step"]) != self._completed:
            raise RuntimeError("CUDA execution did not complete the requested physical horizon")
        return self._snapshot(clocks), elapsed

    def _download_volume_frames(self, step):
        for spec in self._recorder_specs:
            count, remainder = divmod(step, spec.record_interval)
            if remainder:
                continue
            for name, sign, index in zip(spec.canonical_components, spec.component_signs, spec.field_buffer_indices, strict=True):
                component = _FIELDS.index(name.lower())
                history = self.initial.recorded_fields[index]
                frame = history[count - 1].reshape(self.packet.arguments[component].shape)
                self.native.download_into(component, frame)
                if sign < 0:
                    np.negative(frame, out=frame)

    def _restore_recordings(self, state):
        self._recording_prefix = ()
        if state is None or not self._recorder_specs:
            return
        step = int(state.current_step)
        if len(state.recorded_counts) != len(self._recorder_specs):
            raise _Unsupported("Continuation requires a compatible recorder history")
        prefix = []
        for spec in self._recorder_specs:
            rid = spec.recorder_index
            count = int(state.recorded_counts[rid])
            indices = np.asarray(state.recorded_steps[rid])[:count]
            times = np.asarray(state.recorded_times[rid])[:count]
            interval = int(spec.record_interval)
            if count != step // interval or not np.array_equal(indices, np.arange(1, count + 1) * interval):
                raise _Unsupported("Continuation requires a complete recorder prefix")
            for index in spec.field_buffer_indices:
                dst = self.initial.recorded_fields[index]
                src = np.asarray(state.recorded_fields[index])
                if src.dtype != dst.dtype or src.shape[1:] != dst.shape[1:] or src.shape[0] < count:
                    raise _Unsupported("Continuation requires matching recorder frame supports")
                np.copyto(dst[:count], src[:count])
            prefix.append(times.copy())
        self._recording_prefix = tuple(prefix)

    def _reset_recording_destination(self):
        if not getattr(self, "_shared_recordings", False):
            return
        histories = tuple(np.empty_like(value) for value in self.initial.recorded_fields)
        if not self.plan.direct_volume_recording:
            self.native.rebind_recordings(tuple(
                histories[i].reshape(histories[i].shape[0], prod(histories[i].shape[1:]))
                for spec in self._recorder_specs for i in spec.field_buffer_indices
            ))
        self.initial = self.initial._replace(recorded_fields=histories)
        self._shared_recordings = False

    def _recording_result_options(self):
        self._shared_recordings = bool(getattr(self, "_recorder_specs", ())) and hasattr(self.native, "rebind_recordings")
        return {"owned_recordings": True} if self._shared_recordings else {}

    def _recording_state(self, clocks, *, checkpoint):
        if not self._recorder_specs:
            return {}
        step, elapsed = int(clocks["current_step"]), int(clocks["elapsed_steps"])
        origin, dt = np.float32(clocks["observation_origin"]), np.float32(self.program.config.dt)
        frames, steps, times, counts = [], [], [], []
        for spec in self._recorder_specs:
            interval = int(spec.record_interval)
            count = step // interval
            indices = np.arange(1, count + 1, dtype=np.int32) * interval
            physical = origin + (indices - (step - elapsed)).astype(np.float32) * dt
            if self._recording_prefix:
                prefix = self._recording_prefix[spec.recorder_index]
                physical[:prefix.size] = prefix
            for index in spec.field_buffer_indices:
                history = self.initial.recorded_fields[index]
                if checkpoint:
                    detached = np.zeros_like(history)
                    detached[:count] = history[:count]
                    history = detached
                frames.append(history)
            if checkpoint:
                capacity = self.initial.recorded_steps[spec.recorder_index].size
                indices = np.pad(indices, (0, capacity - count))
                physical = np.pad(physical, (0, capacity - count))
            steps.append(indices)
            times.append(physical)
            counts.append(np.asarray(count, np.int32))
        return {
            "recorded_fields": tuple(frames), "recorded_steps": tuple(steps),
            "recorded_times": tuple(times), "recorded_counts": tuple(counts),
        }

    def _snapshot(self, clocks, *, template=None, fields=False, device_checkpoint=False):
        """Detach monitored values, and physical state only for public checkpoints."""
        layout = self.packet._layout
        psi = (
            tuple(
                tuple(f"{phase}_psi_{i}" for i in range(len(getattr(self.initial, scope))))
                for phase, scope in (("h", "cpml_psi_h_terms"), ("e", "cpml_psi_e_terms"))
            ) if fields else ()
        )
        names = (*_FIELDS, *(name for group in psi for name in group), *_DFT) if fields else _DFT
        detached = None
        if device_checkpoint:
            available = self.plan.memory_budget_bytes - self.plan.reserve_bytes - self.plan.payload_bytes
            detached = self.native.download_device_state(max(0, available))
        if detached is not None:
            values = dict(zip(names, detached, strict=True))
        else:
            values = {}
            for name in names:
                buffer = self.packet._buffer(name)
                if fields:
                    # Align the CPU snapshot so JAX can share the backing storage.
                    # The NumPy owner is local and no writable alias escapes.
                    storage = np.empty(buffer.nbytes + 64, np.uint8)
                    start = -storage.ctypes.data % 64
                    value = storage[start:start + buffer.nbytes].view(buffer.dtype).reshape(buffer.shape)
                else:
                    value = np.empty(buffer.shape, buffer.dtype)
                self.native.download_into(layout.outputs[name], value)
                values[name] = value
        if fields:
            values["cpml_psi_h_terms"], values["cpml_psi_e_terms"] = (
                tuple(values.pop(name) for name in group) for group in psi
            )
        if getattr(self, "_recorder_specs", ()):
            values.update(self._recording_state(clocks, checkpoint=fields))
        return (self.initial if template is None else template)._replace(
            **values,
            t=np.asarray(clocks["time"], np.float32),
            current_step=np.asarray(clocks["current_step"], np.int32),
        )

    def _restore_checkpoint(self, simulation, state, contract):
        import jax

        from beamz.simulation.execute import runtime_inputs

        step = contract.initial_step
        time_value = np.float32(state.t)
        t0 = np.float32(simulation.time[0])
        grid_time = t0 + np.float32(simulation.dt) * np.float32(step)
        tolerance = np.finfo(np.float32).eps * max(abs(grid_time), np.finfo(np.float32).tiny)
        if not np.isfinite(time_value):
            raise _Unsupported("Native continuation requires a finite state time")
        self._on_grid = bool(abs(time_value - grid_time) <= tolerance)
        observation_origin, observation_step = (t0, step) if self._on_grid else (time_value, 0)
        # Reuse compatibility semantics for changed boundaries or monitor plans.
        with jax.default_device(jax.devices("cpu")[0]):
            restored = runtime_inputs(self.program, state, monitor_steps=simulation.num_steps - step)
        for name, b in self.packet._state_inputs().items():
            a = getattr(restored, name)
            xa, xb = (a, b) if isinstance(a, tuple) else ((a,), (b,))
            if len(xa) != len(xb) or any(
                tuple(x.shape) != tuple(y.shape) or np.dtype(x.dtype) != np.dtype(y.dtype)
                for x, y in zip(xa, xb, strict=True)
            ):
                raise _Unsupported(f"Explicit state {name} dtype or shape requires the CUDA interoperability binding")
        leaves = (
            *(getattr(restored, name) for name in _FIELDS),
            *restored.cpml_psi_h_terms, *restored.cpml_psi_e_terms,
            *(getattr(restored, name) for name in _DFT),
        )
        self._restore_recordings(restored)
        self.native.restore_state(tuple(np.ascontiguousarray(value) for value in leaves),
                                  float(observation_origin), step, observation_step)
        self._completed = step
        return restored

    def _execute(
        self, total, *, progress, snapshot=True, progress_callback=None, wall_start=None
    ):
        from beamz._helpers import _finish_inline_progress, _print_inline_progress
        from beamz.simulation import execute
        from beamz.simulation.results import RunProgress

        completed = 0
        elapsed = 0.0
        state = None
        chunked = progress or progress_callback is not None
        alignment = (
            self.plan.depth * self.plan.macro_slab_count
            if chunked and self.plan.schedule == "temporal" else 1
        )
        outer = execute._progress_chunk_lengths(total, alignment=alignment) if chunked else (total,)
        final_step = self._completed + total
        callback_start = time.perf_counter() if wall_start is None else wall_start
        try:
            for length in outer:
                state, duration = self._advance(length, snapshot=snapshot and completed + length == total)
                elapsed += duration
                completed += length
                if progress:
                    _print_inline_progress(completed, total, label="Running simulation")
                if progress_callback is not None:
                    clock = self._clock_state() if state is None else state
                    progress_callback(
                        RunProgress(
                            step=int(clock.current_step),
                            total_steps=final_step,
                            time=float(clock.t),
                            elapsed=time.perf_counter() - callback_start,
                        )
                    )
        finally:
            if progress:
                _finish_inline_progress()
        return state, elapsed

    def _field_diagnostics(self):
        return _summarize_diagnostics(self.native.field_diagnostics(), self._domain_measure)

    def _configure_monitor_checks(self, names):
        if not hasattr(self.native, "_configure_monitor_diagnostics"):
            return -1
        ranges = []
        for spec in self.program.monitors:
            if spec.name not in names:
                continue
            width = spec.freq_count * spec.dft_point_count
            slots = (
                range(len(spec.dft_packed_components)) if spec.dft_packed_components is not None
                else np.flatnonzero(np.asarray(spec.dft_component_mask) > 0)
            )
            ranges.extend(
                (spec.dft_value_offset + slot * width, spec.freq_count, spec.dft_point_count)
                for slot in slots
            )
        outputs = self.packet._layout.outputs
        remaining = self.plan.memory_budget_bytes - self.plan.reserve_bytes - self.plan.payload_bytes
        return self.native._configure_monitor_diagnostics(
            outputs["dft_vec_re"], outputs["dft_vec_im"], sorted(ranges),
            min(256 << 20, max(0, remaining)),
        )

    def _monitor_check(self):
        return self.native._monitor_diagnostics()

    def _clock_state(self):
        clocks = self.native.clocks()
        if int(clocks["current_step"]) != self._completed:
            raise RuntimeError("CUDA execution did not complete the requested physical horizon")
        return self.initial._replace(
            current_step=np.asarray(clocks["current_step"], np.int32),
            t=np.asarray(clocks["time"], np.float32),
        )

    def _final_observations(self):
        return self._snapshot(self.native.clocks())

    def _observation_report(self):
        layout = self.packet._layout
        gathers = self.packet.arguments[layout.monitor:layout.monitor + 2]
        return {
            "temporal_blocks": self._temporal_blocks,
            "ordinary_tail_steps": self._ordinary_tail_steps,
            "dft_storage": "compact" if self._compact_monitors else "dense",
            "dft_accumulator_bytes": sum(self.packet._buffer(name).nbytes for name in _DFT),
            "dft_gather_bytes": sum(value.nbytes for value in gathers),
            "dft_gather_layout": "ragged" if len(gathers[0].shape) == 2 else "rectangular",
            "source_waveform_bytes": sum(
                self.packet.arguments[i].nbytes for i in range(layout.source + 1, layout.monitor, 3)
            ),
        }

    def _run(self, simulation: Any = "", *, progress: bool = False, store_full_materials: bool = False, performance: bool = True, termination=None, wall_start=None, progress_callback=None) -> Any:
        from beamz.simulation import execute
        from beamz.simulation.results import SimulationResults

        with self.lock:
            self._reset_recording_destination()
            self._temporal_blocks = self._ordinary_tail_steps = 0
            self.native.reset(); self._completed = 0
            if getattr(self, "_recorder_specs", ()):
                self._restore_recordings(None)
            self._checkpoint_refs = ()
            if termination is not None:
                try:
                    result = execute.run_until_terminated(simulation, termination, progress=progress,
                        store_full_materials=store_full_materials, performance=performance, sharding=None,
                        backend="cuda", _native=self, _wall_start=wall_start,
                        progress_callback=progress_callback)
                finally:
                    if hasattr(self.native, "_clear_monitor_diagnostics"):
                        self.native._clear_monitor_diagnostics()
                object.__setattr__(result, "_cuda_plan", MappingProxyType({**self.plan._report(), "initial_step":0, "final_step":self._completed,
                    **self._observation_report(),
                    "monitor_convergence": "device" if self._monitor_check_bytes >= 0 else "host",
                    "monitor_convergence_gpu_bytes": max(0, self._monitor_check_bytes)}))
                return result
            total = self.state_contract.steps
            state, elapsed = self._execute(total, progress=progress,
                progress_callback=progress_callback, wall_start=wall_start)
            stats = execute._performance_for(self.program,runtime_s=elapsed,steps=total) if performance else None
            result = SimulationResults.from_run(simulation,runtime_fields=self.program.grid,
                monitor_results=execute._decode_monitor_results(simulation,self.program,state,**self._recording_result_options()),completed_steps=total,
                store_full_materials=store_full_materials,source_launch_powers=execute._compiled_source_launch_powers(self.program,len(simulation.sources)),performance=stats)
            object.__setattr__(result,"_cuda_plan",MappingProxyType({**self.plan._report(), **self._observation_report(), "initial_step":self.state_contract.initial_step, "final_step":self.state_contract._stop}))
            if stats is not None: execute._print_performance(stats)
            return result

    def _run_segment(self, simulation, contract, state, *, state_only, progress, store_full_materials, performance, progress_callback=None, wall_start=None):
        import jax

        from beamz.simulation import execute
        from beamz.simulation.results import SimulationResults, SimulationRun

        self._reset_recording_destination()
        self._temporal_blocks = self._ordinary_tail_steps = 0
        leaves, tree = jax.tree_util.tree_flatten(state)
        # Only snapshots created by this owner can match by object identity.
        # Reusing a prior branch or a modified leaf restores it into our bank.
        resident = (
            bool(self._checkpoint_refs) and self._on_grid and contract.initial_step == self._completed
            and tree == self._checkpoint_tree and len(leaves) == len(self._checkpoint_refs)
            and all(ref() is value and not value.is_deleted()
                    for ref, value in zip(self._checkpoint_refs, leaves, strict=True))
        )
        self._checkpoint_refs = ()
        if state is None:
            self.native.reset(); self._completed = 0; self._on_grid = True
            self._restore_recordings(None)
            template = self.initial
        elif resident:
            template = state
        else:
            template = self._restore_checkpoint(simulation, state, contract)
        _, elapsed = self._execute(contract.steps, progress=progress, snapshot=False,
            progress_callback=progress_callback, wall_start=wall_start)
        clocks = self.native.clocks()
        if int(clocks["current_step"]) != contract._stop:
            raise RuntimeError("CUDA execution did not complete the requested physical horizon")
        device_checkpoint = False
        if state_only and not self._recorder_specs and hasattr(self.native, "download_device_state"):
            # CUDA can run independently while JAX is configured CPU-only.
            with suppress(RuntimeError):
                device_checkpoint = bool(jax.devices("gpu"))
        snapshot = self._snapshot(clocks, template=template, fields=True, device_checkpoint=device_checkpoint)
        # Keep existing CUDA-backed leaves and create uncommitted CPU leaves
        # for detached host data and scalar clocks. JAX can place those scalars
        # with device fields when an application consumes the combined state.
        with jax.default_device(jax.devices("cpu")[0]):
            snapshot = jax.tree_util.tree_map(
                lambda v: v if isinstance(v, jax.Array) else jax.device_put(v, may_alias=True),
                snapshot,
            )
        leaves, self._checkpoint_tree = jax.tree_util.tree_flatten(snapshot)
        self._checkpoint_refs = tuple(weakref.ref(value) for value in leaves)
        if state_only:
            return snapshot
        stats = execute._performance_for(self.program, runtime_s=elapsed, steps=contract.steps) if performance else None
        result = SimulationResults.from_run(simulation, runtime_fields=self.program.grid,
            monitor_results=execute._decode_monitor_results(simulation, self.program, snapshot),
            completed_steps=contract._stop, store_full_materials=store_full_materials,
            source_launch_powers=execute._compiled_source_launch_powers(self.program, len(simulation.sources)),
            performance=stats)
        object.__setattr__(result, "_cuda_plan", MappingProxyType({**self.plan._report(), **self._observation_report(),
            "state_ownership": contract.ownership, "initial_step": contract.initial_step,
            "final_step": contract._stop, "reused_resident_checkpoint": resident}))
        if stats is not None:
            execute._print_performance(stats)
        return SimulationRun(results=result, state=snapshot)

    def _close(self) -> None:
        with self.lock: self.native.close()


def _retire_overlapping(simulation, uuids):
    """Retire other resident owners before admitting work on their devices."""
    for owner, session in tuple(_SESSIONS.items()):
        if owner is not simulation and not uuids.isdisjoint(session.device_uuids):
            _clear(owner)


def _clear(simulation: Any = "") -> None:
    with _LOCK:
        if simulation != "":
            session = _SESSIONS.pop(simulation, None)
            if session is not None: session._close()
        else:
            for session in tuple(_SESSIONS.values()): session._close()
            _SESSIONS.clear()


def _dispatch(simulation: Any = "", *, backend: str = "auto", schedule: str | None = "", progress: bool = False, store_full_materials: bool = False, performance: bool = True, interop_reason: str = "", termination=None, _wall_start=None, segment: _StateContract | None = None, state=None, state_only: bool = False, progress_callback=None) -> tuple[Any, _Plan]:
    from beamz.simulation import backend as backends

    wall_start = time.perf_counter() if _wall_start is None else _wall_start
    if termination is not None:
        from beamz.simulation.model import AutoTermination
        if not isinstance(termination, AutoTermination):
            raise TypeError("termination must be an AutoTermination instance or None.")
    name = backends.normalize_backend(backend)
    from beamz.simulation.cuda.mstep.schedule import normalize_schedule_request
    normalized, _ = normalize_schedule_request(name, schedule or None, 1)
    budget = int(os.getenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES","0")) if name != "jax" else 0
    if budget < 0: raise ValueError("CUDA memory budget must be nonnegative")
    if interop_reason: _clear(simulation)
    if name == "jax" or (name == "auto" and os.getenv("JAX_PLATFORMS", "") == "cpu"):
        return None, _Plan(binding="jax",reason="explicit CPU/JAX policy")
    # Compile validates this policy after capability admission; auto/JAX scenes
    # must not fail because of a CUDA-only environment option.
    memory_policy = os.getenv("BEAMZ_CUDA_MEMORY_POLICY", "auto").strip().lower()
    if memory_policy == "capacity" and normalized == "mstep":
        raise backends.CudaBackendUnavailable("Temporal CUDA requires extra workspace; memory policy 'capacity' requires ordinary execution")
    if interop_reason:
        return None, _Plan(binding="ffi",reason=interop_reason)
    if not simulation.is_3d:
        return None, _Plan(binding="ffi",reason="reduced-dimensional physics uses the CUDA interoperability binding")
    try: module = backends._load_extension()
    except (ImportError,OSError): return None, _Plan(binding="ffi",reason="installed component has no coherent native runtime")
    if not getattr(module,"coherent_backend_version",0):
        return None, _Plan(binding="ffi",reason="legacy component: preserve its registered CUDA interface")
    try:
        backends._require_cuda_abi(module)
    except backends.CudaBackendUnavailable as error:
        if name == "auto":
            return None, _Plan(binding="jax", reason=str(error))
        raise
    if segment is not None:
        if not hasattr(module.NativeSession, "restore_state"):
            return None, _Plan(binding="ffi", reason="installed component lacks native continuation")
        if state is not None:
            fields = tuple(getattr(state, name) for name in _FIELDS)
            if state.polarization or any(
                value.dtype != np.dtype("float32")
                or len(getattr(getattr(value, "sharding", None), "device_set", ())) > 1
                for value in fields
            ):
                return None, _Plan(binding="ffi", reason="continuation requires unsharded FP32 Yee state")
    if backends._has_cuda_legacy_options(native_flags=CUDA_GRAPH_CACHE):
        _clear(simulation)
        return None, _Plan(binding="ffi", reason="explicit legacy CUDA options")
    cuda_options = (
        backends.cuda_flags_from_env() & CUDA_GRAPH_CACHE,
        backends.cuda_graph_cache_capacity_from_env(),
    )
    try: device = module.device_info()
    except RuntimeError as error:
        if name == "auto": return None, _Plan(binding="jax",reason="native CUDA device unavailable")
        raise backends.CudaBackendUnavailable(str(error)) from error
    requested = {"streamed":"ordinary", "mstep":"temporal", "auto":"auto"}[normalized]
    with _LOCK:
        device_lock = _DEVICE_LOCKS.setdefault(str(device["uuid"]), threading.RLock())
    with device_lock, ExitStack() as ownership:
        with _LOCK:
            session = cast("_Session | None", _SESSIONS.get(simulation))
            if requested == "auto" and segment is not None and segment.steps < 16 and (
                session is None or session.key != (requested, budget, memory_policy)
                or session.cuda_options != cuda_options
                or session.device_uuid != str(device["uuid"])
                or session._compact_monitors
            ):
                # A short segment can use an already resident plan, but should
                # never allocate new temporal workspace for an all-ordinary tail.
                requested = "ordinary"
            if session is not None and (
                session.key != (requested, budget, memory_policy)
                or session.cuda_options != cuda_options
                or session.device_uuid != str(device["uuid"])
                or (termination is not None and not session.has_diagnostics)
                or (segment is not None and session._compact_monitors)
            ):
                session._close(); del _SESSIONS[simulation]; session = None
            if session is None:
                # Keep one reusable owner per device, retiring previous idle storage
                # before constructing another scene. Its device lock covers execution.
                _retire_overlapping(simulation, {str(device["uuid"])})
                from beamz.simulation.compile import _PROGRAM_CACHE
                existing = set(_PROGRAM_CACHE)
                try: session = _Session(simulation,module,requested=requested,budget=budget,memory_policy=memory_policy,
                    diagnostics=termination is not None, compact_monitors=segment is None)
                except _Unsupported as error:
                    if requested == "temporal" and segment is None:
                        raise backends.CudaBackendUnavailable(str(error)) from error
                    return None, _Plan(binding="ffi",reason=str(error))
                finally:
                    # The Session owns the lowered program; retire transient unlowered
                    # coefficient preparation on success as well as rejection.
                    for key in tuple(_PROGRAM_CACHE):
                        if key.native_preparation and key not in existing: _PROGRAM_CACHE.pop(key)
                _SESSIONS[simulation] = session
            # Reserve the Session before dropping cache admission ownership, so
            # concurrent explicit clearing cannot close it before _run starts.
            ownership.enter_context(session.lock)
        if segment is not None:
            try:
                result = session._run_segment(simulation, segment, state, state_only=state_only,
                    progress=progress, store_full_materials=store_full_materials, performance=performance,
                    progress_callback=progress_callback, wall_start=wall_start)
            except _Unsupported as error:
                return None, _Plan(binding="ffi", reason=str(error))
            return result, session.plan
        return session._run(
            simulation, progress=progress, store_full_materials=store_full_materials,
            performance=performance, termination=termination, wall_start=wall_start,
            progress_callback=progress_callback,
        ), session.plan
