"""Predictive layout selection for large, ordinary RTX3090 CPML programs.

The default uses geometry alone, without executing a calibration workload.
Explicit calibration considers existing storage orders and shell tiles on a
private state; precision, physical shape and CPML stay unchanged in either mode.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import operator
import os
import statistics
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import numpy as np

from beamz.simulation import _cuda_abi as abi

_SCHEMA = 2
_AXES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
_TILES = ("64x4", "32x8")
_REPORTS = OrderedDict()
_TUNING_LOCK = threading.RLock()
_GEOMETRY_MODEL = "cpml12-queue-v1"
_MINIMUM_WORK_REDUCTION = 0.25


def tuning_policy_from_env():
    """Include manual overrides and tuning policy in the program cache key."""
    mode = os.environ.get("BEAMZ_CUDA_AUTOTUNE", "auto").strip().lower()
    if mode not in {"auto", "off", "calibrate", "refresh"}:
        raise ValueError("BEAMZ_CUDA_AUTOTUNE must be auto, off, calibrate, or refresh")
    overrides = tuple(
        sorted(
            (key, value)
            for key, value in os.environ.items()
            if key.startswith("BEAMZ_CUDA_")
            and key
            not in {
                "BEAMZ_CUDA_AUTOTUNE",
                "BEAMZ_CUDA_TUNING_CACHE",
                "BEAMZ_CUDA_MEMORY_POLICY",
            }
        )
    )
    return mode, os.environ.get("BEAMZ_CUDA_TUNING_CACHE", ""), overrides


def tuning_report(program):
    """Return a detached diagnostic report for a selected compiled program."""
    entry = _REPORTS.get(id(program))
    return json.loads(json.dumps(entry[1])) if entry and entry[0] is program else None


def _remember(program, report):
    _REPORTS[id(program)] = (program, report)
    _REPORTS.move_to_end(id(program))
    while len(_REPORTS) > 8:
        _REPORTS.popitem(last=False)
    return program


def _apply(program, choice):
    axes = tuple(choice["axes"])
    tile = choice["shell_tile"]
    if (
        any(type(axis) is not int for axis in axes)
        or axes not in _AXES
        or tile not in _TILES
    ):
        raise ValueError("Invalid cached CUDA layout")
    flags = program.config.cuda_flags & ~(abi.CUDA_SHELL32X8 | abi.CUDA_SHELL32X4)
    if tile == "32x8":
        flags |= abi.CUDA_SHELL32X8
    return replace(
        program,
        config=replace(program.config, cuda_storage_axes=axes, cuda_flags=flags),
    )


def _queue_work(shape):
    """Count 64x4 queue thread slots per phase, including inactive lanes.

    Mirrors LaunchCpmlQueueTile in cuda/src/update.cu for CPML12. Across
    either Yee triplet, the per-axis maximum extent is n+1 and minimum is n.
    The queue partitions z faces, then y faces, then x faces, then the core.
    This is a geometric work proxy, not a prediction of memory traffic or time;
    native fusion can replace the core with a different kernel.
    """
    z, y, x = shape
    iz, iy, ix = z - 24, y - 24, x - 24

    def tiles(n, width):
        return (n + width - 1) // width

    z_blocks = tiles(x + 1, 64) * tiles(y + 1, 4) * 25
    y_blocks = tiles(x + 1, 64) * tiles(25, 4) * iz
    x_blocks = tiles(25, 64) * tiles(iy, 4) * iz
    core_blocks = tiles(ix, 64) * tiles(iy, 4) * iz
    shell_slots = 256 * (z_blocks + y_blocks + x_blocks)
    core_slots = 256 * core_blocks
    return dict(
        shell_slots=shell_slots,
        core_slots=core_slots,
        total_slots=shell_slots + core_slots,
        slots_per_cell=(shell_slots + core_slots) / (z * y * x),
    )


def predict_layout(shape):
    """Estimate an RTX3090 CPML12 storage order from logical (z,y,x) extents.

    Pure host arithmetic: no device queries, trials or persistent profiles.
    Only right-handed cyclic permutations are considered. Require a 25% queue
    work reduction to change layout: smaller proxy wins have not reliably
    predicted timing wins in the recorded shape study. This margin is a policy
    choice, not a measured speedup or statistical confidence bound. Keep 64x4:
    32x8 has no established benefit after repairing the narrow-x layout.

    Runtime selection separately checks device and supported workload scope.
    See docs/reviews/cuda-layout-prediction-2026-09-18.md for evidence and limits.
    """
    try:
        shape = tuple(operator.index(n) for n in shape)
    except TypeError as exc:
        raise ValueError("Expected three integer domain extents") from exc
    if len(shape) != 3 or min(shape) <= 24:
        raise ValueError("CPML12 layout prediction needs three extents above 24")
    candidates: dict[str, dict[str, Any]] = {
        "".join(map(str, axes)): dict(
            stored_shape=tuple(shape[a] for a in axes),
            **_queue_work(tuple(shape[a] for a in axes)),
        )
        for axes in _AXES
    }
    baseline = candidates["012"]["total_slots"]
    best = min(candidates, key=lambda name: candidates[name]["total_slots"])
    reduction = 1 - candidates[best]["total_slots"] / baseline
    selected = best if reduction >= _MINIMUM_WORK_REDUCTION else "012"
    return dict(
        model=_GEOMETRY_MODEL,
        choice=dict(axes=tuple(map(int, selected)), shell_tile="64x4"),
        logical_shape=shape,
        candidate_work=candidates,
        minimum_work_reduction=_MINIMUM_WORK_REDUCTION,
        best_work_reduction=reduction,
        reason=(
            "Substantial reduction in CPML queue thread slots"
            if selected != "012"
            else "Canonical layout retained; no substantial proxy improvement"
        ),
    )


def choose_winner(samples, *, minimum_gain=0.03):
    """Keep baseline unless a candidate wins both balanced timing halves."""
    baseline = samples["012/64x4"]
    median = statistics.median
    eligible = ["012/64x4"]
    for key, values in samples.items():
        if len(values) != len(baseline) or len(values) < 4:
            continue
        half = len(values) // 2
        if (
            median(values) <= median(baseline) * (1 - minimum_gain)
            and median(values[:half]) < median(baseline[:half])
            and median(values[half:]) < median(baseline[half:])
        ):
            eligible.append(key)
    return min(eligible, key=lambda key: median(samples[key]))


@lru_cache(maxsize=1)
def _implementation_identity():
    extension = importlib.import_module("beamz._cuda")

    assert extension.__file__ is not None
    digest = hashlib.sha256(Path(extension.__file__).read_bytes())
    # Invalidate decisions for Python dispatch, arithmetic and layout changes too.
    simulation = Path(__file__).resolve().parents[1]
    for path in sorted(simulation.rglob("*.py")):
        digest.update(str(path.relative_to(simulation)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_path(signature, device, policy, hardware=None):
    program_identity = asdict(signature)
    # Cache location and refresh mode do not change numerical workload identity.
    program_identity.pop("cuda_tuning_policy", None)
    identity = dict(
        schema=_SCHEMA,
        program=program_identity,
        implementation=_implementation_identity(),
        jax=jax.__version__,
        device=device.device_kind,
        device_id=device.id,
        platform=device.client.platform_version,
        hardware=hardware,
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    base = policy[1] or str(
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "beamz/cuda-tuning"
    )
    return Path(base) / (key + ".json"), key


def _write_record(path, record):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as handle:
            tmp = Path(handle.name)
            json.dump(record, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except OSError:
        # An unwritable cache must not prevent simulation execution.
        if "tmp" in locals():
            with suppress(OSError):
                tmp.unlink(missing_ok=True)


def _gpu_status() -> dict[str, Any] | None:
    """Read physical headroom independently of the JAX allocator's own limit."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,power.limit,memory.free,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        rows = result.stdout.strip().splitlines()
        if len(rows) != 1:
            return None
        uuid, driver, power, free, temperature = (x.strip() for x in rows[0].split(","))
        return dict(
            uuid=uuid,
            driver=driver,
            power_limit=float(power),
            free_mib=float(free),
            temperature_c=float(temperature),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _safe_headroom():
    status = _gpu_status()
    return (
        status is not None
        and status["free_mib"] >= 4096
        and status["temperature_c"] < 85
    )


def _eligible(program):
    cfg = program.config
    if not (
        cfg.backend == "cuda_streamed"
        and cfg.is_3d
        and not cfg.sharding.enabled
        and cfg.metric_kind == "isotropic_uniform"
    ):
        return False
    if cfg.cuda_flags != abi.CUDA_DEFAULT_FLAGS or cfg.cuda_storage_axes != (0, 1, 2):
        return False
    if cfg.num_steps < 32:
        return False
    shapes = program.boundary.logical_component_shapes
    shape = tuple(min(shapes[name][a] for name in ("Ex", "Ey", "Ez")) for a in range(3))
    if min(shape) <= 24 or np.prod(shape) < 8 * 1024 * 1024:
        return False
    cpml = program.boundary.cpml
    if not cpml.enabled or len(cpml.h_terms) != 6 or len(cpml.e_terms) != 6:
        return False
    if not all(t.slab.low == t.slab.high == 12 for t in (*cpml.h_terms, *cpml.e_terms)):
        return False
    coefficients = program.coefficients
    if coefficients.e_inverse_offdiagonal.size:
        return False
    # Packed E material IDs use e_decay_* to carry their coefficient lookup
    # tables. Inspect conductivity rather than mistaking those tables for decay.
    if np.any(np.asarray(program.grid.conductivity) != 0):
        return False
    if not all(
        np.asarray(getattr(coefficients, f"{p}_decay_{c}")).shape == ()
        and float(getattr(coefficients, f"{p}_decay_{c}")) == 1.0
        for p in "h"
        for c in "xyz"
    ):
        return False
    if not all(s.is_slab for s in program.sources):
        return False
    return all(
        m.recorder_index < 0
        and not m.accumulate_power
        and not m.accumulate_frequency
        and m.dft_enabled
        and m.freq_count > 0
        and m.dft_point_count > 0
        for m in program.monitors
    )


def _state_hashes(state):
    hashes = []
    for leaf in jax.tree.leaves(state):
        value = np.asarray(jax.device_get(leaf))
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite calibration state")
        hashes.append(hashlib.sha256(value.tobytes()).hexdigest())
    return hashes


def select_program(program, signature, policy):
    """Predict by default; measurement and persistent lookup require opt-in."""
    if policy[0] == "off" or policy[2] or not _eligible(program):
        return program
    with _TUNING_LOCK:
        if policy[0] == "auto":
            devices = jax.devices()
            if len(devices) != 1 or not devices[0].device_kind.endswith("RTX 3090"):
                return program
            shapes = program.boundary.logical_component_shapes
            shape = tuple(
                min(shapes[name][a] for name in ("Ex", "Ey", "Ez")) for a in range(3)
            )
            report = dict(
                predict_layout(shape),
                selection_method="geometry",
                calibration_s=0.0,
                calibration_steps=0,
                requested_steps=program.config.num_steps,
                validated=False,  # No validation trial was executed for this request.
                cache_hit=False,
            )
            return _remember(_apply(program, report["choice"]), report)
        return _select_program(program, signature, policy)


def _select_program(program, signature, policy):
    devices = jax.devices()
    if len(devices) != 1 or not devices[0].device_kind.endswith("RTX 3090"):
        return program
    device = devices[0]
    status = _gpu_status()
    if status is None:
        return _remember(
            program, dict(reason="GPU telemetry unavailable; calibration skipped")
        )
    hardware = {k: status[k] for k in ("uuid", "driver", "power_limit")}
    path, key = _cache_path(signature, device, policy, hardware)
    if policy[0] != "refresh":
        try:
            record = json.loads(path.read_text())
            if (
                record["key"] == key
                and record["schema"] == _SCHEMA
                and record["validated"] is True
            ):
                selected = _apply(program, record["choice"])
                return _remember(selected, dict(record, cache_hit=True))
        except (OSError, ValueError, KeyError, TypeError):
            pass

    from beamz.simulation import observe, sharding
    from beamz.simulation.execute import (
        CUDA_GRAPH_MAX_STEPS,
        build_scan,
        initial_program_state,
    )

    if not _safe_headroom():
        return _remember(
            program, dict(reason="Insufficient GPU headroom; calibration skipped")
        )
    started = time.perf_counter()
    # Long executions replay native chunks of this size. Profile one chunk with
    # the real source/monitor plans, then retain the original execution horizon.
    calibration_steps = min(program.config.num_steps, CUDA_GRAPH_MAX_STEPS)
    probe = replace(
        program, config=replace(program.config, num_steps=calibration_steps)
    )
    state = initial_program_state(
        program, t=signature.t0, current_step=0, monitor_steps=program.config.num_steps
    )
    state = sharding.prepare_state(
        program, state, replicated_fields=(*observe.MONITOR_FIELDS, "t", "current_step")
    )
    coefficients = sharding.place_tree(program, program.coefficients)
    jax.block_until_ready(state)
    choices = {
        "".join(map(str, axes)) + "/" + tile: dict(axes=axes, shell_tile=tile)
        for axes in _AXES
        for tile in _TILES
    }
    executables = {}
    rejected = {}
    reference = None
    for name, choice in choices.items():
        if not _safe_headroom():
            return _remember(
                program, dict(reason="Calibration stopped for GPU headroom")
            )
        candidate = _apply(probe, choice)
        try:
            executable = build_scan(candidate).lower(state, coefficients).compile()
        except ValueError as exc:
            if name == "012/64x4":
                raise
            rejected[name] = str(exc)
            continue
        result = jax.block_until_ready(executable(state, coefficients))
        try:
            hashes = _state_hashes(result)
        except FloatingPointError:
            if reference is None:
                return _remember(
                    program,
                    dict(
                        reason="Nonfinite baseline calibration state", validated=False
                    ),
                )
            rejected[name] = "Nonfinite calibration state"
            continue
        del result
        if reference is None:
            reference = hashes
        elif hashes != reference:
            rejected[name] = "Complete state differs from baseline"
            continue
        for _ in range(2):
            jax.block_until_ready(executable(state, coefficients))
        executables[name] = executable
    names = list(executables)
    samples = {name: [] for name in names}
    # Three rotations, then the same orders reversed: every candidate appears in
    # early and late positions. No input donation and no live-state mutation.
    orders = [names[i:] + names[:i] for i in range(3)]
    orders += [order[::-1] for order in orders[::-1]]
    for order in orders:
        if not _safe_headroom():
            return _remember(
                program, dict(reason="Calibration stopped for GPU headroom")
            )
        for name in order:
            start = time.perf_counter()
            result = jax.block_until_ready(executables[name](state, coefficients))
            samples[name].append(time.perf_counter() - start)
            del result
    winner = choose_winner(samples)
    record = dict(
        schema=_SCHEMA,
        key=key,
        choice=choices[winner],
        samples_s=samples,
        rejected=rejected,
        validated=True,
        minimum_gain=0.03,
        calibration_s=time.perf_counter() - started,
        timing_orders=orders,
        cache_hit=False,
        calibration_steps=calibration_steps,
        requested_steps=program.config.num_steps,
        scope="RTX3090 lossless CPML12; long runs calibrated on one native chunk",
    )
    _write_record(path, record)
    return _remember(_apply(program, choices[winner]), record)
