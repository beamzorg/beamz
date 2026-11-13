#!/usr/bin/env python3
"""Freeze complete Mstep ABI buffers and 30-leaf JAX/CUDA state references.

JAX is used only by this exporter. The paired native runtime reader imports
NumPy and an independently loaded extension, never BeamZ's Python package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_fdtdx_comparison import NATIVE_SHA, file_digest

FIELDS_NATIVE = ("hx", "hy", "hz", "ex", "ey", "ez")
WORKSPACE_INPUTS = frozenset((*range(74, 86), *range(123, 126)))
ZERO_WORKSPACE_INPUTS = frozenset(range(80, 86))
PERSISTENT_INPUTS = frozenset(
    (*range(6), *range(31, 37), *range(62, 68), *range(120, 123))
)


def buffer_digest(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape)).encode())
    if value.size:
        digest.update(memoryview(np.ascontiguousarray(value).reshape(-1)).cast("B"))
    return digest.hexdigest()


def abi_input_names():
    names = list(FIELDS_NATIVE)
    for phase in ("h", "e"):
        names += [
            f"{phase}_{kind}_{axis}" for kind in ("decay", "source") for axis in "xyz"
        ]
        names.append(f"{phase}_term_metadata")
        names += [
            f"{phase}_term_{i}_{kind}"
            for i in range(6)
            for kind in ("a", "b", "inv_kappa")
        ]
        names += [f"{phase}_psi_{i}" for i in range(6)]
    names += [f"metric_{phase}_{axis}" for phase in ("h", "e") for axis in "zyx"]
    names += [f"final_{name}" for name in FIELDS_NATIVE]
    names += [f"scratch_{name}" for name in FIELDS_NATIVE]
    for phase, components in (
        ("pre_e", ("ex", "ey", "ez")),
        ("h", ("hx", "hy", "hz")),
        ("e", ("ex", "ey", "ez")),
    ):
        for component in components:
            names += [
                f"source_{phase}_{component}_{kind}"
                for kind in ("coefficients", "waveforms", "starts")
            ]
    names += [
        "monitor_" + name
        for name in (
            "indices",
            "weights",
            "frequencies",
            "component_masks",
            "counts",
            "codes",
            "windows",
            "dft_re",
            "dft_im",
            "dft_weight",
            "phase_sin",
            "phase_cos",
            "phase_window",
            "origin",
            "current_step",
            "elapsed_steps",
        )
    ]
    names += [
        "plan_bounds",
        "plan_core_masks",
        "plan_slots",
        "plan_first_cells",
        "plan_edge_cells",
    ]
    assert len(names) == 134
    return names


def flatten_state(state):
    result = []
    for name, value in state._asdict().items():
        if isinstance(value, tuple):
            result.extend((f"{name}.{index}", item) for index, item in enumerate(value))
        else:
            result.append((name, value))
    return result


def seed_state(state, seed):
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    changes = {}
    for name, value in state._asdict().items():
        if name in ("t", "current_step"):
            continue
        if isinstance(value, tuple):
            changes[name] = tuple(
                jnp.asarray(rng.standard_normal(item.shape).astype(np.float32) * 1e-3)
                for item in value
            )
        elif np.dtype(value.dtype).kind == "f" and value.size:
            scale = 3e-6 if name in ("hx", "hy", "hz") else 1e-3
            changes[name] = jnp.asarray(
                rng.standard_normal(value.shape).astype(np.float32) * scale
            )
    return state._replace(**changes)


def initial_native_state(program, *, t, current_step, monitor_steps):
    """Keep fresh lattice/CPML zeros as host descriptors until native upload."""
    from beamz.simulation import execute
    from beamz.simulation.model import SimulationState

    dispersion = program.dispersion
    if program.sharding.layout.enabled or (
        dispersion is not None and (dispersion.regions or dispersion.interfaces)
    ):
        return execute.initial_program_state(program, t=t, current_step=current_step,
                                             monitor_steps=monitor_steps)
    fields = {name.lower(): np.asarray(getattr(program.grid, name))
              for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    if any(value.dtype != np.float32 for value in fields.values()):
        raise ValueError("Native descriptor state requires FP32 fields")
    def psi(terms):
        return tuple(np.broadcast_to(np.zeros((), np.float32), term.slab.shape) for term in terms)
    return SimulationState(**fields, cpml_psi_h_terms=psi(program.boundary.cpml.h_terms),
        cpml_psi_e_terms=psi(program.boundary.cpml.e_terms), polarization=(),
        **execute.monitor_runtime.empty_monitor_values(program, num_steps=max(1, monitor_steps)),
        t=np.asarray(t, np.float32), current_step=np.asarray(current_step, np.int32))


def capture_mstep(program, state, steps, *, materialize_workspaces=True):
    from beamz.simulation.cuda import pack_dft_monitors
    from beamz.simulation.cuda import runtime as rt
    from beamz.simulation.cuda.mstep import build_mstep_plan, run_mstep
    from beamz.simulation.execute import SOURCE_PHASE_COMPONENTS, build_step_context

    ctx = build_step_context(program)
    groups = tuple(
        ctx.source_batches[(phase, component)][0]
        for phase, components in SOURCE_PHASE_COMPONENTS.items()
        for component in components
    )
    if any(rest for _, rest in ctx.source_batches.values()):
        raise ValueError("Fixture supports exact batched slab sources only")
    monitors = pack_dft_monitors(program.monitors)
    shapes = tuple(tuple(getattr(state, name).shape) for name in FIELDS_NATIVE)
    plan = build_mstep_plan(shapes, ctx.boundary.cpml, groups, monitors[0])
    if not plan.core_cells:
        raise ValueError(
            "Correctness fixture must exercise a real temporal column core"
        )
    if not materialize_workspaces:
        # Trace the ABI descriptors without allocating buffers that the native
        # session defers or owns itself. Use the same canonical layout helpers
        # as run_mstep; no numerical update is reproduced here.
        from beamz.simulation import _cuda_abi as abi
        from beamz.simulation.cuda.mstep import _plan_arrays

        arguments, values, aliases = rt._graph_io(state, ctx, program.coefficients)
        fields = rt._fields(state)
        virtual = tuple(np.broadcast_to(np.zeros((), f.dtype), f.shape) for f in fields)
        virtual += tuple(np.broadcast_to(np.zeros((), np.float32), (plan.scratch_cells,)) for _ in fields)
        for value in virtual:
            aliases[len(arguments)] = len(values)
            arguments, values = (*arguments, value), (*values, value)
        arguments = (*arguments, *rt._source_group_arguments(groups))
        arguments, values, aliases, _, count = rt._monitor_io(
            state, monitors, arguments, values, aliases,
            observation_origin=state.t, observation_step_offset=np.int32(0))
        arguments = (*arguments, *_plan_arrays(plan))
        schedule = rt.NativeSchedulePlan(
            layout=abi.PROGRAM_LAYOUT_YEE_IN_PLACE,
            flags=rt._native_schedule_flags(ctx, program.coefficients, sources=True, monitors=count),
            monitor_count=count,
            coincident_source_group_mask=rt._coincident_source_group_mask(groups),
            disjoint_source_group_mask=rt._disjoint_source_group_mask(groups), temporal_steps=2)
        if len(arguments) != 134 or len(values) != 36:
            raise ValueError("Unexpected descriptor-only Mstep ABI layout")
        return dict(target=abi.CUDA_MSTEP_TARGET, arguments=arguments, result_values=values,
                    aliases=aliases, attributes=rt._program_attributes(ctx, steps, schedule),
                    plan_summary=plan.summary())
    captured = {}
    original = rt._ffi_call

    def intercept(target, result_values, aliases):
        def call(*arguments, **attributes):
            captured.update(
                target=target,
                arguments=arguments,
                result_values=result_values,
                aliases=aliases,
                attributes=attributes,
            )
            return tuple(result_values)

        return call

    rt._ffi_call = intercept
    try:
        run_mstep(
            state,
            ctx,
            program.coefficients,
            groups,
            monitors,
            plan,
            steps,
            observation_origin=state.t,
            observation_step_offset=np.int32(0),
        )
    finally:
        rt._ffi_call = original
    if (
        len(captured.get("arguments", ())) != 134
        or len(captured["result_values"]) != 36
    ):
        raise ValueError("Unexpected Mstep ABI layout; refusing an approximate fixture")
    captured["plan_summary"] = plan.summary()
    return captured


def make_program(args):
    if args.contract:
        from scripts.beamz_release_adapter import configure_common_parameters
        from scripts.export_fdtdx_comparison import build_program

        contract = json.loads(args.contract.read_text())
        _, program, spec = build_program(
            contract["mie_arguments"], expected_native_sha=args.expected_component_sha
        )
        data_path = args.contract.parent / contract["data_file"]
        if file_digest(data_path) != contract["data_sha256"]:
            raise ValueError("Common comparison data changed")
        with np.load(data_path, allow_pickle=False) as data:
            program, audit = configure_common_parameters(program, contract, data)
        steps = args.steps or program.config.num_steps
        program = replace(program, config=replace(program.config, num_steps=steps))
        return program, {
            "kind": "common_mie",
            "contract": str(args.contract),
            "contract_sha256": file_digest(args.contract),
            "spec": spec,
            "parameter_configuration": audit,
        }
    from scripts.benchmark_cuda_realistic import build_simulation

    options = SimpleNamespace(
        shape=tuple(args.shape),
        steps=args.steps or 256,
        pml=12,
        monitors=2,
        frequencies=3,
        material="binary",
        source="mode",
        monitor_type="mode",
        boundary="all",
    )
    sim = build_simulation(options)
    return sim.compile(num_steps=options.steps, backend="cuda_mstep"), {
        "kind": "realistic",
        **vars(options),
    }


def state_output_mapping(name):
    if name in FIELDS_NATIVE:
        return {"kind": "field", "component": FIELDS_NATIVE.index(name)}
    if name.startswith("cpml_psi_h_terms."):
        return {"kind": "output", "index": 6 + int(name.rsplit(".", 1)[1])}
    if name.startswith("cpml_psi_e_terms."):
        return {"kind": "output", "index": 12 + int(name.rsplit(".", 1)[1])}
    if name in ("dft_vec_re", "dft_vec_im", "dft_weight_sum"):
        return {
            "kind": "output",
            "index": 30 + ("dft_vec_re", "dft_vec_im", "dft_weight_sum").index(name),
        }
    if name in ("t", "current_step"):
        return {"kind": "clock", "name": name}
    return {"kind": "unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed-state", type=int)
    parser.add_argument("--split", type=int, nargs="+", default=[])
    parser.add_argument("--expected-component-sha", default=NATIVE_SHA)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import beamz._cuda as extension
    import jax

    from beamz.simulation.execute import build_scan, initial_program_state

    if file_digest(extension.__file__) != args.expected_component_sha:
        raise ValueError(
            "Process-local CUDA component does not match the requested hash"
        )
    program, metadata = make_program(args)
    if not program.boundary.cpml.enabled or program.config.cuda_storage_axes != (
        0,
        1,
        2,
    ):
        raise ValueError("Initial native fixtures require unrotated CPML12")
    steps = program.config.num_steps
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=steps)
    if args.seed_state is not None:
        state = seed_state(state, args.seed_state)
    jax.block_until_ready(state)
    captured = capture_mstep(program, state, steps)
    arrays = {}
    inputs = []
    for index, (name, value) in enumerate(
        zip(abi_input_names(), captured["arguments"], strict=True)
    ):
        host = (
            np.zeros(value.shape, value.dtype)
            if index in WORKSPACE_INPUTS
            else np.asarray(value)
        )
        if host.dtype not in (np.dtype(np.float32), np.dtype(np.int32)):
            raise ValueError(
                f"Initial NumPy native fixtures require FP32/S32 buffers: {name}"
            )
        key = f"input_{index:03d}"
        arrays[key] = host
        role = (
            "persistent_state"
            if index in PERSISTENT_INPUTS
            else "zero_each_call"
            if index in ZERO_WORKSPACE_INPUTS
            else "uninitialized_workspace"
            if index in WORKSPACE_INPUTS
            else "clock"
            if index in (126, 127, 128)
            else "readonly"
        )
        inputs.append(
            {
                "index": index,
                "name": name,
                "key": key,
                "dtype": host.dtype.str,
                "shape": list(host.shape),
                "role": role,
                "sha256": buffer_digest(host),
            }
        )
    leaves = []
    initial_values = dict(flatten_state(state))
    for index, (name, value) in enumerate(flatten_state(state)):
        key = f"initial_{index:02d}"
        arrays[key] = np.asarray(value)
        leaves.append(
            {
                "name": name,
                "initial_key": key,
                "dtype": arrays[key].dtype.str,
                "shape": list(arrays[key].shape),
                "mapping": state_output_mapping(name),
            }
        )
    if len(leaves) != 30:
        raise ValueError(f"Expected all 30 state leaves, found {len(leaves)}")
    reference = jax.block_until_ready(build_scan(program)(state, program.coefficients))
    for leaf, (name, value) in zip(leaves, flatten_state(reference), strict=True):
        if leaf["name"] != name:
            raise ValueError("Reference state structure changed")
        key = "reference_" + leaf["initial_key"].split("_")[1]
        arrays[key] = np.asarray(value)
        leaf["reference_key"] = key
    splits = args.split or ([3, steps - 3] if steps == 7 else [])
    if splits and (sum(splits) != steps or min(splits) < 1):
        raise ValueError("Continuation chunks must partition the fixture step count")
    continuation = state
    for count in splits:
        subprogram = replace(program, config=replace(program.config, num_steps=count))
        continuation = jax.block_until_ready(
            build_scan(subprogram)(continuation, subprogram.coefficients)
        )
    if splits:
        for index, (_, value) in enumerate(flatten_state(continuation)):
            arrays[f"continuation_{index:02d}"] = np.asarray(value)
    for name, value in flatten_state(state):
        if not np.array_equal(np.asarray(value), np.asarray(initial_values[name])):
            raise ValueError("Reference solver mutated its immutable input")
    attributes = {
        name: np.asarray(value).item() for name, value in captured["attributes"].items()
    }
    outputs = [
        {"index": i, "shape": list(v.shape), "dtype": np.dtype(v.dtype).str}
        for i, v in enumerate(captured["result_values"])
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = args.output.with_suffix(".npz")
    np.savez_compressed(data, **arrays)
    report = {
        "schema": "beamz.cuda-native-fixture.v1",
        "kind": "mstep",
        "target": captured["target"],
        "attributes": attributes,
        "attribute_dtypes": {
            name: np.asarray(value).dtype.str
            for name, value in captured["attributes"].items()
        },
        "inputs": inputs,
        "outputs": outputs,
        "output_aliases": captured["aliases"],
        "state_leaves": leaves,
        "continuation_steps": splits,
        "data_file": data.name,
        "data_sha256": file_digest(data),
        "fixture": metadata,
        "seed": args.seed_state,
        "plan": captured["plan_summary"],
        "native_component_sha256": file_digest(extension.__file__),
        "jax_version": jax.__version__,
        "exporter_sha256": file_digest(__file__),
        "ownership": {
            "preserve": "Restore persistent state inside measured invocation; initialize scratch inside each call",
            "owned": "Mutable resident state; reset outside measured invocation; same scratch initialization",
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "inputs": len(inputs),
                "state_leaves": len(leaves),
                "plan": report["plan"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
