"""Compare a selected checkout using the common realistic workload builder.

Invoke in a fresh process per revision. Setup, validation and result hashing are
outside timing; compiled scan timing includes any internal storage conversion.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shape', nargs=3, type=int, required=True)
    parser.add_argument('--material', choices=('binary', 'smooth'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--save-reference', action='store_true')
    parser.add_argument('--axes', choices=('012', '120'), default='012')
    parser.add_argument('--fusion', choices=('auto', '0'), default='auto')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    for key in list(os.environ):
        if key.startswith('BEAMZ_CUDA_'):
            del os.environ[key]
    if args.axes != '012':
        os.environ['BEAMZ_CUDA_STORAGE_AXES'] = args.axes
    if args.fusion != 'auto':
        os.environ['BEAMZ_CUDA_CPML_CORE_FUSION'] = args.fusion

    import beamz
    import beamz._cuda as extension
    import jax
    import numpy as np
    from beamz.simulation.execute import build_scan, initial_program_state

    assert Path(beamz.__file__).resolve().parent.parent == root
    assert Path(extension.__file__).resolve().parent.parent == root
    workload_path = Path(__file__).with_name('benchmark_cuda_realistic.py')
    spec = importlib.util.spec_from_file_location('comparison_workload', workload_path)
    workload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workload)
    settings = SimpleNamespace(shape=args.shape, steps=256, presteps=0, pml=12,
                               monitors=2, frequencies=3, material=args.material,
                               source='mode', monitor_type='mode')
    start = time.perf_counter()
    sim = workload.build_simulation(settings)
    program = sim.compile(num_steps=256, backend='cuda_streamed')
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=256)
    jax.block_until_ready(state)
    setup_s = time.perf_counter() - start
    start = time.perf_counter()
    executable = build_scan(program, donate_state=False).lower(state, program.coefficients).compile()
    compile_s = time.perf_counter() - start
    for _ in range(4):
        jax.block_until_ready(executable(state, program.coefficients))
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        result = jax.block_until_ready(executable(state, program.coefficients))
        samples.append(time.perf_counter() - start)

    def fingerprints(value):
        return [dict(shape=list(a.shape), dtype=str(a.dtype),
                     sha256=hashlib.sha256(a.tobytes()).hexdigest())
                for leaf in jax.tree.leaves(value)
                for a in [np.asarray(jax.device_get(leaf))]]

    reference = args.reference
    leaves = jax.tree.leaves(result)
    if args.save_reference:
        reference.mkdir(parents=True, exist_ok=False)
    else:
        assert len(list(reference.glob('*.npy'))) == len(leaves)
    validation = []
    for index, leaf in enumerate(leaves):
        actual = np.asarray(jax.device_get(leaf))
        path = reference / f'{index:03}.npy'
        if args.save_reference:
            np.save(path, actual)
            expected = actual
        else:
            expected = np.load(path, mmap_mode='r')
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert np.isfinite(actual).all()
        exact = np.array_equal(actual, expected)
        scale = float(np.max(np.abs(expected), initial=0))
        max_error = float(np.max(np.abs(actual - expected), initial=0))
        tolerance_pass = True
        mismatch_count = 0
        if np.issubdtype(actual.dtype, np.inexact):
            mismatch_count = int(np.count_nonzero(
                np.abs(actual - expected) > max(3e-6, 1e-6 * scale) + 3e-5 * np.abs(expected)))
            tolerance_pass = mismatch_count == 0
        else:
            np.testing.assert_array_equal(actual, expected)
        validation.append(dict(index=index, shape=list(actual.shape), dtype=str(actual.dtype),
                               exact=exact, max_abs_error=max_error, reference_scale=scale,
                               tolerance_pass=tolerance_pass, mismatch_count=mismatch_count,
                               sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    data = dict(workload=vars(settings), root=str(root),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                extension_version=extension.__version__, extension_abi=extension.__abi_version__,
                extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                workload_sha256=hashlib.sha256(workload_path.read_bytes()).hexdigest(),
                jax_version=jax.__version__, device=str(jax.devices()[0]),
                requested_axes=args.axes, actual_axes=getattr(program.config, 'cuda_storage_axes', (0, 1, 2)),
                fusion=args.fusion, cuda_flags=program.config.cuda_flags,
                setup_s=setup_s, compile_s=compile_s, warmups=4, samples_s=samples,
                median_gcups=int(np.prod(args.shape))*256/statistics.median(samples)/1e9,
                initial_state=fingerprints(state), coefficients=fingerprints(program.coefficients),
                sources=[dict(component=s.component, timing=s.timing,
                              slab_starts=s.slab_starts, slab_sizes=s.slab_sizes,
                              arrays=fingerprints((s.coeff, s.waveform))) for s in program.sources],
                final_state_validation=validation, source_specs=len(program.sources),
                cpml_dtypes=sorted({str(x.dtype) for x in (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)}),
                validation_rtol=3e-5, validation_atol='max(3e-6, 1e-6 * reference_leaf_max_abs)')
    assert data['cpml_dtypes'] == ['float32']
    args.output.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({k: data[k] for k in ('root','median_gcups','actual_axes','cuda_flags')}), flush=True)


if __name__ == '__main__':
    main()
