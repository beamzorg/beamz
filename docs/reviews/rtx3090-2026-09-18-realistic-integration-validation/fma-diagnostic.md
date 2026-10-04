# Isolated numerical diagnostic

The diagnostic build is archived locally in `.cache/perf/fma-diagnostic/`.
It uses the same production sources and Python package, with SM86 only for
build speed. The sole arithmetic change is in `cuda/src/yee_primitives.cuh`:

```cpp
return __fmaf_rn(phase == 0 ? -source : source, curl,
                 __fmul_rn(decay, old_field));
```

This replaces `decay * old_field +/- source * curl`. The production build is
unchanged. Run `diagnose_dense_pair.py --steps 256 --output <result.json>` with
the matching Python package first on PYTHONPATH; add `--jax` for the independent
JAX comparison. The JSON records native binary SHA256 and per-leaf errors.
