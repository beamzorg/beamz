"""Write intra-block kernel results straight to global memory.

Futhark places every block-level array of an intra-block kernel in shared
memory, including the block's results, and copies them to global memory when
the block ends. A temporal tile (temporal.fut) streams a whole z column, so
its results cannot fit in shared memory. This patch rewrites each runtime-sized
result of an intra-block kernel to alias the block's slice of the global result
instead:

* the result's shared allocation takes no space, and its pointer becomes
  ``result + block * bytes``;
* the final shared-to-global copy loop is disabled;
* the host launch stops reserving shared memory for it.

The result must be laid out identically in both places, which holds for the
row-major per-block results temporal.fut produces. Every step asserts the code
shape it expects, so a change in Futhark's code generator fails the build
instead of producing a wrong kernel.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_PROGRAM = re.compile(r"(static const char \*gpu_program\[\] = \{)(.*?)(, NULL\};)", re.S)
_CHUNK = re.compile(r'"((?:[^"\\]|\\.)*)"')
# f16 copies are wrapped as fptobits_f16_i16(bitstofp_i16_f16(...)).
_COPY_OUT = re.compile(
    r"if \((slt32\(i_\d+, [^\n]*\))\) \{\n(?:\s*//[^\n]*\n)?"
    r"\s*\(\(__global (\w+) \*\) (mem_\d+)\)\[(gtid_\d+) \* [^\n]*= (?:fptobits_\w+\(bitstofp_\w+\()?\(\(__local \2 \*\) (color_\d+)\)"
)


_BACKING = re.compile(
    r"volatile __local unsigned char \*(color_\d+)_backing_(\d+) = &shared_mem\[[^\]]*\];\n"
    r"\s*const int64_t \1_backing_\2_offset = (.*?) \+ (\(int64_t\) \d+|\(bytes_\d+ \+ srem64[^;]*\));"
)


def _split_sum(expr: str) -> list[str]:
    """Split a sum of allocation sizes at top-level ' + '. A bare first size
    ``bytes_N + srem64(...)`` stays one term."""
    terms, depth, start = [], 0, 0
    for i, ch in enumerate(expr):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and expr.startswith(" + ", i):
            terms.append(expr[start:i])
            start = i + 3
    terms.append(expr[start:])
    merged = []
    for term in terms:
        if term.startswith("srem64(") and merged and re.fullmatch(r"bytes_\d+", merged[-1]):
            merged[-1] = f"{merged[-1]} + {term}"
        else:
            merged.append(term)
    return merged


def _strip(term: str) -> str:
    return re.sub(r"[\s()]", "", term)


def _patch_kernel(kernel: str) -> tuple[str, list[tuple[int, str]]]:
    """Patch one kernel; return the indices and sizes of the moved results."""
    backings = list(_BACKING.finditer(kernel))
    order = [m.group(1) for m in backings]
    moved = []
    for match in _COPY_OUT.finditer(kernel):
        cond, _, result, gtid, color = match.groups()
        if color not in order:
            continue
        backing = backings[order.index(color)]
        _, number, base, size = backing.groups()
        name = f"{color}_backing_{number}_offset"
        kernel = kernel.replace(f"const int64_t {name} = {base} + {size};", f"const int64_t {name} = {base};")
        assign = f"{color} = (__local unsigned char *) {color}_backing_{number};"
        assert kernel.count(assign) == 1, f"no single pointer assignment for {color}"
        nbytes = re.match(r"\((bytes_\d+) \+", size)
        stride = nbytes.group(1) if nbytes else size
        kernel = kernel.replace(assign, f"{color} = (unsigned char *) {result} + {gtid} * {stride};")
        assert kernel.index(f"{gtid} = ") < kernel.index(f"{color} = (unsigned char *) {result}"), (
            "block index assigned after the result pointer"
        )
        kernel = kernel.replace(f"if ({cond}) {{\n", f"if (0 && {cond}) {{\n", 1)
        moved.append((order.index(color), size))
    return kernel, moved


def move_results_to_global(source: Path, kernel_filter: str = "segmap_intrablock") -> list[str]:
    """Patch every intra-block kernel whose name contains `kernel_filter`."""
    src = source.read_text()
    literal = _PROGRAM.search(src)
    assert literal, "no embedded GPU program"
    program = "".join(json.loads("[" + ",".join(f'"{c}"' for c in _CHUNK.findall(literal.group(2))) + "]"))
    patched, patched_before = [], []
    for name in re.findall(r"void (\w*segmap_intrablock\w*)\(", program):
        if kernel_filter not in name:
            continue
        start = program.index(f"void {name}(")
        end = program.find("\nFUTHARK_KERNEL", start)
        end = len(program) if end < 0 else end
        kernel, moved = _patch_kernel(program[start:end])
        if not moved:
            if re.search(r"color_\d+ = \(unsigned char \*\) mem_\d+ \+ gtid_\d+ \* ", kernel):
                patched_before.append(name)
            continue
        program = program[:start] + kernel + program[end:]
        patched.append((name, moved))
    # Experiment: a minimum number of resident blocks per SM, which caps the
    # registers per thread (BEAMZ_TILE_MIN_BLOCKS).
    blocks = os.environ.get("BEAMZ_TILE_MIN_BLOCKS")
    for name, _ in patched if blocks else []:
        program, n = re.subn(
            rf"FUTHARK_KERNEL_SIZED\({name}_dim1, 1, 1\)",
            f'extern "C" __global__ __launch_bounds__({name}_dim1, {int(blocks)})',
            program,
        )
        assert n == 1, f"no launch bounds for {name}"
    body = ", ".join(json.dumps(program[i : i + 4000]) for i in range(0, len(program), 4000))
    src = src[: literal.start(2)] + body + src[literal.end(2) :]
    for name, moved in patched:
        call = re.search(rf"gpu_kernel_{name}\(ctx, [^,]+, 1, 1, [^,]+, 1, 1, ", src)
        assert call, f"no host launch of {name}"
        # The shared-memory size argument runs to the next top-level comma.
        start, depth = call.end(), 0
        for end in range(start, len(src)):
            depth += src[end] == "("
            depth -= src[end] == ")"
            if depth == 0 and src[end] == ",":
                break
        # The host sums the sizes in its own order (and folds constants), so
        # drop one matching term per moved result.
        kept = _split_sum(src[start:end])
        for _, size in moved:
            match = [i for i, t in enumerate(kept) if _strip(t) == _strip(size)]
            assert match, f"host launch of {name} reserves no {size!r}: {kept}"
            del kept[match[0]]
        kept = kept or ["(int64_t) 0"]
        src = src[:start] + " + ".join(kept) + src[end:]
    source.write_text(src)
    return [name for name, _ in patched] + patched_before


if __name__ == "__main__":
    import sys

    for name in move_results_to_global(Path(sys.argv[1]), *sys.argv[2:]):
        print("patched", name)
