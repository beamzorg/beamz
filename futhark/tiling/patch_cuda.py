#!/usr/bin/env python3
"""Move an intra-block kernel's output column from shared to global memory.

Futhark places every block-level array of an intra-block kernel in shared
memory, including a block's result, which it copies to global memory when the
block finishes. A tile that streams a whole z column therefore needs the full
column in shared memory. This patch rewrites such a kernel so that the result
array aliases the block's slice of the global result instead:

* the first, runtime-sized shared allocation (the result) is pointed at
  `result + block * bytes`, and its shared reservation is removed;
* the final shared-to-global copy loop is disabled;
* the host launch stops requesting shared memory for it.

    patch_cuda.py PROGRAM.c KERNEL_SUBSTRING

edits PROGRAM.c in place (both the embedded kernel source and the host
launch). Only kernels whose result is that first runtime-sized array qualify;
the script asserts the expected shapes and fails loudly otherwise.
"""
import json
import re
import sys


def kernel_bounds(src: str, name: str):
    start = src.index(f"void {name}(")
    end = src.find("\nFUTHARK_KERNEL", start)
    return start, end if end >= 0 else len(src)


def patch_kernel(k: str):
    m = re.search(
        r"volatile __local unsigned char \*(color_\d+)_backing_(\d+) = &shared_mem\[0\];\n"
        r"\s*const int64_t \1_backing_\2_offset = 0 \+ \((bytes_\d+) \+ srem64\(\(int64_t\) 8 - srem64\(\3, \(int64_t\) 8\), \(int64_t\) 8\)\);",
        k,
    )
    assert m, "no runtime-sized first shared allocation"
    color, backing, nbytes = m.group(1), m.group(2), m.group(3)
    k = k.replace(m.group(0), m.group(0).split("\n")[0] + f"\n    const int64_t {color}_backing_{backing}_offset = 0;")
    # The final copy: ((__global float *) mem_R)[gtid_B * (...) ...] = ((__local float *) color)[...]
    c = re.search(
        r"if \((slt32\(i_\d+, [^\n]*\))\) \{\n(?:\s*//[^\n]*\n)?\s*\(\(__global float \*\) (mem_\d+)\)\[(gtid_\d+) \* [^\n]*= \(\(__local float \*\) "
        + color + r"\)",
        k,
    )
    assert c, "no final copy loop"
    cond, result, gtid = c.group(1), c.group(2), c.group(3)
    k = k.replace(f"if ({cond}) {{\n", f"if (0 && {cond}) {{\n", 1)
    decl = f"    {color} = (__local unsigned char *) {color}_backing_{backing};\n"
    assert decl in k, "no result pointer assignment"
    k = k.replace(decl, f"    {color} = (unsigned char *) {result} + {gtid} * {nbytes};\n")
    assert k.index(f"{gtid} = ") < k.index(f"{color} = (unsigned char *) {result}"), "block id set too late"
    return k, nbytes


def main():
    path, which = sys.argv[1:]
    src = open(path).read()
    # The kernel source is one C string literal split into chunks.
    lit = re.search(r'(static const char \*gpu_program\[\] = \{)(.*?)(, NULL\};)', src, re.S)
    assert lit, "no embedded program"
    chunks = re.findall(r'"((?:[^"\\]|\\.)*)"', lit.group(2))
    program = json.loads("[" + ",".join(f'"{c}"' for c in chunks) + "]")
    program = "".join(program)
    names = re.findall(r"void (\w*segmap_intrablock\w*)\(", program)
    names = [n for n in names if which in n]
    assert names, "no matching intra-block kernel"
    patched = []
    for name in names:
        s, e = kernel_bounds(program, name)
        kernel, nbytes = patch_kernel(program[s:e])
        program = program[:s] + kernel + program[e:]
        patched.append((name, nbytes))
    body = ", ".join(json.dumps(program[i : i + 4000]) for i in range(0, len(program), 4000))
    src = src[: lit.start(2)] + body + src[lit.end(2) :]
    # Host launch: drop the result's reservation from the shared byte count.
    for name, nbytes in patched:
        launch = re.compile(
            r"(gpu_kernel_" + name + r"\(ctx, [^;]*?\(int64_t\) \d+, 1, 1, )"
            + re.escape(nbytes)
            + r" \+ srem64\(\(int64_t\) 8 - srem64\("
            + re.escape(nbytes)
            + r", \(int64_t\) 8\), \(int64_t\) 8\) \+ "
        )
        src, n = launch.subn(r"\g<1>", src)
        assert n == 1, f"host launch of {name} not patched ({n})"
        print(f"patched {name}: result {nbytes} now in global memory")
    open(path, "w").write(src)


if __name__ == "__main__":
    main()
