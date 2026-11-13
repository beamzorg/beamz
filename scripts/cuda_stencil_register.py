"""Generated rolling stencil with one shared plane per stage and register history.

Synthetic periodic/common-extent/scalar-coefficient probe only. Uses the same
cell equations as cuda_stencil_rolling, but fixed physical-XY thread ownership
keeps prior z samples in registers. E publication is delayed until the next H
stage has finished its transverse reads. The containing source supplies wrap().
"""

from __future__ import annotations

try:
    from scripts.cuda_stencil_rolling import _validate
    from scripts.cuda_stencil_trace import emit_cuda, trace_yee_step
except ModuleNotFoundError:
    from cuda_stencil_rolling import _validate
    from cuda_stencil_trace import emit_cuda, trace_yee_step


def register_resource_estimate(
    tile, depth, threads=256, *, e_history="register", h_history="register"
):
    _validate(tile, depth)
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if e_history not in {"register", "shared"} or h_history not in {
        "register",
        "shared",
    }:
        raise ValueError("e_history/h_history must be register or shared")
    _, ty, tx = tile
    stages = 2 * depth
    area = (tx + stages - 1) * (ty + stages - 1)
    slots = (area + threads - 1) // threads
    areas = [(tx + stages - 1 - k) * (ty + stages - 1 - k) for k in range(stages)]
    return {
        "tile_zyx": tuple(tile),
        "depth": depth,
        "threads": threads,
        "xy_slots_per_thread": slots,
        "shared_bytes": 3 * 4 * sum(areas[:-1]),
        "persistent_history_scalars_per_thread": 3
        * (
            (depth if h_history == "register" else 0)
            + (depth - 1 if e_history == "register" else 0)
        )
        * slots,
        "e_history": e_history,
        "h_history": h_history,
        "transient_h_history_scalars_per_thread": 3 * slots
        if h_history == "shared"
        else 0,
        "transient_e_history_scalars_per_thread": 3 * slots
        if e_history == "shared" and depth > 1
        else 0,
        "current_family_scalars_per_thread": 6 * slots,
        "note": "Register scalar counts are source-level liveness bounds, not ptxas allocations",
    }


def generate_register_kernel(
    tile_zyx, depth, threads=256, *, e_history="register", h_history="register"
):
    """Define register_rolling_D(in,out,nx,ny,nz), requiring threads per CTA.

    Registers hold each intermediate stage's previous plane, while H shared
    planes contain current-wave values and E planes prior-wave values until
    the following H finishes reading them. Only owned final cells are written.
    """
    info = register_resource_estimate(
        tile_zyx, depth, threads, e_history=e_history, h_history=h_history
    )
    tz, ty, tx = tile_zyx
    stages, slots = 2 * depth, info["xy_slots_per_thread"]
    widths = [tx + stages - 1 - k for k in range(stages)]
    heights = [ty + stages - 1 - k for k in range(stages)]
    lows = [(k + 1) // 2 for k in range(stages)]
    offsets, cursor = [], 0
    for width, height in zip(widths[:-1], heights[:-1], strict=True):
        offsets.append(cursor)
        cursor += 3 * width * height
    field_ids = {
        kind + c: j * 3 + i for j, kind in enumerate("HE") for i, c in enumerate("xyz")
    }

    def active(k):
        return f"qx>={lows[k]} && qx<{lows[k] + widths[k]} && qy>={lows[k]} && qy<{lows[k] + heights[k]}"

    def shared(k, component, dy=0, dx=0):
        return f"s[{offsets[k]}+({component}*{heights[k]}+qy+({dy})-{lows[k]})*{widths[k]}+qx+({dx})-{lows[k]}]"

    def previous(k, c):
        return f"prev_{k}_{c}[slot]"

    def current(kind, c):
        return f"cur_{kind}_{c}[slot]"

    def global_read(read):
        dz, dy, dx = read.offset
        return f"in[{field_ids[read.field]}*n+(wrap(z+({dz}),nz)*ny+wrap(y+({dy}),ny))*nx+wrap(x+({dx}),nx)]"

    updates = trace_yee_step()
    bodies = []
    for k in range(stages):
        phase, family = k % 2, "E" if k % 2 else "H"

        def read_name(read, k=k, phase=phase, family=family):
            dz, dy, dx = read.offset
            c = "xyz".index(read.field[1])
            if read.field.startswith(family):
                if k >= 2 and not phase and h_history == "shared":
                    return f"old_H_{c}[slot]"
                if k >= 3 and phase and e_history == "shared":
                    return f"old_E_{c}[slot]"
                return global_read(read) if k < 2 else previous(k - 2, c)
            if k == 0:
                return global_read(read)
            if phase == 0:
                if dz:
                    return current("E", c)
                if dy or dx:
                    return shared(k - 1, c, dy, dx)
                if e_history == "shared":
                    return f"old_E_{c}[slot]"
                return previous(k - 1, c)
            if dz:
                if h_history == "shared":
                    return f"old_H_{c}[slot]"
                return previous(k - 1, c)
            if dy or dx:
                return shared(k - 1, c, dy, dx)
            return current("H", c)

        expressions = [
            emit_cuda(u.value, read_name) for u in updates[3 * phase : 3 * phase + 3]
        ]
        compute = [f"float next_{c}=0.f;" for c in range(3)]
        compute += [f"if(wave>={k} && {active(k)}) {{"]
        if k and phase == 0 and e_history == "shared":
            compute += [f"old_E_{c}[slot]={shared(k - 1, c)};" for c in range(3)]
        compute += [f"next_{c}={value};" for c, value in enumerate(expressions)]
        compute.append("}")
        if k >= 2 and not (
            (phase and e_history == "shared") or (not phase and h_history == "shared")
        ):
            compute += [f"if(wave>={k - 2} && {active(k - 2)}) {{"]
            compute += [f"{previous(k - 2, c)}={current(family, c)};" for c in range(3)]
            compute.append("}")
        if phase == 0 and h_history == "shared":
            compute += [f"if(wave>{k} && {active(k)}) {{"]
            compute += [f"old_H_{c}[slot]={shared(k, c)};" for c in range(3)]
            compute.append("}")
        compute += [f"if(wave>={k} && {active(k)}) {{"]
        compute += [f"{current(family, c)}=next_{c};" for c in range(3)]
        if k == stages - 1:
            compute += [
                "if(x>=0 && x<nx && y>=0 && y<ny && z>=oz && z<oz+"
                + str(tz)
                + " && z<nz) {",
                "const int g=(z*ny+y)*nx+x;",
            ]
            compute += [
                f"out[{3 + c}*n+g]=next_{c}; out[{c}*n+g]={current('H', c)};"
                for c in range(3)
            ]
            compute.append("}")
        compute.append("}")
        body = [
            "{",
            f"const int z=oz+wave-{depth}-{k // 2};",
            "#pragma unroll",
            f"for(int slot=0;slot<{slots};++slot) {{",
            f"const int q=threadIdx.x+slot*{threads},qx=q%{widths[0]},qy=q/{widths[0]};",
            f"const int x=ox-{depth}+qx,y=oy-{depth}+qy;",
            *compute,
            "}",
        ]
        if phase == 0:
            body += [
                "__syncthreads();",
                "#pragma unroll",
                f"for(int slot=0;slot<{slots};++slot) {{",
                f"const int q=threadIdx.x+slot*{threads},qx=q%{widths[0]},qy=q/{widths[0]};",
                f"if(wave>={k} && {active(k)}) {{",
            ]
            body += [f"{shared(k, c)}={current('H', c)};" for c in range(3)]
            body.append("}")
            if k:
                body += [f"if(wave>={k - 1} && {active(k - 1)}) {{"]
                body += [f"{shared(k - 1, c)}={current('E', c)};" for c in range(3)]
                body.append("}")
            body += ["}", "__syncthreads();"]
        body.append("}")
        bodies.append("\n".join(body))
    declarations = [
        f"float prev_{k}_{c}[{slots}]={{}};"
        for k in range(stages - 1)
        if (e_history == "register" and k % 2)
        or (h_history == "register" and k % 2 == 0)
        for c in range(3)
    ]
    if e_history == "shared" and depth > 1:
        declarations += [f"float old_E_{c}[{slots}]={{}};" for c in range(3)]
    if h_history == "shared":
        declarations += [f"float old_H_{c}[{slots}]={{}};" for c in range(3)]
    declarations += [
        f"float cur_{family}_{c}[{slots}]={{}};" for family in "HE" for c in range(3)
    ]
    retire_last = [
        "#pragma unroll",
        f"for(int slot=0;slot<{slots};++slot) {{",
        f"const int q=threadIdx.x+slot*{threads},qx=q%{widths[0]},qy=q/{widths[0]};",
        f"if(wave>={stages - 2} && {active(stages - 2)}) {{",
    ]
    retire_last += [f"{previous(stages - 2, c)}={current('H', c)};" for c in range(3)]
    retire_last += ["}", "}"]
    if h_history == "shared":
        retire_last = []
    return (
        f"__global__ void register_rolling_{depth}(const float* in,float* out,int nx,int ny,int nz) {{\n"
        f"if(blockDim.x!={threads}) return;\nextern __shared__ float s[];\n"
        f"const int ox=blockIdx.x*{tx},oy=blockIdx.y*{ty},oz=blockIdx.z*{tz},n=nx*ny*nz;\n"
        "const float inv_spacing=1.f;\n"
        "const float source_Hx=.07f,source_Hy=.07f,source_Hz=.07f;\n"
        "const float source_Ex=.09f,source_Ey=.09f,source_Ez=.09f;\n"
        "const float decay_Hx=.999f,decay_Hy=.999f,decay_Hz=.999f;\n"
        "const float decay_Ex=.998f,decay_Ey=.998f,decay_Ez=.998f;\n"
        + "\n".join(declarations)
        + f"\nfor(int wave=0;wave<{tz + stages - 1};++wave) {{\n"
        + "\n".join(bodies + ["\n".join(retire_last)])
        + "\n}\n}\n"
    )


def generate_register_probe(
    tile_zyx, depths=(1, 2, 4), *, e_history="register", h_history="register"
):
    """Standalone parent-probe source with independent CPU and global controls.

    Reuse the existing correctness/timing harness; replace only generated
    kernel definitions and their dynamic shared allocation. The CUDA launch
    remains 256 threads, matching the fixed-XY register ownership contract.
    """
    try:
        from scripts.cuda_stencil_rolling import generate_rolling_kernel
        from scripts.probe_cuda_stencil_trace import generate_cuda
    except ModuleNotFoundError:
        from cuda_stencil_rolling import generate_rolling_kernel
        from probe_cuda_stencil_trace import generate_cuda
    source = generate_cuda(tuple(tile_zyx), depths, layout="rolling")
    for depth in depths:
        old = generate_rolling_kernel(tile_zyx, depth).replace(
            f"__global__ void rolling_{depth}(",
            f"template<> __global__ void rolling_dispatch<{depth}>(",
        )
        replacement = generate_register_kernel(
            tile_zyx, depth, e_history=e_history, h_history=h_history
        ).replace(
            f"__global__ void register_rolling_{depth}(",
            f"template<> __global__ void rolling_dispatch<{depth}>(",
        )
        if source.count(old) != 1:
            raise ValueError(
                "Parent rolling probe changed; kernel substitution is ambiguous"
            )
        source = source.replace(old, replacement)
    if source.count("return 24*total;") != 1:
        raise ValueError("Parent rolling probe changed; shared allocation is ambiguous")
    return source.replace("return 24*total;", "return 12*total;")
