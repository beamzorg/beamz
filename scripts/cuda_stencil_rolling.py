"""Generate rolling-plane CUDA from the experimental scalar cell trace.

Periodic equal-extent scalar fields only; this is a research probe, not a
production BeamZ schedule. The containing CUDA source supplies ``wrap(int,int)``.
Initial fields stay immutable while a separate bank receives final H/E fields.
"""

from __future__ import annotations

try:
    from scripts.cuda_stencil_trace import emit_cuda, trace_yee_step
except ModuleNotFoundError:
    from cuda_stencil_trace import emit_cuda, trace_yee_step


def _validate(tile, depth):
    if len(tile) != 3 or any(type(v) is not int or v < 1 for v in tile):
        raise ValueError("tile must contain three positive integers (z,y,x)")
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")


def rolling_resource_estimate(tile, depth):
    """Exact dynamic shared allocation and loop-envelope work, excluding masks.

    Each nonfinal half-step stores three components in a two-z-plane ring.
    There is no shared input-prefetch bank in this first generated schedule.
    Registers, barriers, memory transactions, and device limits need measurement.
    """
    _validate(tile, depth)
    tz, ty, tx = tile
    stages = 2 * depth
    areas = [(tx + stages - 1 - k) * (ty + stages - 1 - k) for k in range(stages)]
    phase_updates = sum(area * (tz + stages - 1 - k) for k, area in enumerate(areas))
    return {
        "tile_zyx": tuple(tile),
        "depth": depth,
        "shared_bytes": 3 * 2 * 4 * sum(areas[:-1]),
        "stored_half_steps": stages - 1,
        "redundant_phase_update_ratio": phase_updates / (stages * tz * ty * tx),
        "transverse_phase_update_ratio": sum(areas) / (stages * ty * tx),
    }


def generate_rolling_kernel(tile_zyx, depth):
    """Define ``rolling_D(in,out,nx,ny,nz)`` for one depth-D advance.

    Launch with a grid covering the owned tile, any positive block size, and
    ``rolling_resource_estimate(...)["shared_bytes"]`` dynamic shared bytes.
    Coordinates outside the physical domain read immutable periodic input.
    Only owned, in-domain cells are published, including partial final tiles.
    """
    _validate(tile_zyx, depth)
    tz, ty, tx = tile_zyx
    stages = 2 * depth
    shapes = [(tx + stages - 1 - k, ty + stages - 1 - k) for k in range(stages)]
    offsets = []
    cursor = 0
    for sx, sy in shapes[:-1]:
        offsets.append(cursor)
        cursor += 3 * 2 * sx * sy
    field_ids = {
        kind + c: k * 3 + i for k, kind in enumerate("HE") for i, c in enumerate("xyz")
    }

    def global_read(field, dz=0, dy=0, dx=0):
        return (
            f"in[{field_ids[field]}*n+"
            f"(wrap(z+({dz}),nz)*ny+wrap(y+({dy}),ny))*nx+wrap(x+({dx}),nx)]"
        )

    def ring(k, component, dz=0, dy=0, dx=0):
        sx, sy = shapes[k]
        return (
            f"s[{offsets[k]}+(({component}*2+((z+({dz}))&1))*{sy}"
            f"+iy+({dy}))*{sx}+ix+({dx})]"
        )

    updates = trace_yee_step()
    bodies = []
    for k in range(stages):
        phase = k % 2
        sx, sy = shapes[k]
        phase_kind = "E" if phase else "H"

        def read_name(read, k=k, phase=phase, phase_kind=phase_kind):
            dz, dy, dx = read.offset
            component = "xyz".index(read.field[1])
            if read.field.startswith(phase_kind):
                if k < 2:
                    return global_read(read.field, dz, dy, dx)
                return ring(k - 2, component, dz, dy + 1, dx + 1)
            if k == 0:
                return global_read(read.field, dz, dy, dx)
            return ring(k - 1, component, dz, dy + phase, dx + phase)

        assignments = []
        for component, update in enumerate(updates[3 * phase : 3 * phase + 3]):
            value = emit_cuda(update.value, read_name)
            if k == stages - 1:
                assignments.append(
                    f"float next_{component}={value};\n"
                    "if(x>=0 && x<nx && y>=0 && y<ny && "
                    f"z>=oz && z<oz+{tz} && z<nz) {{\n"
                    f"  int g=(z*ny+y)*nx+x;\n"
                    f"  out[{3 + component}*n+g]=next_{component};\n"
                    f"  out[{component}*n+g]={ring(k - 1, component, 0, 1, 1)};\n"
                    "}"
                )
            else:
                assignments.append(f"{ring(k, component)}={value};")
        bodies.append(
            f"if(wave>={k}) {{\n"
            f"  const int z=oz+wave-{depth}-{k // 2};\n"
            f"  for(int i=threadIdx.x;i<{sx * sy};i+=blockDim.x) {{\n"
            f"    const int ix=i%{sx},iy=i/{sx};\n"
            f"    const int x=ox-{depth}+{(k + 1) // 2}+ix;\n"
            f"    const int y=oy-{depth}+{(k + 1) // 2}+iy;\n"
            + "\n".join(assignments)
            + "\n  }\n}\n__syncthreads();"
        )
    return (
        f"__global__ void rolling_{depth}(const float* in,float* out,"
        "int nx,int ny,int nz) {\n"
        "extern __shared__ float s[];\n"
        f"const int ox=blockIdx.x*{tx},oy=blockIdx.y*{ty},oz=blockIdx.z*{tz};\n"
        "const int n=nx*ny*nz;\n"
        "const float inv_spacing=1.f;\n"
        "const float source_Hx=.07f,source_Hy=.07f,source_Hz=.07f;\n"
        "const float source_Ex=.09f,source_Ey=.09f,source_Ez=.09f;\n"
        "const float decay_Hx=.999f,decay_Hy=.999f,decay_Hz=.999f;\n"
        "const float decay_Ex=.998f,decay_Ey=.998f,decay_Ez=.998f;\n"
        f"for(int wave=0;wave<{tz + stages - 1};++wave) {{\n"
        + "\n".join(bodies)
        + "\n}\n}\n"
    )
