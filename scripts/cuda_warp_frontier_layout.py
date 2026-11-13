"""Derive inter-warp field publication faces from canonical Yee reads.

An8Z×4Y×8X brick belongs to one warp: each lane owns all8Z, with4Y×8X
lanes. Z reads are same-thread; X/Y reads inside a brick use warp shuffles.
Only X/Y faces crossing a warp brick need shared publication. Layout words
are packed per field without duplicating X/Y face intersections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

try:
    from scripts.cuda_space_time_ir import FIELDS, yee_operator, validate_stream_signature
except ModuleNotFoundError:
    from cuda_space_time_ir import FIELDS, yee_operator, validate_stream_signature


@dataclass(frozen=True)
class Face:
    field: str
    base: int
    count: int
    y_boundary: int | None
    x_boundary: int | None


def widths(warp_y):
    if warp_y not in (2, 4, 8):
        raise ValueError("Supported warp-Y dimensions are2/4/8")
    return warp_y, 32//warp_y


def point(tid, z, *, warp_y=4):
    wy, wx = widths(warp_y)
    warp, lane = tid//32, tid % 32
    return z, (warp//wy)*wy+lane//wx, (warp % wy)*wx+lane % wx


def owner(p, *, warp_y=4):
    wy, wx = widths(warp_y)
    z, y, x = p
    return ((y//wy)*wy+x//wx)*32+(y % wy)*wx+x % wx


@lru_cache(maxsize=3)
def derive(warp_y=4):
    wy, wx = widths(warp_y)
    op = yee_operator(cpml_axes=(0, 1, 2))
    validate_stream_signature(op)
    requirements = {f: {} for f in FIELDS}
    for update in op.updates:
        if update.output.field not in FIELDS:
            continue
        for read in update.reads:
            if read.field not in FIELDS or read.field[0] == update.output.field[0]:
                continue
            for axis, delta in enumerate(read.offset):
                if not delta or axis == 0:
                    continue
                width = wy if axis == 1 else wx
                boundary = 0 if delta == 1 else width-1
                old = requirements[read.field].setdefault(axis, boundary)
                if old != boundary:
                    raise ValueError("Opposite incoming faces for one component unsupported")
    result, base = [], 0
    for field in FIELDS:
        axes = requirements[field]
        if not axes or set(axes) - {1, 2}:
            raise ValueError("Expected at least one inter-warp axial producer face")
        y, x = axes.get(1), axes.get(2)
        count = (4096//wy if y is not None else 0) + (4096//wx if x is not None else 0)
        if y is not None and x is not None:
            count -= 128
        result.append(Face(field, base, count, y, x))
        base += count
    return tuple(result)


FACES = derive()
WORDS = sum(f.count for f in FACES)


def index(field, p, *, warp_y=4):
    wy, wx = widths(warp_y)
    if isinstance(field, str):
        field = FIELDS.index(field)
    f = FACES[field] if warp_y == 4 else derive(warp_y)[field]
    z, y, x = (p[0] % 8, p[1] % 16, p[2] % 32)
    if f.y_boundary is not None and y % wy == f.y_boundary:
        return f.base+(z*(16//wy)+y//wy)*32+x
    if f.x_boundary is not None and x % wx == f.x_boundary:
        if f.y_boundary is not None:
            packed_y = y-y//wy-int(f.y_boundary == 0)
            return f.base+4096//wy+(z*(16-16//wy)+packed_y)*(32//wx)+x//wx
        return f.base+(z*16+y)*(32//wx)+x//wx
    return None


def emit_header(warp_y=4):
    wy, wx = widths(warp_y)
    faces = FACES if warp_y == 4 else derive(warp_y)
    words = sum(f.count for f in faces)
    lines = ["// Generated from canonical Yee read offsets by cuda_warp_frontier_layout.py.",
             "#pragma once", "namespace beamz::cuda::moving_native {",
             f"constexpr int WarpFrontierWords={words};",
             "__device__ __forceinline__ int WarpFrontierIndex(int field,int zs,int ys,int xs){",
             " const int z=zs&7,y=ys&15,x=xs&31;", " switch(field){"]
    for i, f in enumerate(faces):
        lines.append(f"  case {i}:")
        if f.y_boundary is not None:
            lines.append(f"   if((y&{wy-1})=={f.y_boundary})return {f.base}+(z*{16//wy}+y/{wy})*32+x;")
        if f.x_boundary is not None:
            if f.y_boundary is not None:
                correction = "-1" if f.y_boundary == 0 else ""
                lines.append(f"   if((x&{wx-1})=={f.x_boundary})return {f.base+4096//wy}+(z*{16-16//wy}+y-y/{wy}{correction})*{32//wx}+x/{wx};")
            else:
                lines.append(f"   if((x&{wx-1})=={f.x_boundary})return {f.base}+(z*16+y)*{32//wx}+x/{wx};")
        lines.append("   return -1;")
    lines += ["  default:return -1;", " }", "}",
              "__device__ __forceinline__ void WarpFrontierStore(float*frame,int f,int z,int y,int x,float value){",
              " const int key=WarpFrontierIndex(f,z,y,x);if(key>=0)frame[key]=value;", "}",
              "__device__ __forceinline__ float WarpFrontierLoad(const float*frame,int f,int z,int y,int x){",
              " return frame[WarpFrontierIndex(f,z,y,x)];", "}",
              "} // namespace beamz::cuda::moving_native", ""]
    return "\n".join(lines)


def descriptor(warp_y=4):
    wy, wx = widths(warp_y)
    faces = FACES if warp_y == 4 else derive(warp_y)
    return {"shape": [8, 16, 32], "warp_brick_zyx": [8, wy, wx], "threads": 512,
            "register_field_words_per_thread": 48, "shared_words": sum(f.count for f in faces),
            "fields": [asdict(f) for f in faces]}
