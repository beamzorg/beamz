"""Independent frequency-domain Yee and staggered power-balance references."""

import numpy as np
from scipy.linalg import solve_banded

import beamz as bz


def solve_yee(z, eps, frequency, dt):
    """E at edges and time-aligned H at centers; unit incoming E from +z."""
    omega = 2 * np.sin(np.pi * frequency * dt) / dt
    d = np.diff(z)
    left, right = np.r_[d[0], d], np.r_[d, d[-1]]
    dual = (left + right) / 2
    diag = -1 / left - 1 / right + (omega / bz.LIGHT_SPEED) ** 2 * eps * dual
    ql = np.exp(2j * np.arcsin(omega / bz.LIGHT_SPEED * d[0] * np.sqrt(eps[0]) / 2))
    qr = np.exp(2j * np.arcsin(omega / bz.LIGHT_SPEED * d[-1] * np.sqrt(eps[-1]) / 2))
    diag[0] += ql / left[0]
    diag[-1] += qr / right[-1]
    ab = np.zeros((3, len(z)), complex)
    ab[0, 1:], ab[1], ab[2, :-1] = 1 / right[:-1], diag, 1 / left[1:]
    rhs = np.zeros(len(z), complex)
    rhs[-1] = (qr - 1 / qr) / right[-1]
    e = solve_banded((1, 1), ab, rhs)
    h = np.diff(e) / (1j * omega * bz.MU_0 * d)
    # A face current satisfies the discrete Poynting identity exactly for real mu.
    flux = 0.25 * np.real((e[:-1] + e[1:]) * h.conj())
    loss = 0.5 * omega * bz.EPS_0 * eps[1:-1].imag * dual[1:-1] * abs(e[1:-1]) ** 2
    residual = np.diff(flux) + loss
    return e, h, flux, loss, residual


def widths(edges, bounds, location):
    centers = (edges[:-1] + edges[1:]) / 2
    if location == "center":
        low, high = edges[:-1], edges[1:]
    else:
        low = np.r_[edges[0] - (edges[1] - edges[0]) / 2, centers[:-1]]
        high = centers
    return np.maximum(0, np.minimum(high, bounds[1]) - np.maximum(low, bounds[0]))


def closed_box_integrals(e, x, y, z, indices, faces):
    """Outward Yee face powers and integral of |E|² in a staggered box."""
    (i0, i1), (j0, j1), (k0, k1) = faces
    xc, yc, zc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, (z[:-1] + z[1:]) / 2
    xb, yb, zb = (xc[i0], xc[i1]), (yc[j0], yc[j1]), (zc[k0], zc[k1])
    iz0, iz1 = np.searchsorted(indices, [k0, k1])
    wz_e, wz_c = widths(z, zb, "edge")[indices], widths(z, zb, "center")[indices]
    wxe, wxc = widths(x, xb, "edge"), widths(x, xb, "center")
    wye, wyc = widths(y, yb, "edge"), widths(y, yb, "center")

    def fz(k, wye=wye, wxc=wxc, wyc=wyc, wxe=wxe):
        a = 0.25 * np.real((e["Ex"][k] + e["Ex"][k + 1]) * e["Hy"][k].conj())
        b = 0.25 * np.real((e["Ey"][k] + e["Ey"][k + 1]) * e["Hx"][k].conj())
        return np.einsum("fyx,y,x->f", a, wye, wxc) - np.einsum(
            "fyx,y,x->f", b, wyc, wxe
        )

    def fx(i, wyc=wyc, wye=wye):
        a = 0.25 * np.real(
            (e["Ey"][:, :, :, i] + e["Ey"][:, :, :, i + 1]) * e["Hz"][:, :, :, i].conj()
        )
        b = 0.25 * np.real(
            (e["Ez"][:, :, :, i] + e["Ez"][:, :, :, i + 1]) * e["Hy"][:, :, :, i].conj()
        )
        return np.einsum("zfy,z,y->f", a, wz_e, wyc) - np.einsum(
            "zfy,z,y->f", b, wz_c, wye
        )

    def fy(j, wxe=wxe, wxc=wxc):
        a = 0.25 * np.real(
            (e["Ez"][:, :, j, :] + e["Ez"][:, :, j + 1, :]) * e["Hx"][:, :, j, :].conj()
        )
        b = 0.25 * np.real(
            (e["Ex"][:, :, j, :] + e["Ex"][:, :, j + 1, :]) * e["Hz"][:, :, j, :].conj()
        )
        return np.einsum("zfx,z,x->f", a, wz_c, wxe) - np.einsum(
            "zfx,z,x->f", b, wz_e, wxc
        )

    terms = (("Ex", wz_e, wye, wxc), ("Ey", wz_e, wyc, wxe), ("Ez", wz_c, wye, wxe))
    integral = sum(
        np.einsum("zfyx,z,y,x->f", abs(e[c]) ** 2, wz, wy, wx)
        for c, wz, wy, wx in terms
    )
    return fz(iz1), fz(iz0), fx(i1) - fx(i0) + fy(j1) - fy(j0), integral
