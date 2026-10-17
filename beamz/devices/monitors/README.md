A Monitor is a functional mapping

𝑀
:
(
𝐸
,
𝐻
,
𝑡
)
→
observable
(
𝑡
)
M:(E,H,t)→observable(t)

Implementation:

class Monitor:
    def __init__(self, region, op, temporal_reduction=None, spatial_reduction=None):
        self.region = region          # bounding box or surface
        self.op = op                  # function on E,H per cell
        self.temporal_reduction = temporal_reduction  # "instant", "DFT", "accumulate"
        self.spatial_reduction = spatial_reduction    # "none", "integrate", "average"


op could be a lambda or GPU shader snippet like dot(E, H) or abs(E)**2.

temporal_reduction defines whether to store time-series, accumulate FFTs, or just instantaneous values.

spatial_reduction defines whether to integrate or store field maps.

Then:

“flux monitor” = op = cross(E,H).n, temporal=DFT, spatial=integrate

“field recorder” = selected field components, temporal=interval, spatial=none

“mode monitor” = op = dot(E, E_mode*), temporal=DFT, spatial=integrate

“energy monitor” = op = |E|^2 + |H|^2, temporal=accumulate, spatial=integrate

All the same code path, just different parameters.

# 2D modal phase convention

Modal extraction fits forward and backward waves on the monitor's staggered
electric and magnetic Yee supports. Each basis wave carries its own spatial
phase sign, so a mixture of both directions does not receive a forward-only
phase correction. Amplitudes retain the magnetic-plane reference; temporal
leapfrog alignment is applied separately. The 3D collocated projection is
unchanged.

The two amplitudes are extracted with Lorentz power overlaps: measured E is
paired with modal H, and measured H with modal E. This preserves rejection of
power-orthogonal fields, including on weighted transverse grids; a fit dominated
by the larger numerical E values can contaminate TE transmission with radiation.
