"""Create an admitted register-core experiment in a copied CUDA source tree.

The lossless/isotropic-H option proves the six scalar coefficients from host
inputs before allocating device memory. Unsupported packets fail explicitly;
the repository's default backend and its arbitrary-material support are intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

H_ADMISSION = r'''
    // RESEARCH_REGISTER_H_ADMISSION: immutable packet proof, before allocation.
    if (mstep_) {
      float source = 0.f;
      for (int c = 0; c < 3; ++c) {
        const auto decay = Describe(inputs[6+c], numpy_, false);
        const auto scale = Describe(inputs[9+c], numpy_, false);
        if (decay.descriptor.rank != 0 || scale.descriptor.rank != 0 ||
            decay.descriptor.element_type != kBeamzF32 ||
            scale.descriptor.element_type != kBeamzF32)
          throw std::invalid_argument("register specialization needs scalar FP32 H coefficients");
        float d, s;
        std::memcpy(&d, decay.data, sizeof(float));
        std::memcpy(&s, scale.data, sizeof(float));
        if (d != 1.f || !std::isfinite(s) ||
            (c && std::memcmp(&s, &source, sizeof(float))))
          throw std::invalid_argument("register specialization needs lossless isotropic H coefficients");
        source = s;
      }
    }
'''


def patch(source: Path, threads: int, minimum_ctas: int, isotropic_h: bool):
    live = Path(__file__).resolve().parents[1] / "cuda"
    if source.resolve() == live.resolve():
        raise ValueError("Use a copied candidate, not the live repository")
    path = source / "src/mstep_register.cuh"
    text = path.read_text()
    old = "static constexpr int threads = 256;"
    expression = str(threads) if threads else "((wx * wy + 31) / 32) * 32"
    if text.count(old) != 1:
        raise ValueError("Expected the unmodified register-core thread declaration")
    text = text.replace(old, f"static constexpr int threads = {expression};")
    old = "256, BEAMZ_CUDA_MSTEP_REGISTER_FOUR_CTAS && X == 16 && Y == 8 ? 4 : 1)"
    if text.count(old) != 1:
        raise ValueError("Expected the unmodified register-core launch bounds")
    text = text.replace(old, f"MstepRegisterTile<X, Y>::threads, {minimum_ctas})")
    if isotropic_h:
        text = text.replace("hc.decay[c] = *v.h_decay[c];", "hc.decay[c] = 1.f;")
        text = text.replace("hc.source[c] = *v.h_source[c];", "hc.source[c] = *v.h_source[0];")
        native = source / "src/native_extension.cc"
        native_text = native.read_text()
        marker = "    graph_count_ = attributes_.cpml_enabled ? kCpmlGraphInputCount : kYeeGraphInputCount;"
        if native_text.count(marker) != 1:
            raise ValueError("Could not locate admission before device allocations")
        native.write_text(native_text.replace(marker, H_ADMISSION + marker))
    path.write_text(text)


def device_admission(source: Path, threads=0, minimum_ctas=2):
    """Uniform CTA coefficient proof plus the original arbitrary-scalar body.

    Unlike the host-admitted research variant, this preserves packet support.
    Both paths share one launch resource limit; measure fallback spills too.
    """
    patch(source, threads, minimum_ctas, False)
    path = source / "src/mstep_register.cuh"
    text = path.read_text()
    start = text.index("template <int X, int Y, int Layout = 0>\n__global__")
    body = text.index("void MstepRegisterColumnCore(", start)
    text = (text[:start] + "template <int X, int Y, int Layout, bool LosslessH>\n"
            "__device__ __forceinline__ " + text[body:])
    text = text.replace("void MstepRegisterColumnCore(", "void MstepRegisterColumnBody(", 1)
    text = text.replace("hc.decay[c] = *v.h_decay[c];", "hc.decay[c] = LosslessH ? 1.f : *v.h_decay[c];")
    text = text.replace("hc.source[c] = *v.h_source[c];", "hc.source[c] = *v.h_source[LosslessH ? 0 : c];")
    text += f'''
template <int X,int Y,int Layout=0>
__global__ __launch_bounds__(MstepRegisterTile<X,Y>::threads,{minimum_ctas})
void MstepRegisterColumnCore(const MstepPackedCoreView v,const int*masks,
                            const int*bounds,int ntx,int segment) {{
 const unsigned source=__float_as_uint(*v.h_source[0]);
 bool admitted=(source&0x7f800000u)!=0x7f800000u;
 #pragma unroll
 for(int c=0;c<3;++c)
   admitted &= __float_as_uint(*v.h_decay[c])==0x3f800000u &&
               __float_as_uint(*v.h_source[c])==source;
 // Immutable scalar coefficients make this choice uniform across the CTA.
 if(admitted)MstepRegisterColumnBody<X,Y,Layout,true>(v,masks,bounds,ntx,segment);
 else MstepRegisterColumnBody<X,Y,Layout,false>(v,masks,bounds,ntx,segment);
}}
'''
    path.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-source", required=True, type=Path)
    p.add_argument("--threads", type=int, default=0,
                   help="0 assigns one lane per XY coordinate, padded to a warp")
    p.add_argument("--minimum-ctas", type=int, default=1)
    p.add_argument("--isotropic-h", action="store_true")
    p.add_argument("--device-admission", action="store_true",
                   help="Select specialized/original bodies uniformly on device; retain packet support")
    a = p.parse_args()
    if a.threads < 0 or a.threads > 1024 or a.threads % 32 or a.minimum_ctas < 1:
        p.error("threads must be 0 or a warp multiple <=1024; minimum CTAs >=1")
    if a.device_admission and a.isotropic_h:
        p.error("Choose host-only admission or device admission, not both")
    if a.device_admission:
        device_admission(a.candidate_source, a.threads, a.minimum_ctas)
    else:
        patch(a.candidate_source, a.threads, a.minimum_ctas, a.isotropic_h)


if __name__ == "__main__":
    main()
