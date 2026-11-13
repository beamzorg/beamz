"""Copied v13 extension of UniformE to certified isotropic-H NoIO tiles.

All cohorts, storage, clocks, CPML recurrences and point arithmetic stay intact.
The separate NoIO count excludes trajectories without a supported E update.
"""

import argparse
import hashlib
import json
from pathlib import Path

PARENT_CONSUMER = "cca7202977dfb765000881b587de287c132bb7a0f14cb05131366f13b09e642c"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one NoIO UniformE anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source):
    src = source / "src"
    consumer = src / "asymmetric_native.cuh"
    text = consumer.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != PARENT_CONSUMER:
        raise ValueError("Exact frozen v13 consumer required")
    text = once(
        text,
        " if constexpr(Fast){\n  // CTA-uniform",
        " if constexpr(Fast||(Policy==1&&IsotropicH)){\n  // CTA-uniform",
    )
    consumer.write_text(text)

    p = src / "native_extension.cc"
    text = p.read_text()
    text = once(
        text, "size_t certified_bulk=0;", "size_t certified_bulk=0,certified_noio=0;"
    )
    old = """      for(size_t tile=0;tile<moving_plan_.tile_flags.size();++tile)
        if(moving_plan_.tile_flags[tile]==0&&certificate.admitted[tile]){moving_plan_.tile_flags[tile]|=64;++certified_bulk;}
"""
    new = """      for(size_t tile=0;tile<moving_plan_.tile_flags.size();++tile){
        if(!certificate.admitted[tile])continue;
        const uint8_t flags=moving_plan_.tile_flags[tile];
        if(flags==0){moving_plan_.tile_flags[tile]|=64;++certified_bulk;continue;}
        if(!(moving_workspace_.data_flags&beamz::cuda::moving::kLosslessIsotropicH)||beamz::cuda::moving::Policy(flags)!=1)continue;
        // Count only real E update trajectories, not empty high-side boxes.
        const int q[3]={int(tile/(moving_plan_.counts[1]*moving_plan_.counts[2])),int((tile/moving_plan_.counts[2])%moving_plan_.counts[1]),int(tile%moving_plan_.counts[2])};
        bool has_e_update=false;
        for(int c=0;c<3;++c){int64_t first=1,last=depth;const auto&f=base_inputs_[3+c];
          for(int axis=0;axis<3;++axis){const int64_t origin=int64_t(q[axis])*moving_plan_.owned[axis];
            first=std::max<int64_t>(first,origin-f.dims[axis]+1);
            last=std::min<int64_t>(last,origin+moving_plan_.owned[axis]-1);
          }
          has_e_update|=first<=last;
        }
        if(has_e_update){moving_plan_.tile_flags[tile]|=64;++certified_noio;}
      }
"""
    text = once(text, old, new)
    text = once(
        text,
        "moving_uniform_bulk_=certified_bulk;",
        "moving_uniform_bulk_=certified_bulk;moving_uniform_noio_=certified_noio;",
    )
    text = once(
        text,
        "size_t moving_uniform_bulk_=0,moving_uniform_prefix_bytes_=0;",
        "size_t moving_uniform_bulk_=0,moving_uniform_noio_=0,moving_uniform_prefix_bytes_=0;",
    )
    text = once(
        text,
        'result["moving_uniform_bulk_tiles"]=moving_uniform_bulk_;',
        'result["moving_uniform_noio_tiles"]=moving_uniform_noio_;result["moving_noio_uniform_isotropic_only"]=true;result["moving_uniform_bulk_tiles"]=moving_uniform_bulk_;',
    )
    text = once(
        text,
        "NB_MODULE(_cuda_native, module) {",
        'NB_MODULE(_cuda_native, module) {\n  module.attr("moving_noio_uniform_e")=true;\n  module.attr("moving_noio_uniform_isotropic_only")=true;',
    )
    p.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("Only copied private source is allowed")
    patch(args.source)
    print(
        json.dumps(
            {
                "consumer_sha256": hashlib.sha256(
                    (args.source / "src/asymmetric_native.cuh").read_bytes()
                ).hexdigest(),
                "scope": "Existing Fast + certified Policy1 isotropic-H only; same v13 cohorts/geometry/storage and exact arithmetic",
                "noio_positive_witness": "moving_uniform_noio_tiles excludes unsupported E trajectories",
            }
        )
    )


if __name__ == "__main__":
    main()
