//! Opt-in spatial diagnostics for development without dense production outputs.
use super::*;
use std::io::Write;

#[derive(Deserialize)]
struct Inspection {
    scene: Scene,
    grid: Grid,
    supports: Vec<RequestedSupport>,
}

#[derive(Deserialize)]
struct RequestedSupport {
    component: String,
    index: [usize; 3],
}

/// Run with BEAMZ_RASTER_INSPECTION_INPUT and BEAMZ_RASTER_INSPECTION_OUTPUT.
/// Input is {scene, grid, supports: [{component: "cell"|"ex"|"ey"|"ez", index}]}.
/// Each JSONL record describes one resolved physical support, including its
/// fractions, normal, constitutive tensor, and actual smoothing fallback.
#[test]
#[ignore = "requires an explicit scene, grid, support selection, and output path"]
fn write_spatial_ownership_diagnostics() {
    let input = std::env::var("BEAMZ_RASTER_INSPECTION_INPUT").unwrap();
    let output = std::env::var("BEAMZ_RASTER_INSPECTION_OUTPUT").unwrap();
    let request: Inspection = serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    request.scene.validate().unwrap();
    request.grid.validate().unwrap();
    let resolved = ResolvedExtrusions::build(&request.scene).unwrap();
    let resolved_index = ObjectIndex::build(&resolved.scene, &request.grid);
    let options = IntegrationOptions {
        smoothing: SmoothingMode::FarjadpourDiagonal,
        ..IntegrationOptions::default()
    };
    let mut output = std::io::BufWriter::new(std::fs::File::create(output).unwrap());
    for support in request.supports {
        let component = match support.component.as_str() {
            "cell" => Component::Cell,
            "ex" => Component::Ex,
            "ey" => Component::Ey,
            "ez" => Component::Ez,
            _ => panic!("unknown component"),
        };
        let shape = component.support().logical_shape(&request.grid);
        assert!((0..3).all(|axis| support.index[axis] < shape[axis]));
        let volume = component.support().volume(&request.grid, support.index);
        {
            let scene = &resolved.scene;
            let index = &resolved_index;
            let ownership = Some(&resolved);
            let mut candidates = index.query(&volume);
            candidates.retain(|i| scene.objects[*i].geometry.bounds().intersects(&volume));
            let mixture = integrate_mixture(scene, &volume, &candidates, &options, ownership, None);
            let fractions = mixture.fractions.unwrap_or_else(|| {
                let mut values = vec![0.0; scene.materials.len()];
                values[mixture.uniform_owner.unwrap()] = 1.0;
                values
            });
            let (material, smoothed, fallback) =
                effective_material(scene, &fractions, options.smoothing, mixture.interface);
            let normal = match mixture.interface {
                InterfaceClass::Laminar(normal) => Some(normal),
                _ => None,
            };
            serde_json::to_writer(
                &mut output,
                &serde_json::json!({
                    "component": support.component, "index": support.index,
                    "bounds": volume, "fractions": fractions, "normal": normal,
                    "smoothed": smoothed, "fallback": fallback.map(|reason| format!("{reason:?}")),
                    "epsilon": material.epsilon_r.0, "estimated_fraction_error": mixture.error,
                }),
            )
            .unwrap();
            writeln!(&mut output).unwrap();
        }
    }
}
