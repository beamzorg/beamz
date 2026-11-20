#[cfg(target_os = "linux")]
mod planes;

use fdtd_raster_core::{
    Grid, IntegrationOptions, OutputComponents, Quality, RasterResult, Scene, SmoothingMode,
    TensorArray, TriangleMesh, interface_samples, rasterize_prevalidated,
};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const ENGINE_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "-geometry-2");

#[pyclass(module = "beamz.design.raster._native")]
struct NativeCompiledScene {
    scene: Scene,
    scene_hash: String,
}

#[pyclass(module = "beamz.design.raster._native")]
struct NativeRasterResult {
    result: RasterResult,
}

#[pymethods]
impl NativeCompiledScene {
    #[getter]
    fn scene_hash(&self) -> &str {
        &self.scene_hash
    }

    #[pyo3(signature = (grid_edges, quality, components, staircase_materials = Vec::new(), periodic_axes = [false; 3]))]
    fn interface_samples(
        &self,
        py: Python<'_>,
        grid_edges: (Vec<f64>, Vec<f64>, Vec<f64>),
        quality: &str,
        components: &str,
        staircase_materials: Vec<usize>,
        periodic_axes: [bool; 3],
    ) -> PyResult<Py<PyAny>> {
        let (x, y, z) = grid_edges;
        let grid = Grid::new(x, y, z).map_err(value_error)?;
        let mut options = integration_options(quality, "farjadpour_diagonal", components)?;
        if staircase_materials
            .iter()
            .any(|&id| id >= self.scene.materials.len())
        {
            return Err(value_error("staircase material ID is out of range"));
        }
        options.staircase_materials = staircase_materials;
        options.periodic_axes = periodic_axes;
        let samples = py.detach(|| interface_samples(&self.scene, &grid, &options));
        let result = PyDict::new(py);
        for sample in samples {
            let record = PyDict::new(py);
            record.set_item("indices", PyArray1::from_vec(py, sample.indices))?;
            record.set_item("fractions", PyArray1::from_vec(py, sample.fractions))?;
            record.set_item(
                "normal_squared",
                PyArray1::from_vec(py, sample.normal_squared),
            )?;
            result.set_item(sample.component, record)?;
        }
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature = (
        grid_edges,
        quality = "balanced",
        smoothing = "volume",
        components = "all",
        staircase_materials = Vec::new(),
        periodic_axes = [false; 3],
    ))]
    #[allow(clippy::too_many_arguments)] // Python API keeps explicit backward-compatible options.
    fn rasterize(
        &self,
        py: Python<'_>,
        grid_edges: (Vec<f64>, Vec<f64>, Vec<f64>),
        quality: &str,
        smoothing: &str,
        components: &str,
        staircase_materials: Vec<usize>,
        periodic_axes: [bool; 3],
    ) -> PyResult<NativeRasterResult> {
        let (x_edges, y_edges, z_edges) = grid_edges;
        let grid = Grid::new(x_edges, y_edges, z_edges).map_err(value_error)?;
        let mut options = integration_options(quality, smoothing, components)?;
        if staircase_materials
            .iter()
            .any(|&id| id >= self.scene.materials.len())
        {
            return Err(value_error("staircase material ID is out of range"));
        }
        options.staircase_materials = staircase_materials;
        options.periodic_axes = periodic_axes;
        options.broadcast_uniform = true;
        options.compact_planes = cfg!(target_os = "linux")
            && grid.shape().into_iter().fold(1usize, usize::saturating_mul) >= 2 * 1024 * 1024;
        let result = py
            .detach(|| rasterize_prevalidated(&self.scene, &grid, &options))
            .map_err(value_error)?;
        Ok(NativeRasterResult { result })
    }
}

#[pymethods]
impl NativeRasterResult {
    fn take_arrays(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let arrays = PyDict::new(py);
        for (name, tensor) in ["epsilon_ex", "epsilon_ey", "epsilon_ez"]
            .into_iter()
            .zip(&mut self.result.epsilon)
        {
            insert_tensor(py, &arrays, name, tensor)?;
        }
        insert_tensor(py, &arrays, "epsilon_node", &mut self.result.node_epsilon)?;
        for (name, tensor) in ["mu_hx", "mu_hy", "mu_hz"]
            .into_iter()
            .zip(&mut self.result.mu)
        {
            insert_tensor(py, &arrays, name, tensor)?;
        }
        for (name, tensor) in ["conductivity_ex", "conductivity_ey", "conductivity_ez"]
            .into_iter()
            .zip(&mut self.result.conductivity)
        {
            insert_tensor(py, &arrays, name, tensor)?;
        }
        insert_tensor(py, &arrays, "tensor_epsilon", &mut self.result.cell_epsilon)?;
        insert_tensor(py, &arrays, "tensor_mu", &mut self.result.cell_mu)?;
        insert_tensor(
            py,
            &arrays,
            "tensor_conductivity",
            &mut self.result.cell_conductivity,
        )?;
        arrays.set_item("scene_hash", &self.result.scene_hash)?;
        arrays.set_item(
            "diagnostics_json",
            serde_json::to_string(&self.result.diagnostics).map_err(value_error)?,
        )?;
        Ok(arrays.into_any().unbind())
    }
}

fn insert_tensor(
    py: Python<'_>,
    arrays: &Bound<'_, PyDict>,
    name: &str,
    tensor: &mut TensorArray,
) -> PyResult<()> {
    #[cfg(target_os = "linux")]
    if !tensor.plane_ids.is_empty() {
        return arrays.set_item(name, planes::from_tensor(py, tensor)?);
    }
    if tensor.values.is_empty() {
        return Ok(());
    }
    let broadcast = tensor.values.len() == 1 && tensor.shape.iter().product::<usize>() > 1;
    let values = PyArray1::from_vec(py, std::mem::take(&mut tensor.values));
    if broadcast {
        arrays.set_item(
            name,
            py.import("numpy")?
                .call_method1("broadcast_to", (values, tensor.shape))?,
        )
    } else {
        arrays.set_item(name, values.reshape(tensor.shape)?)
    }
}

fn integration_options(
    quality: &str,
    smoothing: &str,
    components: &str,
) -> PyResult<IntegrationOptions> {
    let quality = match quality {
        "fast" => Quality::Fast,
        "balanced" => Quality::Balanced,
        "reference" => Quality::Reference,
        _ => {
            return Err(PyValueError::new_err(
                "quality must be 'fast', 'balanced', or 'reference'",
            ));
        }
    };
    let mut options = IntegrationOptions::for_quality(quality);
    options.smoothing = match smoothing {
        "volume" => SmoothingMode::Volume,
        "farjadpour_diagonal" => SmoothingMode::FarjadpourDiagonal,
        "farjadpour_full" => SmoothingMode::FarjadpourFull,
        _ => {
            return Err(PyValueError::new_err(
                "smoothing must be 'volume', 'farjadpour_diagonal', or 'farjadpour_full'",
            ));
        }
    };
    let components = if let Some(name) = components.strip_prefix("solver_diagonal_") {
        if options.smoothing != SmoothingMode::FarjadpourDiagonal {
            return Err(PyValueError::new_err(
                "solver diagonal projection requires diagonal smoothing",
            ));
        }
        options.project_yee_diagonal = true;
        name
    } else {
        components
    };
    options.output_components = match components {
        "all" => OutputComponents::All,
        "two_dimensional_tm" => OutputComponents::TwoDimensionalTm,
        "two_dimensional_te" => OutputComponents::TwoDimensionalTe,
        _ => {
            return Err(PyValueError::new_err(
                "components must be 'all', 'two_dimensional_tm', or 'two_dimensional_te'",
            ));
        }
    };
    Ok(options)
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
fn compile_scene(scene_json: String) -> PyResult<NativeCompiledScene> {
    let scene: Scene = serde_json::from_str(&scene_json)
        .map_err(|error| PyValueError::new_err(format!("invalid scene JSON: {error}")))?;
    scene.validate().map_err(value_error)?;
    let scene_hash = scene.stable_hash().map_err(value_error)?;
    Ok(NativeCompiledScene { scene, scene_hash })
}

#[pyfunction]
fn inspect_mesh(vertices: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> PyResult<String> {
    let report = TriangleMesh::inspect(&vertices, &triangles).map_err(value_error)?;
    let valid_for_rasterization = TriangleMesh::new(vertices, triangles).is_ok();
    let mut value = serde_json::to_value(report).map_err(value_error)?;
    value["valid_for_rasterization"] = valid_for_rasterization.into();
    serde_json::to_string(&value).map_err(value_error)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ENGINE_VERSION", ENGINE_VERSION)?;
    #[cfg(target_os = "linux")]
    {
        module.add_class::<planes::PlaneMapping>()?;
        module.add_function(wrap_pyfunction!(planes::map_planes, module)?)?;
    }
    module.add_class::<NativeCompiledScene>()?;
    module.add_class::<NativeRasterResult>()?;
    module.add_function(wrap_pyfunction!(compile_scene, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_mesh, module)?)?;
    Ok(())
}
