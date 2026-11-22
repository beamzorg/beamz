//! Read-only NumPy volumes backed by a dictionary of xy planes. The owned
//! anonymous file has one physical copy of each plane. Only our reserved
//! logical address range is remapped, and every mapped address has an owner.
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray3};
use pyo3::exceptions::{PyMemoryError, PyOSError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct Map {
    address: usize,
    length: usize,
}

impl Map {
    fn new(length: usize, protection: i32, flags: i32, fd: i32) -> io::Result<Self> {
        // No placement hint is given. The kernel chooses an unused range.
        let address = unsafe { libc::mmap(std::ptr::null_mut(), length, protection, flags, fd, 0) };
        if address == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            address: address as usize,
            length,
        })
    }
}
impl Drop for Map {
    fn drop(&mut self) {
        // The entire range was reserved by Map::new, including replacements.
        unsafe {
            libc::munmap(self.address as *mut _, self.length);
        }
    }
}

struct Bank {
    logical: Map,
    physical: Map,
    shape: [usize; 4],
    indices: Vec<usize>,
    pitch: usize,
    count: usize,
}

#[pyclass(frozen, module = "beamz.design.raster._native")]
pub(super) struct PlaneMapping {
    bank: Arc<Bank>,
    physical: bool,
}

fn product(a: usize, b: usize) -> PyResult<usize> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| PyOverflowError::new_err("plane layout exceeds the address space"))
}

fn storage(
    py: Python<'_>,
    shape: [usize; 4],
    indices: Vec<usize>,
    values: &[f32],
) -> PyResult<Py<PyAny>> {
    let [c, z, y, x] = shape;
    let n = product(y, x)?;
    if n == 0
        || z == 0
        || c == 0
        || indices.len() != product(c, z)?
        || values.len() % n != 0
        || indices.iter().any(|&i| i >= values.len() / n)
    {
        return Err(PyValueError::new_err(
            "invalid plane dictionary shape or indices",
        ));
    }
    let count = values.len() / n;
    let bytes = product(n, 4)?;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return Err(io::Error::last_os_error().into());
    }
    let pitch = product(bytes.div_ceil(page as usize), page as usize)?;
    let file_length = product(pitch, count)?;
    let logical_length = product(pitch, indices.len())?;
    let raw = unsafe {
        libc::syscall(
            libc::SYS_memfd_create,
            c"beamz-planes".as_ptr(),
            libc::MFD_CLOEXEC,
        ) as libc::c_int
    };
    if raw < 0 {
        return Err(io::Error::last_os_error().into());
    }
    let file = unsafe { OwnedFd::from_raw_fd(raw) };
    if unsafe { libc::ftruncate(file.as_raw_fd(), file_length as libc::off_t) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let physical = Map::new(
        file_length,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        file.as_raw_fd(),
    )?;
    for (i, plane) in values.chunks_exact(n).enumerate() {
        // The destination is the unique writable mapping of file_length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                plane.as_ptr(),
                (physical.address + i * pitch) as *mut f32,
                n,
            );
        }
    }
    if unsafe { libc::mprotect(physical.address as *mut _, physical.length, libc::PROT_READ) } != 0
    {
        return Err(io::Error::last_os_error().into());
    }
    let logical = Map::new(
        logical_length,
        libc::PROT_NONE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
    )?;
    for (j, &i) in indices.iter().enumerate() {
        // pitch is page-aligned and i,j were range-checked; MAP_FIXED can
        // replace only pages inside our own logical reservation.
        let address = (logical.address + j * pitch) as *mut _;
        let mapped = unsafe {
            libc::mmap(
                address,
                pitch,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_FIXED,
                file.as_raw_fd(),
                (i * pitch) as libc::off_t,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(io::Error::last_os_error().into());
        }
    }
    let bank = Arc::new(Bank {
        logical,
        physical,
        shape,
        indices,
        pitch,
        count,
    });
    array(py, bank, false)
}

fn array(py: Python<'_>, bank: Arc<Bank>, physical: bool) -> PyResult<Py<PyAny>> {
    let owner = Py::new(py, PlaneMapping { bank, physical })?;
    Ok(py
        .import("numpy")?
        .call_method1("asarray", (owner,))?
        .unbind())
}

#[pymethods]
impl PlaneMapping {
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let b = &self.bank;
        let [c, z, _y, x] = b.shape;
        let out = PyDict::new(py);
        if self.physical {
            out.set_item("shape", (b.count, b.pitch / 4))?;
            out.set_item("strides", (b.pitch, 4))?;
        } else {
            out.set_item("shape", (c, z, _y, x))?;
            out.set_item("strides", (z * b.pitch, b.pitch, x * 4, 4))?;
        }
        out.set_item(
            "typestr",
            if cfg!(target_endian = "little") {
                "<f4"
            } else {
                ">f4"
            },
        )?;
        let address = if self.physical {
            b.physical.address
        } else {
            b.logical.address
        };
        out.set_item("data", (address, true))?;
        out.set_item("version", 3)?;
        Ok(out)
    }
    #[getter]
    fn logical_shape(&self) -> [usize; 4] {
        self.bank.shape
    }
    #[getter]
    fn address(&self) -> usize {
        self.bank.logical.address
    }
    #[getter]
    fn pitch(&self) -> usize {
        self.bank.pitch
    }
    #[getter]
    fn physical(&self) -> bool {
        self.physical
    }
    fn physical_planes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        array(py, self.bank.clone(), true)
    }
    fn plane_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_vec(py, self.bank.indices.clone())
    }
}

pub(super) fn from_tensor(
    py: Python<'_>,
    tensor: &mut fdtd_raster_core::TensorArray,
) -> PyResult<Py<PyAny>> {
    let ids = std::mem::take(&mut tensor.plane_ids);
    let values = std::mem::take(&mut tensor.values);
    let result = storage(py, tensor.shape, ids.clone(), &values);
    if result
        .as_ref()
        .is_err_and(|e| e.is_instance_of::<PyOSError>(py))
    {
        // Sandboxes may deny anonymous files. Retain ordinary dense NumPy
        // behavior when the host cannot supply the optional shared mapping.
        let n = product(tensor.shape[2], tensor.shape[3])?;
        let mut dense = Vec::new();
        dense
            .try_reserve_exact(product(ids.len(), n)?)
            .map_err(|_| PyMemoryError::new_err("dense raster does not fit in memory"))?;
        for i in ids {
            dense.extend_from_slice(&values[i * n..(i + 1) * n]);
        }
        return Ok(PyArray1::from_vec(py, dense)
            .reshape(tensor.shape)?
            .into_any()
            .unbind());
    }
    result
}

#[pyfunction]
pub(super) fn map_planes(
    py: Python<'_>,
    values: PyReadonlyArray3<'_, f32>,
    ids: Vec<usize>,
    shape: [usize; 4],
) -> PyResult<Py<PyAny>> {
    let (_, rows, columns) = values.as_array().dim();
    if rows != shape[2] || columns != shape[3] {
        return Err(PyValueError::new_err(
            "physical and logical plane extents must match",
        ));
    }
    storage(py, shape, ids, values.as_slice()?)
}
