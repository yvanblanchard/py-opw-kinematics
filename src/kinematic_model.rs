use pyo3::prelude::*;

use rs_opw_kinematics::kinematics_impl::OPWKinematics;
use rs_opw_kinematics::parameters::opw_kinematics::Parameters;

#[pyclass(frozen)] // Declare the class as frozen to provide immutability.
#[derive(Clone)]
pub struct KinematicModel {
    pub a1: f64,
    pub a2: f64,
    pub b: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
    pub c4: f64,
    pub offsets: [f64; 6],
    pub flip_axes: [bool; 6], // Renamed and changed to boolean array
    pub has_parallelogram: bool,
    /// Index of the joint that drives the parallelogram linkage (0-based, default: 1 = J2)
    pub parallelogram_driven: usize,
    /// Index of the joint coupled through the parallelogram linkage (0-based, default: 2 = J3)
    pub parallelogram_coupled: usize,
    /// Scaling factor: coupled_angle += scaling * driven_angle (default: 1.0)
    pub parallelogram_scaling: f64,
}

impl KinematicModel {
    pub fn to_opw_kinematics(&self, degrees: bool) -> OPWKinematics {
        let sign_corrections = self.flip_axes.map(|x| if x { -1 } else { 1 });
        OPWKinematics::new(Parameters {
            a1: self.a1,
            a2: self.a2,
            b: self.b,
            c1: self.c1,
            c2: self.c2,
            c3: self.c3,
            c4: self.c4,
            offsets: if degrees {
                self.offsets
                    .iter()
                    .map(|&x| x.to_radians())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            } else {
                self.offsets
            },
            sign_corrections,
            dof: 6,
        })
    }
}

#[pymethods]
impl KinematicModel {
    #[new]
    #[pyo3(signature = (
        a1 = 0.0,
        a2 = 0.0,
        b = 0.0,
        c1 = 0.0,
        c2 = 0.0,
        c3 = 0.0,
        c4 = 0.0,
        offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        flip_axes = (false, false, false, false, false, false),
        has_parallelogram = false,
        parallelogram_driven = 1,
        parallelogram_coupled = 2,
        parallelogram_scaling = 1.0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a1: f64,
        a2: f64,
        b: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        offsets: (f64, f64, f64, f64, f64, f64),
        flip_axes: (bool, bool, bool, bool, bool, bool),
        has_parallelogram: bool,
        parallelogram_driven: usize,
        parallelogram_coupled: usize,
        parallelogram_scaling: f64,
    ) -> PyResult<Self> {
        if parallelogram_driven >= 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "parallelogram_driven must be in range 0..5",
            ));
        }
        if parallelogram_coupled >= 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "parallelogram_coupled must be in range 0..5",
            ));
        }
        if parallelogram_driven == parallelogram_coupled {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "parallelogram_driven and parallelogram_coupled must differ",
            ));
        }
        Ok(KinematicModel {
            a1,
            a2,
            b,
            c1,
            c2,
            c3,
            c4,
            offsets: offsets.into(),
            flip_axes: flip_axes.into(),
            has_parallelogram,
            parallelogram_driven,
            parallelogram_coupled,
            parallelogram_scaling,
        })
    }

    // Getter methods to provide access to attributes since the class is frozen.
    #[getter]
    pub fn a1(&self) -> f64 {
        self.a1
    }

    #[getter]
    pub fn a2(&self) -> f64 {
        self.a2
    }

    #[getter]
    pub fn b(&self) -> f64 {
        self.b
    }

    #[getter]
    pub fn c1(&self) -> f64 {
        self.c1
    }

    #[getter]
    pub fn c2(&self) -> f64 {
        self.c2
    }

    #[getter]
    pub fn c3(&self) -> f64 {
        self.c3
    }

    #[getter]
    pub fn c4(&self) -> f64 {
        self.c4
    }

    #[getter]
    pub fn offsets(&self) -> Vec<f64> {
        self.offsets.to_vec() // Convert the array to a Vec for easier handling in Python.
    }

    #[getter]
    pub fn flip_axes(&self) -> Vec<bool> {
        self.flip_axes.to_vec() // Convert the array to a Vec for easier handling in Python.
    }

    #[getter]
    pub fn has_parallelogram(&self) -> bool {
        self.has_parallelogram
    }

    #[getter]
    pub fn parallelogram_driven(&self) -> usize {
        self.parallelogram_driven
    }

    #[getter]
    pub fn parallelogram_coupled(&self) -> usize {
        self.parallelogram_coupled
    }

    #[getter]
    pub fn parallelogram_scaling(&self) -> f64 {
        self.parallelogram_scaling
    }

    pub fn __repr__(&self) -> String {
        let mut s = format!(
            "KinematicModel(\n    a1={},\n    a2={},\n    b={},\n    c1={},\n    c2={},\n    c3={},\n    c4={},\n    offsets={:?},\n    flip_axes={:?},\n    has_parallelogram={}",
            self.a1, self.a2, self.b, self.c1, self.c2, self.c3, self.c4,
            self.offsets, self.flip_axes, self.has_parallelogram
        );
        if self.has_parallelogram {
            s.push_str(&format!(
                ",\n    parallelogram_driven={},\n    parallelogram_coupled={},\n    parallelogram_scaling={}",
                self.parallelogram_driven, self.parallelogram_coupled, self.parallelogram_scaling
            ));
        }
        s.push_str("\n)");
        s
    }
}
