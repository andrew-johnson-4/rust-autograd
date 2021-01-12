#[cfg(feature = "blas")]
pub(crate) use crate::cblas_sys::*;

#[cfg(all(feature = "blas", feature = "intel-mkl"))]
pub(crate) use crate::intel_mkl_sys::*;

#[cfg(feature = "blas")]
pub(crate) enum MemoryOrder {
    C,
    F,
}

#[cfg(feature = "blas")]
pub(crate) type BlasIF = i32;