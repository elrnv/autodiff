///
/// This crate provides a library for performing automatic differentiation in forward mode.
///
/// # Examples
///
/// The following example differentiates a 1D function defined by a closure.
///
/// ```rust
/// ```
///
/// To compute the gradient of a function, use the function `grad` as follows:
///
/// ```rust
/// ```
///
/// Compute a specific derivative of a multi-variable function:
///
/// ```rust
/// ```
///
/// The following example shows how to compute a Jacobian product and evaluate the function at the same time.
///
/// ```
/// ```
mod autofloat;
pub use autofloat::*;
#[cfg(feature = "nalgebra")]
mod approx;
#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "nalgebra")]
mod simba;
