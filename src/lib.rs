//!
//! This crate provides a library for performing automatic differentiation.
//!
//! # Examples
//!
//! The following example differentiates a 1D function defined by a closure.
//!
//! ```rust
//! use autodiff::*;
//! // Define a function `f(x) = e^{-0.5*x^2}`
//! let f = |x: F1| (-x * x / F::cst(2.0)).exp();
//!
//! // Differentiate `f` at zero.
//! println!("{}", diff(f, 0.0)); // prints `0`
//! #   assert_eq!(diff(f, 0.0), 0.0);
//! ```
//!
//! To compute the gradient of a function, use the function `grad` as follows:
//!
//! ```rust
//! use autodiff::*;
//! // Define a function `f(x,y) = x*y^2`
//! let f = |x: &[F1]| x[0] * x[1] * x[1];
//!
//! // Differentiate `f` at `(1,2)`.
//! let g = grad(f, &vec![1.0, 2.0]);
//! println!("({}, {})", g[0], g[1]); // prints `(4, 4)`
//! #   assert_eq!(g, vec![4.0, 4.0]);
//! ```
//!
//! Compute a specific derivative of a multi-variable function:
//!
//! ```rust
//! use autodiff::*;
//! // Define a function `f(x,y) = x*y^2`.
//! let f = |v: &[F1]| v[0] * v[1] * v[1];
//!
//! // Differentiate `f` at `(1,2)` with respect to `x` (the first unknown) only.
//! let v = vec![
//!     F::var(1.0), // Create a variable.
//!     F::cst(2.0), // Create a constant.
//! ];
//! println!("{}", f(&v).deriv()); // prints `4`
//! # assert_eq!(f(&v).deriv(), 4.0);
//! ```
//!
//! Compute higher order derivatives by nesting the generic parameter of `F`. For convenience we
//! provide type aliases for the first 3 orders:
//!
//! ```ignore
//! type F1 = F<f64>
//! type F2 = F<F<f64>>
//! type F3 = F<F<F<f64>>>
//! ```
//!
//! To compute the third order derivative, we can use the `F3` type as follows.
//!
//! ```rust
//! use autodiff::*;
//! // Define a function `f(x) = (x - 1)^3`.
//! let f = |x: F3| (x - 1.0_f64).powi(3);
//!
//! // Compute the 3rd derivative of `f` at `x = 0`.
//! println!("{}", f(F3::var(0.0)).deriv().deriv().deriv()); // prints `6`
//! # assert_eq!(f(F3::var(0.0)).deriv().deriv().deriv(), 6.0);
//! ```

#[cfg(feature = "cgmath")]
pub mod cgmath;
pub mod forward_autodiff;

#[cfg(feature = "cgmath")]
pub use crate::cgmath::*;
pub use forward_autodiff::*;

// Re-export useful traits for performing computations.
pub use num_traits::{Float, FloatConst, NumCast, One, ToPrimitive, Zero};
