///
/// This crate provides a library for performing automatic differentiation in forward mode.
///
/// # Examples
///
/// The following example differentiates a 1D function defined by a closure.
///
/// ```rust
/// use autofloat::*;
/// use num_traits::float::Float;
/// // Define a function `f(x) = e^{-0.5*x^2}`
/// let f = |x: FT<f64>| (-x * x / F::cst(2.0)).exp();
///
/// // Differentiate `f` at zero.
/// println!("{}", diff(f, 0.0)); // prints `0`
/// #   assert_eq!(diff(f, 0.0), 0.0);
/// ```
///
/// To compute the gradient of a function, use the function `grad` as follows:
///
/// ```rust
/// use autofloat::*;
/// use num_traits::float::Float;
/// // Define a function `f(x,y) = x*y^2`
/// let f = |x: &[FT<f64>]| x[0] * x[1] * x[1];
///
/// // Differentiate `f` at `(1,2)`.
/// let g = grad(f, &vec![1.0, 2.0]);
/// println!("({}, {})", g[0], g[1]); // prints `(4, 4)`
/// #   assert_eq!(g, vec![4.0, 4.0]);
/// ```
///
/// Compute a specific derivative of a multi-variable function:
///
/// ```rust
/// use autofloat::*;
/// // Define a function `f(x,y) = x*y^2`.
/// let f = |v: &[FT<f64>]| v[0] * v[1] * v[1];
///
/// // Differentiate `f` at `(1,2)` with respect to `x` (the first unknown) only.
/// let v = vec![
///     F::var(1.0), // Create a variable.
///     F::cst(2.0), // Create a constant.
/// ];
/// println!("{}", f(&v).deriv()); // prints `4`
/// # assert_eq!(f(&v).deriv(), 4.0);
/// ```
///
/// The following example shows how to compute a Jacobian product and evaluate the function at the same time.
///
/// ```
/// use autofloat::*;
/// // Define a function `f(x,y) = (x*y^2, x/y)`.
/// let f = |v: &[FT<f64>]| vec![v[0] * v[1] * v[1], v[0]/v[1]];
///
/// // Compute the Jacobian of `f` at `x = (1,2)` multiplied by a vector `p = (3,4)`.
/// let xp = vec![
///     F1::new(1.0, 3.0),
///     F1::new(2.0, 4.0),
/// ];
/// let jp = f(&xp);
/// println!("({}, {})", jp[0].value(), jp[1].value()); // prints `(4.0, 0.5)`
/// println!("({}, {})", jp[0].deriv(), jp[1].deriv()); // prints `(28.0, 0.5)`
/// # assert_eq!(jp[0].value(), 4.0);
/// # assert_eq!(jp[1].value(), 0.5);
/// # assert_eq!(jp[0].deriv(), 28.0);
/// # assert_eq!(jp[1].deriv(), 0.5);
/// ```
mod autofloat;
pub use autofloat::*;
#[cfg(feature = "nalgebra")]
mod approx;
#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "nalgebra")]
mod simba;
