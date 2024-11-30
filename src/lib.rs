// The code in this repository is based on and was forked from https://github.com/elrnv/autofloat in 2024.
// The copyright notice is reproduced below:
//
// ```
// Copyright (c) 2018 Egor Larionov
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ```
//
// The repository mentioned above was also forked from https://github.com/ibab/rust-ad in 2016.
// The copyright notice is reproduced below:
//
// ```
// Copyright (c) 2014 Igor Babuschkin
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// ```
//
// This crate is licensed under the terms described in the README.md, which is located at the root
// directory of this crate.

/// This crate provides a library for performing efficient automatic differentiation in forward mode.
///
/// # Usage
///
/// `autofloat`can compute derivatives for single and multivariate functions.
/// The library provides a float-like type `AutoFloat` to automatically compute the derivate while the target function is computed.
///
/// First, make sure that the function for which you want to compute a derivate can handle the `AutoFloat` type (either by generics or explicitly).
/// Then simply instantiate the variables for which you want to compute the derivative and pass them into your target function, that's it!
///
/// Here's a simple example, which computes the gradient of a function wrt. two input variables.
/// The function is implemented using generics and can be used with different floating point types.
///
/// ```rust
/// use autofloat::AutoFloat2;
/// use num_traits::float::FloatCore;
///
/// // Define some target function for which we want to compute the derivative.
/// // This variant is generic in T, but you could also use the `AutoFloat` type directly.
/// fn quadratic_func<T>(x: T, y: T) -> T
/// where
///     T: FloatCore,
/// {
///     (x - T::one()) * (T::from(2).unwrap() * y - T::one())
/// }
///
/// fn main() {
///     // Use AutoFloat2 because we use a 2-dimensional function and we want a 2-dimensional gradient.
///     // The first parameter determines the value of the variable.
///     // The second prameter determines the index of the derivative for this variable within the gradient vector.
///     let x = AutoFloat2::variable(2.25, 0);
///     let y = AutoFloat2::variable(-1.75, 1);
///
///     let result = quadratic_func(x, y);
///
///     println!(
///         "result={} gradient_x={} gradient_y={}",
///         result.x, result.dx[0], result.dx[1]
///     );
/// }
/// ```
mod autofloat;

pub use autofloat::{
    AutoFloat, AutoFloat1, AutoFloat2, AutoFloat3, AutoFloat4, AutoFloat5, AutoFloat6,
};

#[cfg(feature = "approx")]
mod approx;
#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "simba")]
mod simba;

#[cfg(test)]
mod test {

    /// Convenience macro for comparing `AutoFloats`s in full.
    macro_rules! assert_autofloat_eq {
        ($lhs:expr, $rhs:expr) => {
            assert_eq!($lhs.x, $rhs.x);
            assert_eq!($lhs.dx, $rhs.dx);
        };
    }
    pub(crate) use assert_autofloat_eq;

    #[cfg(feature = "float_impl")]
    mod float_impl {
        use core::f64;

        macro_rules! assert_near {
            ($lhs:expr, $rhs:expr, $eps:expr) => {{
                let lhs = &$lhs;
                let rhs = &$rhs;
                let eps = $eps;
                let diff = (*lhs - *rhs).abs();
                assert!(
                    diff < eps,
                    "assertion `(left ~= right) failed` \n\
                 left: {:?} \n\
                 right: {:?} \n\
                 difference {:?} is greater than epsilon {:?}",
                    *lhs,
                    *rhs,
                    diff,
                    eps
                );
            }};
        }
        pub(crate) use assert_near;

        macro_rules! assert_autofloat_near {
            ($lhs:expr, $rhs:expr, $eps:expr) => {{
                assert_near!($lhs.x, $rhs.x, $eps);
                for (&l, &r) in $lhs.dx.iter().zip($rhs.dx.iter()) {
                    assert_near!(l, r, $eps);
                }
            }};
        }
        pub(crate) use assert_autofloat_near;

        pub(crate) fn compute_numeric_derivative<F>(x: f64, func: F) -> f64
        where
            F: Fn(f64) -> f64,
        {
            // use a central differences approach
            let two_eps = f64::EPSILON.sqrt();
            let eps = 0.5 * two_eps;
            let forward = func(x + eps);
            let backward = func(x - eps);

            (forward - backward) / two_eps
        }

        macro_rules! execute_numeric_test {
            ($x:expr, $func:tt) => {{
                let eps = 1e-6;
                let actual = AutoFloat2::variable($x, 0).$func();
                let deriv = compute_numeric_derivative($x, |v| v.$func());
                assert_autofloat_near!(actual, AutoFloat::new($x.$func(), [deriv, 0.0]), eps);
            }};
        }

        pub(crate) use execute_numeric_test;
    }

    #[cfg(feature = "float_impl")]
    pub(crate) use float_impl::*;
}
