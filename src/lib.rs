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
#[cfg(feature = "approx")]
mod approx;
#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "simba")]
mod simba;

fn unary_op<T, V, F, const N: usize>(array: [T; N], func: F) -> [V; N]
where
    T: Clone,
    F: Fn(T) -> V,
    V: Copy + Default,
{
    let mut result = [V::default(); N];
    for (dst, src) in result.iter_mut().zip(array.into_iter()) {
        *dst = func(src);
    }
    result
}

fn binary_op<T, U, V, F, const N: usize>(lhs: [T; N], rhs: [U; N], func: F) -> [V; N]
where
    F: Fn(T, U) -> V,
    V: Copy + Default,
{
    let mut result = [V::default(); N];
    for (dst, (l, r)) in result.iter_mut().zip(lhs.into_iter().zip(rhs.into_iter())) {
        *dst = func(l, r);
    }
    result
}

#[cfg(test)]
mod test {
    /// Convenience macro for comparing `AutoFloats`s in full.
    macro_rules! assert_autofloat_eq {
        ($lhs:expr, $rhs:expr $(,)?) => {
            assert_eq!($lhs.x, $rhs.x);
            assert_eq!($lhs.dx, $rhs.dx);
        };
    }
    pub(crate) use assert_autofloat_eq;
}
