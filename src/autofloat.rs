// // The code in this repository is based on and was forked from https://github.com/elrnv/autofloat in 2024.
// // The copyright notice is reproduced below:
// //
// // ```
// // Copyright (c) 2018 Egor Larionov
// //
// // Permission is hereby granted, free of charge, to any person obtaining a copy
// // of this software and associated documentation files (the "Software"), to deal
// // in the Software without restriction, including without limitation the rights
// // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // copies of the Software, and to permit persons to whom the Software is
// // furnished to do so, subject to the following conditions:
// //
// // The above copyright notice and this permission notice shall be included in all
// // copies or substantial portions of the Software.
// //
// // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// // SOFTWARE.
// // ```
// //
// // The repository mentioned above was also forked from https://github.com/ibab/rust-ad in 2016.
// // The copyright notice is reproduced below:
// //
// // ```
// // Copyright (c) 2014 Igor Babuschkin
// //
// // Permission is hereby granted, free of charge, to any person obtaining a copy
// // of this software and associated documentation files (the "Software"), to deal
// // in the Software without restriction, including without limitation the rights
// // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // copies of the Software, and to permit persons to whom the Software is
// // furnished to do so, subject to the following conditions:
// //
// // The above copyright notice and this permission notice shall be included in all
// // copies or substantial portions of the Software.
// // ```
// //
// // This crate is licensed under the terms described in the README.md, which is located at the root
// // directory of this crate.

mod add;
mod div;
mod float;
mod mul;
mod num;
mod rem;
mod scalar;
mod sub;

pub use scalar::*;

fn unary_op<T, F, const N: usize>(array: [T; N], func: F) -> [T; N]
where
    T: Clone,
    F: Fn(T) -> T,
{
    let mut result = array;
    for dst in result.iter_mut() {
        *dst = func(dst.clone());
    }
    result
}

fn binary_op<T, F, const N: usize>(lhs: [T; N], rhs: [T; N], func: F) -> [T; N]
where
    T: Clone,
    F: Fn(T, T) -> T,
{
    let mut result = lhs;
    for (dst, src) in result.iter_mut().zip(rhs.into_iter()) {
        *dst = func(dst.clone(), src);
    }
    result
}

// impl<V, D> AutoFloat<V, D>
// where
//     V: Float,
//     D: Mul<V, Output = D> + Sub<Output = D> + Div<V, Output = D>,
// {
//     #[inline]
//     pub(crate) fn atan2_impl(self, other: AutoFloat<V, D>) -> AutoFloat<V, D> {
//         //let self_r = self.reduce_order();
//         //let other_r = other.reduce_order();
//         let self_r = self.x;
//         let other_r = other.x;
//         AutoFloat {
//             x: Float::atan2(self.x, other.x),
//             dx: (self.dx * other_r - other.dx * self_r) / (self_r * self_r + other_r * other_r),
//         }
//     }
// }

// impl<V, D> AutoFloat<V, D> {
//     /// Create a new dual number with value `x` and initial derivative `d`.
//     ///
//     /// This is equivalent to setting the fields of `F` directly.
//     #[inline]
//     pub fn new(x: V, dx: D) -> AutoFloat<V, D> {
//         AutoFloat { x, dx }
//     }
// }

// impl<V, D: Zero> AutoFloat<V, D> {
//     /// Create a new constant.
//     ///
//     /// Use this also to convert from a variable to a constant.
//     #[inline]
//     pub fn cst(x: impl Into<V>) -> AutoFloat<V, D> {
//         AutoFloat {
//             x: x.into(),
//             dx: D::zero(),
//         }
//     }
// }

// impl<V, D: One> AutoFloat<V, D> {
//     /// Create a new variable.
//     ///
//     /// Use this also to convert from a constant to a variable.
//     #[inline]
//     pub fn var(x: impl Into<V>) -> AutoFloat<V, D> {
//         AutoFloat::new(x.into(), D::one())
//     }
// }

// impl<V: Clone, D> AutoFloat<V, D> {
//     /// Get the value of this variable.
//     #[inline]
//     pub fn value(&self) -> V {
//         self.x.clone()
//     }
// }

// impl<V, D: Clone> AutoFloat<V, D> {
//     /// Get the current derivative of this variable.
//     ///
//     /// This will be zero if this `F` is a constant.
//     #[inline]
//     pub fn deriv(&self) -> D {
//         self.dx.clone()
//     }
// }

// impl<V, D> AutoFloat<V, D>
// where
//     Self: Float,
// {
//     /// Raise this number to the `n`'th power.
//     ///
//     /// This is a generic version of `Float::powf`.
//     #[inline]
//     pub fn pow(self, n: impl Into<AutoFloat<V, D>>) -> AutoFloat<V, D> {
//         self.powf(n.into())
//     }
// }

// /// Evaluates the derivative of `f` at `x0`
// ///
// /// # Examples
// ///
// /// ```rust
// /// use autofloat::*;
// /// use num_traits::float::Float;
// /// // Define a function `f(x) = e^{-0.5*x^2}`
// /// let f = |x: FT<f64>| (-x * x / F::cst(2.0)).exp();
// ///
// /// // Differentiate `f` at zero.
// /// println!("{}", diff(f, 0.0)); // prints `0`
// /// # assert_eq!(diff(f, 0.0), 0.0);
// /// ```
// pub fn diff<G>(f: G, x0: f64) -> f64
// where
//     G: FnOnce(FT<f64>) -> FT<f64>,
// {
//     f(F1::var(x0)).deriv()
// }

// /// Evaluates the gradient of `f` at `x0`
// ///
// /// Note that it is much more efficient to use Backward or Reverse-mode automatic
// /// differentiation for computing gradients of scalar valued functions.
// ///
// /// # Examples
// ///
// /// ```rust
// /// use autofloat::*;
// /// // Define a multivariate function `f(x,y) = x*y^2`
// /// let f = |x: &[FT<f64>]| x[0] * x[1] * x[1];
// ///
// /// // Differentiate `f` at `(1,2)`.
// /// let g = grad(f, &vec![1.0, 2.0]);
// /// println!("({}, {})", g[0], g[1]); // prints `(4, 4)`
// /// # assert_eq!(g, vec![4.0, 4.0]);
// pub fn grad<G>(f: G, x0: &[f64]) -> Vec<f64>
// where
//     G: Fn(&[FT<f64>]) -> FT<f64>,
// {
//     let mut nums: Vec<FT<f64>> = x0.iter().map(|&x| F1::cst(x)).collect();

//     let mut results = Vec::new();

//     for i in 0..nums.len() {
//         nums[i] = F1::var(nums[i]);
//         results.push(f(&nums).deriv());
//         nums[i] = F1::cst(nums[i]);
//     }

//     results
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     /// Compare the values and derivatives of two dual numbers for equality.
//     trait DualEq {
//         fn dual_eq(&self, rhs: &Self) -> bool;
//     }

//     impl DualEq for f32 {
//         /// Compare two single precision floats for equality.
//         #[inline]
//         fn dual_eq(&self, rhs: &f32) -> bool {
//             self == rhs
//         }
//     }

//     impl DualEq for f64 {
//         /// Compare two double precision floats for equality.
//         #[inline]
//         fn dual_eq(&self, rhs: &f64) -> bool {
//             self == rhs
//         }
//     }

//     impl<V: PartialEq, D: DualEq> DualEq for AutoFloat<V, D> {
//         /// Compare two `F`s in full, including the derivative part.
//         #[inline]
//         fn dual_eq(&self, rhs: &AutoFloat<V, D>) -> bool {
//             self.x == rhs.x && self.dx.dual_eq(&rhs.dx)
//         }
//     }

//     /// Convenience macro for comparing `F`s in full.
//     macro_rules! assert_dual_eq {
//         ($x:expr, $y:expr $(,)?) => {
//             {
//                 let x = &$x;
//                 let y = &$y;
//                 assert!(F::dual_eq(x, y), "\nleft:  {:?}\nright: {:?}\n", x, y);
//             }
//         };
//         ($x:expr, $y:expr, $($args:tt)+) => {
//             assert!(F::dual_eq(&$x, &$y), $($args)+);
//         };
//     }

//     #[test]
//     fn basic_arithmetic_test() {
//         // Test basic arithmetic on F.
//         let mut x = F1::var(1.0);
//         let y = F1::var(2.0);

//         assert_dual_eq!(-x, AutoFloat::new(-1.0, -1.0)); // negation

//         assert_dual_eq!(x + y, AutoFloat::new(3.0, 2.0)); // addition
//         assert_dual_eq!(x + 2.0, AutoFloat::new(3.0, 1.0)); // addition
//         assert_dual_eq!(2.0 + x, AutoFloat::new(3.0, 1.0)); // addition
//         x += y;
//         assert_dual_eq!(x, AutoFloat::new(3.0, 2.0)); // assign add
//         x += 1.0;
//         assert_dual_eq!(x, AutoFloat::new(4.0, 2.0)); // assign add

//         assert_dual_eq!(x - y, AutoFloat::new(2.0, 1.0)); // subtraction
//         assert_dual_eq!(x - 1.0, AutoFloat::new(3.0, 2.0)); // subtraction
//         assert_dual_eq!(1.0 - x, AutoFloat::new(-3.0, -2.)); // subtraction
//         x -= y;
//         assert_dual_eq!(x, AutoFloat::new(2.0, 1.0)); // subtract assign
//         x -= 1.0;
//         assert_dual_eq!(x, AutoFloat::new(1.0, 1.0)); // subtract assign

//         assert_dual_eq!(x * y, AutoFloat::new(2.0, 3.0)); // multiplication
//         assert_dual_eq!(x * 2.0, AutoFloat::new(2.0, 2.0)); // multiplication
//         assert_dual_eq!(2.0 * x, AutoFloat::new(2.0, 2.0)); // multiplication
//         x *= y;
//         assert_dual_eq!(x, AutoFloat::new(2.0, 3.0)); // multiply assign
//         x *= 2.0;
//         assert_dual_eq!(x, AutoFloat::new(4.0, 6.0)); // multiply assign

//         assert_dual_eq!(x / y, AutoFloat::new(2.0, 2.0)); // division
//         assert_dual_eq!(x / 2.0, AutoFloat::new(2.0, 3.0)); // division
//         assert_dual_eq!(2.0 / x, AutoFloat::new(0.5, -0.75)); // division
//         x /= y;
//         assert_dual_eq!(x, AutoFloat::new(2.0, 2.0)); // divide assign
//         x /= 2.0;
//         assert_dual_eq!(x, AutoFloat::new(1.0, 1.0)); // divide assign

//         assert_dual_eq!(x % y, AutoFloat::new(1.0, 1.0)); // mod
//         assert_dual_eq!(x % 2.0, AutoFloat::new(1.0, 1.0)); // mod
//         assert_dual_eq!(2.0 % x, AutoFloat::new(0.0, -2.0)); // mod
//         x %= y;
//         assert_dual_eq!(x, AutoFloat::new(1.0, 1.0)); // mod assign
//         x %= 2.0;
//         assert_dual_eq!(x, AutoFloat::new(1.0, 1.0)); // mod assign
//     }

//     // Test the min and max functions
//     #[test]
//     fn min_max_test() {
//         // Test basic arithmetic on F.
//         let a = F1::var(1.0);
//         let mut b = AutoFloat::cst(2.0);

//         b = b.min(a);
//         assert_dual_eq!(b, AutoFloat::new(1.0, 1.0));

//         b = AutoFloat::cst(2.0);
//         b = a.min(b);
//         assert_dual_eq!(b, AutoFloat::new(1.0, 1.0));

//         let b = AutoFloat::cst(2.0);

//         let c = a.max(b);
//         assert_dual_eq!(c, AutoFloat::new(2.0, 0.0));

//         // Make sure that our min and max are consistent with the internal implementation to avoid
//         // inconsistencies in the future. In particular we look at tie breaking.

//         let b = AutoFloat::cst(1.0);
//         let minf = a.x.min(b.x);
//         assert_dual_eq!(
//             a.min(b),
//             AutoFloat {
//                 x: minf,
//                 dx: if minf == a.x { a.dx } else { b.dx }
//             }
//         );

//         let maxf = a.x.max(b.x);
//         assert_dual_eq!(
//             a.max(b),
//             AutoFloat {
//                 x: maxf,
//                 dx: if maxf == a.x { a.dx } else { b.dx }
//             }
//         );
//     }

//     // Test iterator sum
//     #[test]
//     fn sum_test() {
//         let v = vec![1.0, 2.0].into_iter();
//         let ad_v = vec![F1::var(1.0), F1::var(2.0)].into_iter();
//         assert_dual_eq!(ad_v.clone().sum::<FT<f64>>(), AutoFloat::new(3.0, 2.0));
//         assert_dual_eq!(v.sum::<FT<f64>>(), AutoFloat::new(3.0, 0.0));
//     }

//     // Test the different ways to compute a derivative of a quadratic.
//     #[test]
//     fn quadratic() {
//         let f1 = |x: FT<f64>| (x - 1.0f64).pow(2.0);
//         let f2 = |x: FT<f64>| (x - 1.0f64) * (x - 1.0f64);

//         // Derivative at 0
//         let dfdx1: FT<f64> = f1(F1::var(0.0));
//         let dfdx2: FT<f64> = f2(F1::var(0.0));

//         assert_dual_eq!(dfdx1, dfdx2);

//         let f1 = |x: FT<f64>| x.pow(2.0);
//         let f2 = |x: FT<f64>| x * x;

//         // Derivative at 0
//         let dfdx1: FT<f64> = f1(F1::var(0.0));
//         let dfdx2: FT<f64> = f2(F1::var(0.0));

//         assert_dual_eq!(dfdx1, dfdx2);
//     }

//     #[test]
//     fn sqrt() {
//         let x = F1::var(0.2).sqrt();
//         assert_dual_eq!(x, F1::new(0.2.sqrt(), 0.5 / 0.2.sqrt()), "{:?}", x);

//         // Test that taking a square root of zero does not produce NaN.
//         // By convention we take 0/0 = 0 here.
//         let x = F1::cst(0.0).sqrt();
//         assert_dual_eq!(x, F1::new(0.0, 0.0), "{:?}", x);
//     }

//     #[test]
//     fn cbrt() {
//         let x = F1::var(0.2).cbrt();
//         assert_dual_eq!(
//             x,
//             F1::new(0.2.cbrt(), 1.0 / (3.0 * 0.2.cbrt() * 0.2.cbrt())),
//             "{:?}",
//             x
//         );

//         // Test that taking a cube root of zero does not produce NaN.
//         // By convention we take 0/0 = 0 here.
//         let x = F1::cst(0.0).cbrt();
//         assert_dual_eq!(x, F1::new(0.0, 0.0), "{:?}", x);
//     }

//     #[test]
//     fn to_degrees() {
//         let x = F1::var(0.2).to_degrees();
//         assert_dual_eq!(x, F1::new(0.2.to_degrees(), 1.0.to_degrees()), "{:?}", x);
//     }

//     #[test]
//     fn to_radians() {
//         let x = F1::var(0.2).to_radians();
//         assert_dual_eq!(x, F1::new(0.2.to_radians(), 1.0.to_radians()), "{:?}", x);
//     }
// }
