// The majority of the code in this module was forked from https://github.com/ibab/rust-ad in 2016.
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

use num_traits::{Float, FloatConst, NumCast, One, ToPrimitive, Zero};
use std::f64;
use std::num::FpCategory;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

#[derive(Copy, Clone, Debug)]
pub struct Num {
    pub x: f64,
    pub dx: f64,
}

/// Panic-less conversion into `f64` type.
impl Into<f64> for Num {
    fn into(self) -> f64 {
        self.x
    }
}

impl Neg for Num {
    type Output = Num;
    fn neg(self) -> Num {
        Num {
            x: -self.x,
            dx: -self.dx,
        }
    }
}

impl Add<Num> for Num {
    type Output = Num;
    fn add(self, _rhs: Num) -> Num {
        Num {
            x: self.x + _rhs.x,
            dx: self.dx + _rhs.dx,
        }
    }
}

impl Sub<Num> for Num {
    type Output = Num;
    fn sub(self, _rhs: Num) -> Num {
        Num {
            x: self.x - _rhs.x,
            dx: self.dx - _rhs.dx,
        }
    }
}

impl Mul<Num> for Num {
    type Output = Num;
    fn mul(self, _rhs: Num) -> Num {
        Num {
            x: self.x * _rhs.x,
            dx: self.dx * _rhs.x + self.x * _rhs.dx,
        }
    }
}

impl Mul<Num> for f64 {
    type Output = Num;
    fn mul(self, _rhs: Num) -> Num {
        Num {
            x: self * _rhs.x,
            dx: self * _rhs.dx,
        }
    }
}

impl Mul<f64> for Num {
    type Output = Num;
    fn mul(self, _rhs: f64) -> Num {
        Num {
            x: self.x * _rhs,
            dx: self.dx * _rhs,
        }
    }
}

impl Div<Num> for Num {
    type Output = Num;
    fn div(self, _rhs: Num) -> Num {
        Num {
            x: self.x / _rhs.x,
            dx: (self.dx * _rhs.x - self.x * _rhs.dx) / (_rhs.x * _rhs.x),
        }
    }
}

impl Rem<Num> for Num {
    type Output = Num;
    fn rem(self, _rhs: Num) -> Num {
        panic!("Remainder not implemented")
    }
}

impl Default for Num {
    fn default() -> Self {
        Num {
            x: f64::default(),
            dx: 0.0,
        }
    }
}

impl PartialEq<Num> for Num {
    fn eq(&self, _rhs: &Num) -> bool {
        self.x == _rhs.x
    }
}

impl PartialOrd<Num> for Num {
    fn partial_cmp(&self, other: &Num) -> Option<::std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.x, &other.x)
    }
}

impl ToPrimitive for Num {
    fn to_i64(&self) -> Option<i64> {
        self.x.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.x.to_u64()
    }
    fn to_isize(&self) -> Option<isize> {
        self.x.to_isize()
    }
    fn to_i8(&self) -> Option<i8> {
        self.x.to_i8()
    }
    fn to_i16(&self) -> Option<i16> {
        self.x.to_i16()
    }
    fn to_i32(&self) -> Option<i32> {
        self.x.to_i32()
    }
    fn to_usize(&self) -> Option<usize> {
        self.x.to_usize()
    }
    fn to_u8(&self) -> Option<u8> {
        self.x.to_u8()
    }
    fn to_u16(&self) -> Option<u16> {
        self.x.to_u16()
    }
    fn to_u32(&self) -> Option<u32> {
        self.x.to_u32()
    }
    fn to_f32(&self) -> Option<f32> {
        self.x.to_f32()
    }
    fn to_f64(&self) -> Option<f64> {
        self.x.to_f64()
    }
}

impl NumCast for Num {
    fn from<T: ToPrimitive>(n: T) -> Option<Num> {
        let _x = n.to_f64();
        match _x {
            Some(x) => Some(Num { x: x, dx: 0.0 }),
            None => None,
        }
    }
}

impl Zero for Num {
    fn zero() -> Num {
        Num { x: 0.0, dx: 0.0 }
    }
    fn is_zero(&self) -> bool {
        self.x.is_zero()
    }
}

impl One for Num {
    fn one() -> Num {
        Num { x: 1.0, dx: 0.0 }
    }
}

impl ::num_traits::Num for Num {
    type FromStrRadixErr = ::num_traits::ParseFloatError;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(src, radix).map(|x| Num { x: x, dx: 0.0 })
    }
}

impl FloatConst for Num {
    fn E() -> Num {
        Num {
            x: f64::consts::E,
            dx: 0.0,
        }
    }
    fn FRAC_1_PI() -> Num {
        Num {
            x: f64::consts::FRAC_1_PI,
            dx: 0.0,
        }
    }
    fn FRAC_1_SQRT_2() -> Num {
        Num {
            x: f64::consts::FRAC_1_SQRT_2,
            dx: 0.0,
        }
    }
    fn FRAC_2_PI() -> Num {
        Num {
            x: f64::consts::FRAC_2_PI,
            dx: 0.0,
        }
    }
    fn FRAC_2_SQRT_PI() -> Num {
        Num {
            x: f64::consts::FRAC_2_SQRT_PI,
            dx: 0.0,
        }
    }
    fn FRAC_PI_2() -> Num {
        Num {
            x: f64::consts::FRAC_PI_2,
            dx: 0.0,
        }
    }
    fn FRAC_PI_3() -> Num {
        Num {
            x: f64::consts::FRAC_PI_3,
            dx: 0.0,
        }
    }
    fn FRAC_PI_4() -> Num {
        Num {
            x: f64::consts::FRAC_PI_4,
            dx: 0.0,
        }
    }
    fn FRAC_PI_6() -> Num {
        Num {
            x: f64::consts::FRAC_PI_6,
            dx: 0.0,
        }
    }
    fn FRAC_PI_8() -> Num {
        Num {
            x: f64::consts::FRAC_PI_8,
            dx: 0.0,
        }
    }
    fn LN_10() -> Num {
        Num {
            x: f64::consts::LN_10,
            dx: 0.0,
        }
    }
    fn LN_2() -> Num {
        Num {
            x: f64::consts::LN_2,
            dx: 0.0,
        }
    }
    fn LOG10_E() -> Num {
        Num {
            x: f64::consts::LOG10_E,
            dx: 0.0,
        }
    }
    fn LOG2_E() -> Num {
        Num {
            x: f64::consts::LOG2_E,
            dx: 0.0,
        }
    }
    fn PI() -> Num {
        Num {
            x: f64::consts::PI,
            dx: 0.0,
        }
    }
    fn SQRT_2() -> Num {
        Num {
            x: f64::consts::SQRT_2,
            dx: 0.0,
        }
    }
}

impl Float for Num {
    fn nan() -> Num {
        Num {
            x: f64::NAN,
            dx: 0.0,
        }
    }
    fn infinity() -> Num {
        Num {
            x: f64::INFINITY,
            dx: 0.0,
        }
    }
    fn neg_infinity() -> Num {
        Num {
            x: f64::NEG_INFINITY,
            dx: 0.0,
        }
    }
    fn neg_zero() -> Num {
        Num { x: -0.0, dx: 0.0 }
    }
    fn min_value() -> Num {
        Num {
            x: f64::MIN,
            dx: 0.0,
        }
    }
    fn min_positive_value() -> Num {
        Num {
            x: f64::MIN_POSITIVE,
            dx: 0.0,
        }
    }
    fn max_value() -> Num {
        Num {
            x: f64::MAX,
            dx: 0.0,
        }
    }
    fn is_nan(self) -> bool {
        self.x.is_nan() || self.dx.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.x.is_infinite() || self.dx.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.dx.is_finite()
    }
    fn is_normal(self) -> bool {
        self.x.is_normal() && self.dx.is_normal()
    }
    fn classify(self) -> FpCategory {
        self.x.classify()
    }

    fn floor(self) -> Num {
        Num {
            x: self.x.floor(),
            dx: self.dx,
        }
    }
    fn ceil(self) -> Num {
        Num {
            x: self.x.ceil(),
            dx: self.dx,
        }
    }
    fn round(self) -> Num {
        Num {
            x: self.x.round(),
            dx: self.dx,
        }
    }
    fn trunc(self) -> Num {
        Num {
            x: self.x.trunc(),
            dx: self.dx,
        }
    }
    fn fract(self) -> Num {
        Num {
            x: self.x.fract(),
            dx: self.dx,
        }
    }
    fn abs(self) -> Num {
        Num {
            x: self.x.abs(),
            dx: if self.x >= 0.0 { self.dx } else { -self.dx },
        }
    }
    fn signum(self) -> Num {
        Num {
            x: self.x.signum(),
            dx: 0.0,
        }
    }
    fn is_sign_positive(self) -> bool {
        self.x.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.x.is_sign_negative()
    }
    fn mul_add(self, a: Num, b: Num) -> Num {
        self * a + b
    }
    fn recip(self) -> Num {
        Num {
            x: self.x.recip(),
            dx: -self.dx / (self.x * self.x),
        }
    }
    fn powi(self, n: i32) -> Num {
        Num {
            x: self.x.powi(n),
            dx: self.dx * n as f64 * self.x.powi(n - 1),
        }
    }
    fn powf(self, n: Num) -> Num {
        Num {
            x: Float::powf(self.x, n.x),
            dx: (Float::ln(self.x) * n.dx + n.x * self.dx / self.x) * Float::powf(self.x, n.x),
        }
    }
    fn sqrt(self) -> Num {
        Num {
            x: self.x.sqrt(),
            dx: self.dx * 0.5 / self.x.sqrt(),
        }
    }

    fn exp(self) -> Num {
        Num {
            x: Float::exp(self.x),
            dx: self.dx * Float::exp(self.x),
        }
    }
    fn exp2(self) -> Num {
        Num {
            x: Float::exp2(self.x),
            dx: self.dx * Float::ln(2.0) * Float::exp(self.x),
        }
    }
    fn ln(self) -> Num {
        Num {
            x: Float::ln(self.x),
            dx: self.dx * self.x.recip(),
        }
    }
    fn log(self, b: Num) -> Num {
        Num {
            x: Float::log(self.x, b.x),
            dx: -Float::ln(self.x) * b.dx / (b.x * Float::powi(Float::ln(b.x), 2))
                + self.dx / (self.x * Float::ln(b.x)),
        }
    }
    fn log2(self) -> Num {
        Float::log(self, Num { x: 2.0, dx: 0.0 })
    }
    fn log10(self) -> Num {
        Float::log(self, Num { x: 10.0, dx: 0.0 })
    }
    fn max(self, other: Num) -> Num {
        Num {
            x: Float::max(self.x, other.x),
            dx: 0.0,
        }
    }
    fn min(self, other: Num) -> Num {
        Num {
            x: Float::min(self.x, other.x),
            dx: 0.0,
        }
    }
    fn abs_sub(self, other: Num) -> Num {
        if self > other {
            Num {
                x: Float::abs_sub(self.x, other.x),
                dx: (self - other).dx,
            }
        } else {
            Num { x: 0.0, dx: 0.0 }
        }
    }
    fn cbrt(self) -> Num {
        Num {
            x: Float::cbrt(self.x),
            dx: 1.0 / 3.0 * self.x.powf(-2.0 / 3.0) * self.dx,
        }
    }
    fn hypot(self, other: Num) -> Num {
        Float::sqrt(Float::powi(self, 2) + Float::powi(other, 2))
    }
    fn sin(self) -> Num {
        Num {
            x: Float::sin(self.x),
            dx: self.dx * Float::cos(self.x),
        }
    }
    fn cos(self) -> Num {
        Num {
            x: Float::cos(self.x),
            dx: -self.dx * Float::sin(self.x),
        }
    }
    fn tan(self) -> Num {
        let t = Float::tan(self.x);
        Num {
            x: t,
            dx: self.dx * (t * t + 1.0),
        }
    }
    fn asin(self) -> Num {
        Num {
            x: Float::asin(self.x),
            dx: self.dx / Float::sqrt(1.0 - Float::powi(self.x, 2)),
        }
    }
    fn acos(self) -> Num {
        Num {
            x: Float::acos(self.x),
            dx: -self.dx / Float::sqrt(1.0 - Float::powi(self.x, 2)),
        }
    }
    fn atan(self) -> Num {
        Num {
            x: Float::atan(self.x),
            dx: self.dx / Float::sqrt(Float::powi(self.x, 2) + 1.0),
        }
    }
    fn atan2(self, other: Num) -> Num {
        Num {
            x: Float::atan2(self.x, other.x),
            dx: (other.x * self.dx - self.x * other.dx)
                / (Float::powi(self.x, 2) + Float::powi(other.x, 2)),
        }
    }
    fn sin_cos(self) -> (Num, Num) {
        let (s, c) = Float::sin_cos(self.x);
        let sn = Num {
            x: s,
            dx: self.dx * c,
        };
        let cn = Num {
            x: c,
            dx: -self.dx * s,
        };
        (sn, cn)
    }
    fn exp_m1(self) -> Num {
        Num {
            x: Float::exp_m1(self.x),
            dx: self.dx * Float::exp(self.x),
        }
    }
    fn ln_1p(self) -> Num {
        Num {
            x: Float::ln_1p(self.x),
            dx: self.dx / (self.x + 1.0),
        }
    }
    fn sinh(self) -> Num {
        Num {
            x: Float::sinh(self.x),
            dx: self.dx * Float::cosh(self.x),
        }
    }
    fn cosh(self) -> Num {
        Num {
            x: Float::cosh(self.x),
            dx: self.dx * Float::sinh(self.x),
        }
    }
    fn tanh(self) -> Num {
        Num {
            x: Float::tanh(self.x),
            dx: self.dx * (1.0 - Float::powi(Float::tanh(self.x), 2)),
        }
    }
    fn asinh(self) -> Num {
        Num {
            x: Float::asinh(self.x),
            dx: self.dx * (Float::powi(self.x, 2) + 1.0),
        }
    }
    fn acosh(self) -> Num {
        Num {
            x: Float::acosh(self.x),
            dx: self.dx * (Float::powi(self.x, 2) - 1.0),
        }
    }
    fn atanh(self) -> Num {
        Num {
            x: Float::atanh(self.x),
            dx: self.dx * (-Float::powi(self.x, 2) + 1.0),
        }
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    fn epsilon() -> Num {
        Num {
            x: f64::EPSILON,
            dx: 0.0,
        }
    }
    fn to_degrees(self) -> Num {
        Num {
            x: Float::to_degrees(self.x),
            dx: 0.0,
        }
    }
    fn to_radians(self) -> Num {
        Num {
            x: Float::to_radians(self.x),
            dx: 0.0,
        }
    }
}

impl Num {
    /// Create a new constant. Use this also to convert from a variable to a constant.
    pub fn cst<T: Into<f64>>(x: T) -> Num {
        Num {
            x: x.into(),
            dx: 0.0,
        }
    }

    /// Create a new variable. Use this also to convert from a constant to a variable.
    pub fn var<T: Into<f64>>(x: T) -> Num {
        Num {
            x: x.into(),
            dx: 1.0,
        }
    }

    /// Get the value of this variable.
    pub fn value(&self) -> f64 {
        self.x
    }

    /// Get the current derivative of this variable. This will be zero if this `Num` is a
    /// constant.
    pub fn deriv(&self) -> f64 {
        self.dx
    }
}

/// Evaluates the derivative of `f` at `x0`
///
/// # Examples
///
/// ```rust
/// # extern crate autodiff;
/// # use autodiff::*;
/// # fn main() {
///     // Define a function `f(x) = e^{-0.5*x^2}`
///     let f = |x: Num| (-x * x / Num::cst(2.0)).exp();
///
///     // Differentiate `f` at zero.
///     println!("{}", diff(f, 0.0)); // prints `0`
/// #   assert_eq!(diff(f, 0.0), 0.0);
/// # }
/// ```
pub fn diff<F>(f: F, x0: f64) -> f64
where
    F: FnOnce(Num) -> Num,
{
    f(Num::var(x0)).deriv()
}

/// Evaluates the gradient of `f` at `x0`
///
/// # Examples
///
/// ```rust
/// # extern crate autodiff;
/// # use autodiff::*;
/// # fn main() {
///     // Define a multivariate function `f(x,y) = x*y^2`
///     let f = |x: &[Num]| x[0] * x[1] * x[1];
///
///     // Differentiate `f` at `(1,2)`.
///     let g = grad(f, &vec![1.0, 2.0]);
///     println!("({}, {})", g[0], g[1]); // prints `(4, 4)`
/// #   assert_eq!(g, vec![4.0, 4.0]);
/// # }
pub fn grad<F>(f: F, x0: &[f64]) -> Vec<f64>
where
    F: Fn(&[Num]) -> Num,
{
    let mut nums: Vec<Num> = x0.iter().map(|&x| Num::cst(x)).collect();

    let mut results = Vec::new();

    for i in 0..nums.len() {
        nums[i] = Num::var(nums[i]);
        results.push(f(&nums).deriv());
        nums[i] = Num::cst(nums[i]);
    }

    results
}
