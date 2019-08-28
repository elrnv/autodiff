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

use num_traits::{Float, FloatConst, Num, NumCast, One, ToPrimitive, Zero};
use std::f64;
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Copy, Clone, Debug)]
pub struct F {
    pub x: f64,
    pub dx: f64,
}

/// Panic-less conversion into `f64` type.
impl Into<f64> for F {
    #[inline]
    fn into(self) -> f64 {
        self.x
    }
}

/// Panic-less conversion into `f32` type.
impl Into<f32> for F {
    #[inline]
    fn into(self) -> f32 {
        self.x as f32
    }
}

impl Neg for F {
    type Output = F;
    #[inline]
    fn neg(self) -> F {
        F {
            x: -self.x,
            dx: -self.dx,
        }
    }
}

impl Add<F> for F {
    type Output = F;
    #[inline]
    fn add(self, rhs: F) -> F {
        F {
            x: self.x + rhs.x,
            dx: self.dx + rhs.dx,
        }
    }
}

impl Add<f64> for F {
    type Output = F;
    #[inline]
    fn add(self, rhs: f64) -> F {
        F {
            x: self.x + rhs,
            dx: self.dx,
        }
    }
}

impl Add<F> for f64 {
    type Output = F;
    #[inline]
    fn add(self, rhs: F) -> F {
        F {
            x: self + rhs.x,
            dx: rhs.dx,
        }
    }
}

impl Add<f32> for F {
    type Output = F;
    #[inline]
    fn add(self, rhs: f32) -> F {
        self + rhs as f64
    }
}

impl Add<F> for f32 {
    type Output = F;
    #[inline]
    fn add(self, rhs: F) -> F {
        self as f64 + rhs
    }
}

impl AddAssign for F {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        self.x += rhs.x;
        self.dx += rhs.dx;
    }
}

impl AddAssign<f64> for F {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        self.x += rhs;
    }
}

impl AddAssign<f32> for F {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        *self += rhs as f64;
    }
}

impl Sub<F> for F {
    type Output = F;
    #[inline]
    fn sub(self, rhs: F) -> F {
        F {
            x: self.x - rhs.x,
            dx: self.dx - rhs.dx,
        }
    }
}

impl Sub<f64> for F {
    type Output = F;
    #[inline]
    fn sub(self, rhs: f64) -> F {
        F {
            x: self.x - rhs,
            dx: self.dx,
        }
    }
}

impl Sub<F> for f64 {
    type Output = F;
    #[inline]
    fn sub(self, rhs: F) -> F {
        F {
            x: self - rhs.x,
            dx: -rhs.dx,
        }
    }
}

impl Sub<f32> for F {
    type Output = F;
    #[inline]
    fn sub(self, rhs: f32) -> F {
        self - rhs as f64
    }
}

impl Sub<F> for f32 {
    type Output = F;
    #[inline]
    fn sub(self, rhs: F) -> F {
        self as f64 - rhs
    }
}

impl SubAssign for F {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        self.x -= rhs.x;
        self.dx -= rhs.dx;
    }
}

impl SubAssign<f64> for F {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        self.x -= rhs;
    }
}

impl SubAssign<f32> for F {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        *self -= rhs as f64;
    }
}

/*
 * Multiplication
 */

impl Mul<F> for F {
    type Output = F;
    #[inline]
    fn mul(self, rhs: F) -> F {
        F {
            x: self.x * rhs.x,
            dx: self.dx * rhs.x + self.x * rhs.dx,
        }
    }
}

// Multiply by double precision floats (treated as constants)

impl Mul<F> for f64 {
    type Output = F;
    #[inline]
    fn mul(self, rhs: F) -> F {
        // self is treated as a constant
        F {
            x: self * rhs.x,
            dx: self * rhs.dx,
        }
    }
}

impl Mul<f64> for F {
    type Output = F;
    #[inline]
    fn mul(self, rhs: f64) -> F {
        // rhs is treated as a constant
        F {
            x: self.x * rhs,
            dx: self.dx * rhs,
        }
    }
}

// Multiply by single precision floats

impl Mul<F> for f32 {
    type Output = F;
    #[inline]
    fn mul(self, rhs: F) -> F {
        self as f64 * rhs
    }
}

impl Mul<f32> for F {
    type Output = F;
    #[inline]
    fn mul(self, rhs: f32) -> F {
        self * rhs as f64
    }
}

// Multiply assign operators

impl MulAssign for F {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for F {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        // rhs is treated as a constant
        self.x *= rhs;
        self.dx *= rhs;
    }
}

impl MulAssign<f32> for F {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self *= rhs as f64;
    }
}

// MulAssign<F> for f64 is not implemented deliberately, because this operation erases the
// tracking of the derivative information.

/*
 * Division
 */

impl Div<F> for F {
    type Output = F;
    #[inline]
    fn div(self, rhs: F) -> F {
        F {
            x: self.x / rhs.x,
            dx: (self.dx * rhs.x - self.x * rhs.dx) / (rhs.x * rhs.x),
        }
    }
}

// Division by double precision floats

impl Div<f64> for F {
    type Output = F;
    #[inline]
    fn div(self, rhs: f64) -> F {
        F {
            x: self.x / rhs,
            dx: self.dx / rhs,
        }
    }
}

impl Div<F> for f64 {
    type Output = F;
    #[inline]
    fn div(self, rhs: F) -> F {
        F {
            x: self / rhs.x,
            dx: -self * rhs.dx / (rhs.x * rhs.x),
        }
    }
}

// Division by single precision floats

impl Div<f32> for F {
    type Output = F;
    #[inline]
    fn div(self, rhs: f32) -> F {
        self / rhs as f64
    }
}

impl Div<F> for f32 {
    type Output = F;
    #[inline]
    fn div(self, rhs: F) -> F {
        self as f64 / rhs
    }
}

impl DivAssign for F {
    #[inline]
    fn div_assign(&mut self, rhs: F) {
        *self = *self / rhs; // reuse quotient rule implementation
    }
}

impl DivAssign<f64> for F {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.dx /= rhs;
    }
}

impl DivAssign<f32> for F {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self /= rhs as f64;
    }
}

// DivAssign<F> for f64 is not implemented deliberately, because this operation erases the
// tracking of the derivative information.

/*
 * Remainder function
 */

impl Rem<F> for F {
    type Output = F;
    #[inline]
    fn rem(self, rhs: F) -> F {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self.x % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: self.dx - (self.x / rhs.x).trunc() * rhs.dx,
        }
    }
}

impl Rem<f64> for F {
    type Output = F;
    #[inline]
    fn rem(self, rhs: f64) -> F {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self.x % rhs, // x % y = x - [x/|y|]*|y|
            dx: self.dx,
        }
    }
}

impl Rem<F> for f64 {
    type Output = F;
    #[inline]
    fn rem(self, rhs: F) -> F {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: -(self / rhs.x).trunc() * rhs.dx,
        }
    }
}

impl RemAssign for F {
    #[inline]
    fn rem_assign(&mut self, rhs: F) {
        *self = *self % rhs; // resuse non-trivial implementation
    }
}

impl RemAssign<f64> for F {
    #[inline]
    fn rem_assign(&mut self, rhs: f64) {
        *self = *self % rhs; // resuse non-trivial implementation
    }
}

impl RemAssign<f32> for F {
    #[inline]
    fn rem_assign(&mut self, rhs: f32) {
        *self = *self % rhs as f64; // resuse non-trivial implementation
    }
}

impl Default for F {
    #[inline]
    fn default() -> Self {
        F {
            x: f64::default(),
            dx: 0.0,
        }
    }
}

impl PartialEq<F> for F {
    #[inline]
    fn eq(&self, rhs: &F) -> bool {
        self.x == rhs.x
    }
}

impl PartialOrd<F> for F {
    #[inline]
    fn partial_cmp(&self, other: &F) -> Option<::std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.x, &other.x)
    }
}

impl ToPrimitive for F {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.x.to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.x.to_u64()
    }
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        self.x.to_isize()
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        self.x.to_i8()
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        self.x.to_i16()
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        self.x.to_i32()
    }
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        self.x.to_usize()
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        self.x.to_u8()
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        self.x.to_u16()
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        self.x.to_u32()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.x.to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.x.to_f64()
    }
}

impl NumCast for F {
    fn from<T: ToPrimitive>(n: T) -> Option<F> {
        let _x = n.to_f64();
        match _x {
            Some(x) => Some(F { x: x, dx: 0.0 }),
            None => None,
        }
    }
}

impl Zero for F {
    #[inline]
    fn zero() -> F {
        F { x: 0.0, dx: 0.0 }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.x.is_zero()
    }
}

impl One for F {
    #[inline]
    fn one() -> F {
        F { x: 1.0, dx: 0.0 }
    }
}

impl Num for F {
    type FromStrRadixErr = ::num_traits::ParseFloatError;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(src, radix).map(|x| F::cst(x))
    }
}

impl FloatConst for F {
    #[inline]
    fn E() -> F {
        F {
            x: f64::consts::E,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_1_PI() -> F {
        F {
            x: f64::consts::FRAC_1_PI,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_1_SQRT_2() -> F {
        F {
            x: f64::consts::FRAC_1_SQRT_2,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_2_PI() -> F {
        F {
            x: f64::consts::FRAC_2_PI,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_2_SQRT_PI() -> F {
        F {
            x: f64::consts::FRAC_2_SQRT_PI,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_PI_2() -> F {
        F {
            x: f64::consts::FRAC_PI_2,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_PI_3() -> F {
        F {
            x: f64::consts::FRAC_PI_3,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_PI_4() -> F {
        F {
            x: f64::consts::FRAC_PI_4,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_PI_6() -> F {
        F {
            x: f64::consts::FRAC_PI_6,
            dx: 0.0,
        }
    }
    #[inline]
    fn FRAC_PI_8() -> F {
        F {
            x: f64::consts::FRAC_PI_8,
            dx: 0.0,
        }
    }
    #[inline]
    fn LN_10() -> F {
        F {
            x: f64::consts::LN_10,
            dx: 0.0,
        }
    }
    #[inline]
    fn LN_2() -> F {
        F {
            x: f64::consts::LN_2,
            dx: 0.0,
        }
    }
    #[inline]
    fn LOG10_E() -> F {
        F {
            x: f64::consts::LOG10_E,
            dx: 0.0,
        }
    }
    #[inline]
    fn LOG2_E() -> F {
        F {
            x: f64::consts::LOG2_E,
            dx: 0.0,
        }
    }
    #[inline]
    fn PI() -> F {
        F {
            x: f64::consts::PI,
            dx: 0.0,
        }
    }
    #[inline]
    fn SQRT_2() -> F {
        F {
            x: f64::consts::SQRT_2,
            dx: 0.0,
        }
    }
}

impl Float for F {
    #[inline]
    fn nan() -> F {
        F {
            x: f64::NAN,
            dx: 0.0,
        }
    }
    #[inline]
    fn infinity() -> F {
        F {
            x: f64::INFINITY,
            dx: 0.0,
        }
    }
    #[inline]
    fn neg_infinity() -> F {
        F {
            x: f64::NEG_INFINITY,
            dx: 0.0,
        }
    }
    #[inline]
    fn neg_zero() -> F {
        F { x: -0.0, dx: 0.0 }
    }
    #[inline]
    fn min_value() -> F {
        F {
            x: f64::MIN,
            dx: 0.0,
        }
    }
    #[inline]
    fn min_positive_value() -> F {
        F {
            x: f64::MIN_POSITIVE,
            dx: 0.0,
        }
    }
    #[inline]
    fn max_value() -> F {
        F {
            x: f64::MAX,
            dx: 0.0,
        }
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.x.is_nan() || self.dx.is_nan()
    }
    #[inline]
    fn is_infinite(self) -> bool {
        self.x.is_infinite() || self.dx.is_infinite()
    }
    #[inline]
    fn is_finite(self) -> bool {
        self.x.is_finite() && self.dx.is_finite()
    }
    #[inline]
    fn is_normal(self) -> bool {
        self.x.is_normal() && self.dx.is_normal()
    }
    #[inline]
    fn classify(self) -> FpCategory {
        self.x.classify()
    }

    #[inline]
    fn floor(self) -> F {
        F {
            x: self.x.floor(),
            dx: self.dx,
        }
    }
    #[inline]
    fn ceil(self) -> F {
        F {
            x: self.x.ceil(),
            dx: self.dx,
        }
    }
    #[inline]
    fn round(self) -> F {
        F {
            x: self.x.round(),
            dx: self.dx,
        }
    }
    #[inline]
    fn trunc(self) -> F {
        F {
            x: self.x.trunc(),
            dx: self.dx,
        }
    }
    #[inline]
    fn fract(self) -> F {
        F {
            x: self.x.fract(),
            dx: self.dx,
        }
    }
    #[inline]
    fn abs(self) -> F {
        F {
            x: self.x.abs(),
            dx: if self.x >= 0.0 { self.dx } else { -self.dx },
        }
    }
    #[inline]
    fn signum(self) -> F {
        F {
            x: self.x.signum(),
            dx: 0.0,
        }
    }
    #[inline]
    fn is_sign_positive(self) -> bool {
        self.x.is_sign_positive()
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        self.x.is_sign_negative()
    }
    #[inline]
    fn mul_add(self, a: F, b: F) -> F {
        self * a + b
    }
    #[inline]
    fn recip(self) -> F {
        F {
            x: self.x.recip(),
            dx: -self.dx / (self.x * self.x),
        }
    }
    #[inline]
    fn powi(self, n: i32) -> F {
        F {
            x: self.x.powi(n),
            dx: self.dx * n as f64 * self.x.powi(n - 1),
        }
    }
    #[inline]
    fn powf(self, n: F) -> F {
        F {
            x: Float::powf(self.x, n.x),
            dx: (Float::ln(self.x) * n.dx + n.x * self.dx / self.x) * Float::powf(self.x, n.x),
        }
    }
    #[inline]
    fn sqrt(self) -> F {
        F {
            x: self.x.sqrt(),
            dx: self.dx * 0.5 / self.x.sqrt(),
        }
    }

    #[inline]
    fn exp(self) -> F {
        F {
            x: Float::exp(self.x),
            dx: self.dx * Float::exp(self.x),
        }
    }
    #[inline]
    fn exp2(self) -> F {
        F {
            x: Float::exp2(self.x),
            dx: self.dx * Float::ln(2.0) * Float::exp(self.x),
        }
    }
    #[inline]
    fn ln(self) -> F {
        F {
            x: Float::ln(self.x),
            dx: self.dx * self.x.recip(),
        }
    }
    #[inline]
    fn log(self, b: F) -> F {
        F {
            x: Float::log(self.x, b.x),
            dx: -Float::ln(self.x) * b.dx / (b.x * Float::powi(Float::ln(b.x), 2))
                + self.dx / (self.x * Float::ln(b.x)),
        }
    }
    #[inline]
    fn log2(self) -> F {
        Float::log(self, F { x: 2.0, dx: 0.0 })
    }
    #[inline]
    fn log10(self) -> F {
        Float::log(self, F { x: 10.0, dx: 0.0 })
    }
    #[inline]
    fn max(self, other: F) -> F {
        if self.x < other.x {
            other
        } else {
            self
        }
    }
    #[inline]
    fn min(self, other: F) -> F {
        if self.x > other.x {
            other
        } else {
            self
        }
    }
    #[inline]
    fn abs_sub(self, other: F) -> F {
        if self > other {
            F {
                x: Float::abs_sub(self.x, other.x),
                dx: (self - other).dx,
            }
        } else {
            F { x: 0.0, dx: 0.0 }
        }
    }
    #[inline]
    fn cbrt(self) -> F {
        F {
            x: Float::cbrt(self.x),
            dx: 1.0 / 3.0 * self.x.powf(-2.0 / 3.0) * self.dx,
        }
    }
    #[inline]
    fn hypot(self, other: F) -> F {
        Float::sqrt(Float::powi(self, 2) + Float::powi(other, 2))
    }
    #[inline]
    fn sin(self) -> F {
        F {
            x: Float::sin(self.x),
            dx: self.dx * Float::cos(self.x),
        }
    }
    #[inline]
    fn cos(self) -> F {
        F {
            x: Float::cos(self.x),
            dx: -self.dx * Float::sin(self.x),
        }
    }
    #[inline]
    fn tan(self) -> F {
        let t = Float::tan(self.x);
        F {
            x: t,
            dx: self.dx * (t * t + 1.0),
        }
    }
    #[inline]
    fn asin(self) -> F {
        F {
            x: Float::asin(self.x),
            dx: self.dx / Float::sqrt(1.0 - Float::powi(self.x, 2)),
        }
    }
    #[inline]
    fn acos(self) -> F {
        F {
            x: Float::acos(self.x),
            dx: -self.dx / Float::sqrt(1.0 - Float::powi(self.x, 2)),
        }
    }
    #[inline]
    fn atan(self) -> F {
        F {
            x: Float::atan(self.x),
            dx: self.dx / Float::sqrt(Float::powi(self.x, 2) + 1.0),
        }
    }
    #[inline]
    fn atan2(self, other: F) -> F {
        F {
            x: Float::atan2(self.x, other.x),
            dx: (other.x * self.dx - self.x * other.dx)
                / (Float::powi(self.x, 2) + Float::powi(other.x, 2)),
        }
    }
    #[inline]
    fn sin_cos(self) -> (F, F) {
        let (s, c) = Float::sin_cos(self.x);
        let sn = F {
            x: s,
            dx: self.dx * c,
        };
        let cn = F {
            x: c,
            dx: -self.dx * s,
        };
        (sn, cn)
    }
    #[inline]
    fn exp_m1(self) -> F {
        F {
            x: Float::exp_m1(self.x),
            dx: self.dx * Float::exp(self.x),
        }
    }
    #[inline]
    fn ln_1p(self) -> F {
        F {
            x: Float::ln_1p(self.x),
            dx: self.dx / (self.x + 1.0),
        }
    }
    #[inline]
    fn sinh(self) -> F {
        F {
            x: Float::sinh(self.x),
            dx: self.dx * Float::cosh(self.x),
        }
    }
    #[inline]
    fn cosh(self) -> F {
        F {
            x: Float::cosh(self.x),
            dx: self.dx * Float::sinh(self.x),
        }
    }
    #[inline]
    fn tanh(self) -> F {
        F {
            x: Float::tanh(self.x),
            dx: self.dx * (1.0 - Float::powi(Float::tanh(self.x), 2)),
        }
    }
    #[inline]
    fn asinh(self) -> F {
        F {
            x: Float::asinh(self.x),
            dx: self.dx * (Float::powi(self.x, 2) + 1.0),
        }
    }
    #[inline]
    fn acosh(self) -> F {
        F {
            x: Float::acosh(self.x),
            dx: self.dx * (Float::powi(self.x, 2) - 1.0),
        }
    }
    #[inline]
    fn atanh(self) -> F {
        F {
            x: Float::atanh(self.x),
            dx: self.dx * (-Float::powi(self.x, 2) + 1.0),
        }
    }
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    #[inline]
    fn epsilon() -> F {
        F {
            x: f64::EPSILON,
            dx: 0.0,
        }
    }
    #[inline]
    fn to_degrees(self) -> F {
        F {
            x: Float::to_degrees(self.x),
            dx: 0.0,
        }
    }
    #[inline]
    fn to_radians(self) -> F {
        F {
            x: Float::to_radians(self.x),
            dx: 0.0,
        }
    }
}

impl std::iter::Sum for F {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut res = Self::zero();
        for x in iter {
            res += x;
        }
        res
    }
}

impl std::iter::Sum<f64> for F {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = f64>,
    {
        iter.map(|x| F::cst(x)).sum()
    }
}

impl F {
    /// Create a new constant. Use this also to convert from a variable to a constant.
    /// This constructor panics if `x` cannot be converted to `f64`.
    #[inline]
    pub fn cst<T: ToPrimitive>(x: T) -> F {
        F {
            x: x.to_f64().unwrap(),
            dx: 0.0,
        }
    }

    /// Create a new variable. Use this also to convert from a constant to a variable.
    /// This constructor panics if `x` cannot be converted to `f64`.
    #[inline]
    pub fn var<T: ToPrimitive>(x: T) -> F {
        F {
            x: x.to_f64().unwrap(),
            dx: 1.0,
        }
    }

    /// Compare two `F`s in full, including the derivative part.
    pub fn full_eq(&self, rhs: &F) -> bool {
        self.x == rhs.x && self.dx == rhs.dx
    }

    /// Get the value of this variable.
    #[inline]
    pub fn value(&self) -> f64 {
        self.x
    }

    /// Get the current derivative of this variable. This will be zero if this `F` is a
    /// constant.
    #[inline]
    pub fn deriv(&self) -> f64 {
        self.dx
    }
}

/// Evaluates the derivative of `f` at `x0`
///
/// # Examples
///
/// ```rust
/// # use autodiff::*;
/// # fn main() {
///     // Define a function `f(x) = e^{-0.5*x^2}`
///     let f = |x: F| (-x * x / F::cst(2.0)).exp();
///
///     // Differentiate `f` at zero.
///     println!("{}", diff(f, 0.0)); // prints `0`
/// #   assert_eq!(diff(f, 0.0), 0.0);
/// # }
/// ```
pub fn diff<G>(f: G, x0: f64) -> f64
where
    G: FnOnce(F) -> F,
{
    f(F::var(x0)).deriv()
}

/// Evaluates the gradient of `f` at `x0`
///
/// # Examples
///
/// ```rust
/// # use autodiff::*;
/// # fn main() {
///     // Define a multivariate function `f(x,y) = x*y^2`
///     let f = |x: &[F]| x[0] * x[1] * x[1];
///
///     // Differentiate `f` at `(1,2)`.
///     let g = grad(f, &vec![1.0, 2.0]);
///     println!("({}, {})", g[0], g[1]); // prints `(4, 4)`
/// #   assert_eq!(g, vec![4.0, 4.0]);
/// # }
pub fn grad<G>(f: G, x0: &[f64]) -> Vec<f64>
where
    G: Fn(&[F]) -> F,
{
    let mut nums: Vec<F> = x0.iter().map(|&x| F::cst(x)).collect();

    let mut results = Vec::new();

    for i in 0..nums.len() {
        nums[i] = F::var(nums[i]);
        results.push(f(&nums).deriv());
        nums[i] = F::cst(nums[i]);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience macro for comparing `F`s in full.
    macro_rules! assert_full_eq {
        ($x:expr, $y:expr) => {
            assert!(F::full_eq(&$x, &$y));
        };
    }

    #[test]
    fn basic_arithmetic_test() {
        // Test basic arithmetic on F.
        let mut x = F::var(1.0);
        let y = F::var(2.0);

        assert_full_eq!(-x, F { x: -1.0, dx: -1.0 }); // negation

        assert_full_eq!(x + y, F { x: 3.0, dx: 2.0 }); // addition
        assert_full_eq!(x + 2.0, F { x: 3.0, dx: 1.0 }); // addition
        assert_full_eq!(2.0 + x, F { x: 3.0, dx: 1.0 }); // addition
        x += y;
        assert_full_eq!(x, F { x: 3.0, dx: 2.0 }); // assign add
        x += 1.0;
        assert_full_eq!(x, F { x: 4.0, dx: 2.0 }); // assign add

        assert_full_eq!(x - y, F { x: 2.0, dx: 1.0 }); // subtraction
        assert_full_eq!(x - 1.0, F { x: 3.0, dx: 2.0 }); // subtraction
        assert_full_eq!(1.0 - x, F { x: -3.0, dx: -2.0 }); // subtraction
        x -= y;
        assert_full_eq!(x, F { x: 2.0, dx: 1.0 }); // subtract assign
        x -= 1.0;
        assert_full_eq!(x, F { x: 1.0, dx: 1.0 }); // subtract assign

        assert_full_eq!(x * y, F { x: 2.0, dx: 3.0 }); // multiplication
        assert_full_eq!(x * 2.0, F { x: 2.0, dx: 2.0 }); // multiplication
        assert_full_eq!(2.0 * x, F { x: 2.0, dx: 2.0 }); // multiplication
        x *= y;
        assert_full_eq!(x, F { x: 2.0, dx: 3.0 }); // multiply assign
        x *= 2.0;
        assert_full_eq!(x, F { x: 4.0, dx: 6.0 }); // multiply assign

        assert_full_eq!(x / y, F { x: 2.0, dx: 2.0 }); // division
        assert_full_eq!(x / 2.0, F { x: 2.0, dx: 3.0 }); // division
        assert_full_eq!(2.0 / x, F { x: 0.5, dx: -0.75 }); // division
        x /= y;
        assert_full_eq!(x, F { x: 2.0, dx: 2.0 }); // divide assign
        x /= 2.0;
        assert_full_eq!(x, F { x: 1.0, dx: 1.0 }); // divide assign

        assert_full_eq!(x % y, F { x: 1.0, dx: 1.0 }); // mod
        assert_full_eq!(x % 2.0, F { x: 1.0, dx: 1.0 }); // mod
        assert_full_eq!(2.0 % x, F { x: 0.0, dx: -2.0 }); // mod
        x %= y;
        assert_full_eq!(x, F { x: 1.0, dx: 1.0 }); // mod assign
        x %= 2.0;
        assert_full_eq!(x, F { x: 1.0, dx: 1.0 }); // mod assign
    }

    // Test the min and max functions
    #[test]
    fn min_max_test() {
        // Test basic arithmetic on F.
        let a = F::var(1.0);
        let mut b = F::cst(2.0);

        b = b.min(a);
        assert_full_eq!(b, F { x: 1.0, dx: 1.0 });

        b = F::cst(2.0);
        b = a.min(b);
        assert_full_eq!(b, F { x: 1.0, dx: 1.0 });

        let b = F::cst(2.0);

        let c = a.max(b);
        assert_full_eq!(c, F { x: 2.0, dx: 0.0 });

        // Make sure that our min and max are consistent with the internal implementation to avoid
        // inconsistencies in the future. In particular we look at tie breaking.

        let b = F::cst(1.0);
        let minf = a.x.min(b.x);
        assert_full_eq!(
            a.min(b),
            F {
                x: minf,
                dx: if minf == a.x { a.dx } else { b.dx }
            }
        );

        let maxf = a.x.max(b.x);
        assert_full_eq!(
            a.max(b),
            F {
                x: maxf,
                dx: if maxf == a.x { a.dx } else { b.dx }
            }
        );
    }

    // Test iterator sum
    #[test]
    fn sum_test() {
        let v = vec![1.0, 2.0].into_iter();
        let ad_v = vec![F::var(1.0), F::var(2.0)].into_iter();
        assert_full_eq!(ad_v.clone().sum(), F { x: 3.0, dx: 2.0 });
        assert_full_eq!(v.sum::<F>(), F { x: 3.0, dx: 0.0 });
    }
}
