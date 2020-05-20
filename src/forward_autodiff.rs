// Some of the code in this module was forked from https://github.com/ibab/rust-ad in 2016.
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

use num_traits::{Float, FloatConst, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::f64;
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// A Helper trait to drop the highest order derivative.
///
/// This provides the mechanism to propagate higher order derivatives.
pub trait ReduceOrder {
    type Reduced;
    /// Produce an equivalent dual number with a lower differentiation order.
    fn reduce_order(&self) -> Self::Reduced;
    /// Get the value of the reduced dual.
    fn reduced_value(r: &Self::Reduced) -> f64;
}

/// First order differentiator.
pub type F1 = F<f64>;

/// Second order differentiator.
pub type F2 = F<F1>;

/// Third order differentiator.
pub type F3 = F<F2>;

/// A generic forward differentiation `Dual` number.
///
/// The derivative is generic to support higher order differentiation.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct F<D> {
    /// The value of the variable
    pub x: f64,
    /// The derivative of the variable
    pub dx: D,
}

// Base case
impl ReduceOrder for F1 {
    type Reduced = f64;
    fn reduce_order(&self) -> Self::Reduced {
        self.x
    }
    fn reduced_value(r: &Self::Reduced) -> f64 {
        *r
    }
}

// Recursive case
impl<D: ReduceOrder> ReduceOrder for F<D> {
    type Reduced = F<D::Reduced>;
    fn reduce_order(&self) -> Self::Reduced {
        F {
            x: self.x,
            dx: self.dx.reduce_order(),
        }
    }
    fn reduced_value(r: &Self::Reduced) -> f64 {
        r.x
    }
}

impl<D> Into<f64> for F<D> {
    /// Convert the value into an `f64` type.
    #[inline]
    fn into(self) -> f64 {
        self.x
    }
}

impl<D> Into<f32> for F<D> {
    /// Convert the value into an `f32` type.
    #[inline]
    fn into(self) -> f32 {
        self.x as f32
    }
}

impl<D: Zero> From<f64> for F<D> {
    /// Construct a constant from an `f64`.
    #[inline]
    fn from(x: f64) -> F<D> {
        F::cst(x)
    }
}

impl<D: Zero> From<f32> for F<D> {
    /// Construct a constant from an `f32`.
    #[inline]
    fn from(x: f32) -> F<D> {
        F::cst(x)
    }
}

impl<D: Neg<Output = D>> Neg for F<D> {
    type Output = F<D>;
    #[inline]
    fn neg(self) -> F<D> {
        F {
            x: -self.x,
            dx: -self.dx,
        }
    }
}

impl<D: Add<D, Output = D>> Add<F<D>> for F<D> {
    type Output = F<D>;
    #[inline]
    fn add(self, rhs: F<D>) -> F<D> {
        F {
            x: self.x + rhs.x,
            dx: self.dx + rhs.dx,
        }
    }
}

impl<D> Add<f64> for F<D> {
    type Output = F<D>;
    #[inline]
    fn add(self, rhs: f64) -> F<D> {
        F {
            x: self.x + rhs,
            dx: self.dx,
        }
    }
}

impl Add<F1> for f64 {
    type Output = F1;
    #[inline]
    fn add(self, rhs: F1) -> F1 {
        F {
            x: self + rhs.x,
            dx: rhs.dx,
        }
    }
}

impl<D> Add<f32> for F<D> {
    type Output = F<D>;
    #[inline]
    fn add(self, rhs: f32) -> F<D> {
        self + rhs as f64
    }
}

impl Add<F1> for f32 {
    type Output = F1;
    #[inline]
    fn add(self, rhs: F1) -> F1 {
        self as f64 + rhs
    }
}

impl<D: AddAssign> AddAssign for F<D> {
    #[inline]
    fn add_assign(&mut self, rhs: F<D>) {
        self.x += rhs.x;
        self.dx += rhs.dx;
    }
}

impl<D> AddAssign<f64> for F<D> {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        self.x += rhs;
    }
}

impl<D> AddAssign<f32> for F<D> {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        *self += rhs as f64;
    }
}

impl<D: Sub<D, Output = D>> Sub<F<D>> for F<D> {
    type Output = F<D>;
    #[inline]
    fn sub(self, rhs: F<D>) -> F<D> {
        F {
            x: self.x - rhs.x,
            dx: self.dx - rhs.dx,
        }
    }
}

impl<D> Sub<f64> for F<D> {
    type Output = F<D>;
    #[inline]
    fn sub(self, rhs: f64) -> F<D> {
        F {
            x: self.x - rhs,
            ..self
        }
    }
}

impl Sub<F1> for f64 {
    type Output = F1;
    #[inline]
    fn sub(self, rhs: F1) -> F1 {
        F {
            x: self - rhs.x,
            dx: -rhs.dx,
        }
    }
}

impl Sub<f32> for F1 {
    type Output = F1;
    #[inline]
    fn sub(self, rhs: f32) -> F1 {
        self - rhs as f64
    }
}

impl Sub<F1> for f32 {
    type Output = F1;
    #[inline]
    fn sub(self, rhs: F1) -> F1 {
        self as f64 - rhs
    }
}

impl<D: SubAssign> SubAssign for F<D> {
    #[inline]
    fn sub_assign(&mut self, rhs: F<D>) {
        self.x -= rhs.x;
        self.dx -= rhs.dx;
    }
}

impl<D> SubAssign<f64> for F<D> {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        self.x -= rhs;
    }
}

impl<D> SubAssign<f32> for F<D> {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        *self -= rhs as f64;
    }
}

/*
 * Multiplication
 */

impl<D, R> Mul<F<D>> for F<D>
where
    D: Copy + Mul<R, Output = D> + Add<Output = D>,
    F<D>: ReduceOrder<Reduced = R>,
{
    type Output = F<D>;
    #[inline]
    fn mul(self, rhs: F<D>) -> F<D> {
        F {
            x: self.x * rhs.x,
            dx: self.dx * rhs.reduce_order() + rhs.dx * self.reduce_order(),
        }
    }
}

// Multiply by double precision floats (treated as constants)

impl Mul<F1> for f64 {
    type Output = F1;
    #[inline]
    fn mul(self, rhs: F1) -> F1 {
        // self is treated as a constant
        F {
            x: self * rhs.x,
            dx: self * rhs.dx,
        }
    }
}

impl<D: Mul<f64, Output = D>> Mul<f64> for F<D> {
    type Output = F<D>;
    #[inline]
    fn mul(self, rhs: f64) -> F<D> {
        // rhs is treated as a constant
        F {
            x: self.x * rhs,
            dx: self.dx * rhs,
        }
    }
}

// Multiply by single precision floats

impl Mul<F1> for f32 {
    type Output = F1;
    #[inline]
    fn mul(self, rhs: F1) -> F1 {
        self as f64 * rhs
    }
}

impl<D: Mul<f64, Output = D>> Mul<f32> for F<D> {
    type Output = F<D>;
    #[inline]
    fn mul(self, rhs: f32) -> F<D> {
        self * rhs as f64
    }
}

// Multiply assign operators

impl<D, R> MulAssign for F<D>
where
    D: MulAssign<R> + Mul<R, Output = D> + AddAssign,
    F<D>: ReduceOrder<Reduced = R>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: F<D>) {
        // Product rule
        self.dx *= rhs.reduce_order();
        self.dx += rhs.dx * self.reduce_order();
        self.x *= rhs.x;
    }
}

impl<D: MulAssign<f64>> MulAssign<f64> for F<D> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        // rhs is treated as a constant
        self.x *= rhs;
        self.dx *= rhs;
    }
}

impl<D: MulAssign<f64>> MulAssign<f32> for F<D> {
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

impl<D, R> Div<F<D>> for F<D>
where
    D: Copy + Div<R, Output = D> + Mul<R, Output = D> + Sub<Output = D>,
    F<D>: ReduceOrder<Reduced = R>,
    R: Copy + Mul<Output = R>,
{
    type Output = F<D>;
    #[inline]
    fn div(self, rhs: F<D>) -> F<D> {
        let rhs_r = rhs.reduce_order();
        F {
            x: self.x / rhs.x,
            dx: (self.dx * rhs_r - rhs.dx * self.reduce_order()) / (rhs_r * rhs_r),
        }
    }
}

// Division by double precision floats

impl<D: Div<f64, Output = D>> Div<f64> for F<D> {
    type Output = F<D>;
    #[inline]
    fn div(self, rhs: f64) -> F<D> {
        F {
            x: self.x / rhs,
            dx: self.dx / rhs,
        }
    }
}

impl Div<F1> for f64 {
    type Output = F1;
    #[inline]
    fn div(self, rhs: F1) -> F1 {
        F {
            x: self / rhs.x,
            dx: -self * rhs.dx / (rhs.x * rhs.x),
        }
    }
}

// Division by single precision floats

impl<D> Div<f32> for F<D>
where
    D: Div<f64, Output = D> + Mul<f64, Output = D> + Sub<Output = D>,
{
    type Output = F<D>;
    #[inline]
    fn div(self, rhs: f32) -> F<D> {
        self / rhs as f64
    }
}

impl Div<F1> for f32 {
    type Output = F1;
    #[inline]
    fn div(self, rhs: F1) -> F1 {
        self as f64 / rhs
    }
}

impl<D, R> DivAssign for F<D>
where
    D: Mul<R, Output = D> + DivAssign<R> + SubAssign,
    F<D>: ReduceOrder<Reduced = R>,
    R: Copy + Mul<Output = R> + Div<R, Output = R>,
{
    #[inline]
    fn div_assign(&mut self, rhs: F<D>) {
        let rhs_r = rhs.reduce_order();
        self.dx /= rhs_r;
        self.dx -= rhs.dx * (self.reduce_order() / (rhs_r * rhs_r));
        self.x /= rhs.x;
    }
}

impl<D: DivAssign<f64>> DivAssign<f64> for F<D> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.dx /= rhs;
    }
}

impl<D: DivAssign<f64>> DivAssign<f32> for F<D> {
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

impl<D, R> Rem<F<D>> for F<D>
where
    D: Copy + Mul<R, Output = D> + Sub<Output = D>,
    F<D>: ReduceOrder<Reduced = R>,
    R: Float,
{
    type Output = F<D>;
    #[inline]
    fn rem(self, rhs: F<D>) -> F<D> {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self.x % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: self.dx - rhs.dx * (self.reduce_order() / rhs.reduce_order()).trunc(),
        }
    }
}

impl<D: Rem<f64, Output = D>> Rem<f64> for F<D> {
    type Output = F<D>;
    #[inline]
    fn rem(self, rhs: f64) -> F<D> {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self.x % rhs, // x % y = x - [x/|y|]*|y|
            dx: self.dx % rhs,
        }
    }
}

impl Rem<F1> for f64 {
    type Output = F1;
    #[inline]
    fn rem(self, rhs: F1) -> F1 {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: -(self / rhs.x).trunc() * rhs.dx,
        }
    }
}

impl<D, R> RemAssign for F<D>
where
    D: Copy + Mul<R, Output = D> + SubAssign,
    F<D>: ReduceOrder<Reduced = R>,
    R: Float,
{
    #[inline]
    fn rem_assign(&mut self, rhs: F<D>) {
        // x % y = x - [x/|y|]*|y|
        self.dx -= rhs.dx * (self.reduce_order() / rhs.reduce_order()).trunc();
        self.x %= rhs.x;
    }
}

impl<D: RemAssign<f64>> RemAssign<f64> for F<D> {
    #[inline]
    fn rem_assign(&mut self, rhs: f64) {
        self.x %= rhs;
        self.dx %= rhs;
    }
}

impl<D: RemAssign<f64>> RemAssign<f32> for F<D> {
    #[inline]
    fn rem_assign(&mut self, rhs: f32) {
        self.x %= rhs as f64;
        self.dx %= rhs as f64;
    }
}

impl<D: Default> Default for F<D> {
    #[inline]
    fn default() -> Self {
        F {
            x: f64::default(),
            dx: D::default(),
        }
    }
}

impl<D> ToPrimitive for F<D> {
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

impl<D: Zero> NumCast for F<D> {
    fn from<T: ToPrimitive>(n: T) -> Option<F<D>> {
        let x = n.to_f64();
        match x {
            Some(x) => Some(F { x, dx: D::zero() }),
            None => None,
        }
    }
}

impl<D: Zero> FromPrimitive for F<D> {
    #[inline]
    fn from_isize(n: isize) -> Option<Self> {
        FromPrimitive::from_isize(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_i8(n: i8) -> Option<Self> {
        FromPrimitive::from_i8(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_i16(n: i16) -> Option<Self> {
        FromPrimitive::from_i16(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_i32(n: i32) -> Option<Self> {
        FromPrimitive::from_i32(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        FromPrimitive::from_i64(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_i128(n: i128) -> Option<Self> {
        FromPrimitive::from_i128(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_usize(n: usize) -> Option<Self> {
        FromPrimitive::from_usize(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_u8(n: u8) -> Option<Self> {
        FromPrimitive::from_u8(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_u16(n: u16) -> Option<Self> {
        FromPrimitive::from_u16(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_u32(n: u32) -> Option<Self> {
        FromPrimitive::from_u32(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        FromPrimitive::from_u64(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_u128(n: u128) -> Option<Self> {
        FromPrimitive::from_u128(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        FromPrimitive::from_f32(n).map(F::cst::<f64>)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        FromPrimitive::from_f64(n).map(F::cst::<f64>)
    }
}

impl<D: Zero> Zero for F<D> {
    #[inline]
    fn zero() -> F<D> {
        F {
            x: 0.0,
            dx: D::zero(),
        }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.x.is_zero()
    }
}

impl<D: Copy + std::fmt::Debug + Zero + Mul<<F<D> as ReduceOrder>::Reduced, Output = D>> One
    for F<D>
where
    F<D>: ReduceOrder,
{
    #[inline]
    fn one() -> F<D> {
        F {
            x: 1.0,
            dx: D::zero(),
        }
    }
}

impl<D, R> Num for F<D>
where
    D: Copy
        + std::fmt::Debug
        + Zero
        + PartialEq
        + Sub<Output = D>
        + Mul<f64, Output = D>
        + Mul<R, Output = D>
        + Div<f64, Output = D>
        + Div<R, Output = D>,
    F<D>: ReduceOrder<Reduced = R>,
    R: Float,
{
    type FromStrRadixErr = ::num_traits::ParseFloatError;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(src, radix).map(F::cst)
    }
}

impl<D: Zero> FloatConst for F<D> {
    #[inline]
    fn E() -> F<D> {
        F::cst(f64::consts::E)
    }
    #[inline]
    fn FRAC_1_PI() -> F<D> {
        F::cst(f64::consts::FRAC_1_PI)
    }
    #[inline]
    fn FRAC_1_SQRT_2() -> F<D> {
        F::cst(f64::consts::FRAC_1_SQRT_2)
    }
    #[inline]
    fn FRAC_2_PI() -> F<D> {
        F::cst(f64::consts::FRAC_2_PI)
    }
    #[inline]
    fn FRAC_2_SQRT_PI() -> F<D> {
        F::cst(f64::consts::FRAC_2_SQRT_PI)
    }
    #[inline]
    fn FRAC_PI_2() -> F<D> {
        F::cst(f64::consts::FRAC_PI_2)
    }
    #[inline]
    fn FRAC_PI_3() -> F<D> {
        F::cst(f64::consts::FRAC_PI_3)
    }
    #[inline]
    fn FRAC_PI_4() -> F<D> {
        F::cst(f64::consts::FRAC_PI_4)
    }
    #[inline]
    fn FRAC_PI_6() -> F<D> {
        F::cst(f64::consts::FRAC_PI_6)
    }
    #[inline]
    fn FRAC_PI_8() -> F<D> {
        F::cst(f64::consts::FRAC_PI_8)
    }
    #[inline]
    fn LN_10() -> F<D> {
        F::cst(f64::consts::LN_10)
    }
    #[inline]
    fn LN_2() -> F<D> {
        F::cst(f64::consts::LN_2)
    }
    #[inline]
    fn LOG10_E() -> F<D> {
        F::cst(f64::consts::LOG10_E)
    }
    #[inline]
    fn LOG2_E() -> F<D> {
        F::cst(f64::consts::LOG2_E)
    }
    #[inline]
    fn PI() -> F<D> {
        F::cst(f64::consts::PI)
    }
    #[inline]
    fn SQRT_2() -> F<D> {
        F::cst(f64::consts::SQRT_2)
    }
}

impl<D, R> Float for F<D>
where
    D: std::fmt::Debug
        + Float
        + Zero
        + Neg<Output = D>
        + Mul<R, Output = D>
        + Mul<f64, Output = D>
        + Add<Output = D>
        + Div<f64, Output = D>
        + Div<R, Output = D>
        + Sub<Output = D>
        + Copy
        + PartialOrd,
    F<D>: ReduceOrder<Reduced = R>,
    R: Float + Mul<f64, Output = R>,
{
    #[inline]
    fn nan() -> F<D> {
        F::cst(f64::NAN)
    }
    #[inline]
    fn infinity() -> F<D> {
        F::cst(f64::INFINITY)
    }
    #[inline]
    fn neg_infinity() -> F<D> {
        F::cst(f64::NEG_INFINITY)
    }
    #[inline]
    fn neg_zero() -> F<D> {
        F::cst(-0.0)
    }
    #[inline]
    fn min_value() -> F<D> {
        F::cst(f64::MIN)
    }
    #[inline]
    fn min_positive_value() -> F<D> {
        F::cst(f64::MIN_POSITIVE)
    }
    #[inline]
    fn max_value() -> F<D> {
        F::cst(f64::MAX)
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
    fn floor(self) -> F<D> {
        F {
            x: self.x.floor(),
            dx: self.dx,
        }
    }
    #[inline]
    fn ceil(self) -> F<D> {
        F {
            x: self.x.ceil(),
            dx: self.dx,
        }
    }
    #[inline]
    fn round(self) -> F<D> {
        F {
            x: self.x.round(),
            dx: self.dx,
        }
    }
    #[inline]
    fn trunc(self) -> F<D> {
        F {
            x: self.x.trunc(),
            dx: self.dx,
        }
    }
    #[inline]
    fn fract(self) -> F<D> {
        F {
            x: self.x.fract(),
            dx: self.dx,
        }
    }
    #[inline]
    fn abs(self) -> F<D> {
        F {
            x: self.x.abs(),
            dx: if self.x >= 0.0 { self.dx } else { -self.dx },
        }
    }
    #[inline]
    fn signum(self) -> F<D> {
        F::cst(self.x.signum())
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
    fn mul_add(self, a: F<D>, b: F<D>) -> F<D> {
        self * a + b
    }
    #[inline]
    fn recip(self) -> F<D> {
        F {
            x: self.x.recip(),
            dx: -self.dx / (self.reduce_order() * self.reduce_order()),
        }
    }
    #[inline]
    fn powi(self, n: i32) -> F<D> {
        F {
            x: self.x.powi(n),
            dx: self.dx * (self.reduce_order().powi(n - 1) * n as f64),
        }
    }
    #[inline]
    fn powf(self, n: F<D>) -> F<D> {
        let self_r = self.reduce_order();
        let n_r = n.reduce_order();
        // Avoid imaginary values in the ln.
        let dn = if n.dx.is_zero() {
            D::zero()
        } else {
            n.dx * Float::ln(self_r)
        };

        let x = Float::powf(self_r, n_r);

        // Avoid division by zero.
        let x_df = if self.x == 0.0 && x.is_zero() {
            D::zero()
        } else {
            self.dx * (x * n_r / self_r)
        };

        F {
            x: Self::reduced_value(&x),
            dx: dn * x + x_df,
        }
    }
    #[inline]
    fn sqrt(self) -> F<D> {
        F {
            x: self.x.sqrt(),
            dx: self.dx / (self.reduce_order().sqrt() * 2.0),
        }
    }

    #[inline]
    fn exp(self) -> F<D> {
        F {
            x: Float::exp(self.x),
            dx: self.dx * Float::exp(self.reduce_order()),
        }
    }
    #[inline]
    fn exp2(self) -> F<D> {
        F {
            x: Float::exp2(self.x),
            dx: self.dx * (Float::exp2(self.reduce_order()) * Float::ln(2.0)),
        }
    }
    #[inline]
    fn ln(self) -> F<D> {
        F {
            x: Float::ln(self.x),
            dx: self.dx * self.reduce_order().recip(),
        }
    }
    #[inline]
    fn log(self, b: F<D>) -> F<D> {
        let s_r = self.reduce_order();
        let b_r = b.reduce_order();
        F {
            x: Float::log(self.x, b.x),
            dx: b.dx * (-Float::ln(s_r)) / (b_r * Float::powi(Float::ln(b_r), 2))
                + self.dx / (s_r * Float::ln(b_r)),
        }
    }
    #[inline]
    fn log2(self) -> F<D> {
        Float::log(self, F::cst(2.0))
    }
    #[inline]
    fn log10(self) -> F<D> {
        Float::log(self, F::cst(10.0))
    }
    #[inline]
    fn max(self, other: F<D>) -> F<D> {
        if self.x < other.x {
            other
        } else {
            self
        }
    }
    #[inline]
    fn min(self, other: F<D>) -> F<D> {
        if self.x > other.x {
            other
        } else {
            self
        }
    }
    #[inline]
    fn abs_sub(self, other: F<D>) -> F<D> {
        if self > other {
            F {
                x: Float::abs_sub(self.x, other.x),
                dx: (self - other).dx,
            }
        } else {
            F::cst(0.0)
        }
    }
    #[inline]
    fn cbrt(self) -> F<D> {
        F {
            x: Float::cbrt(self.x),
            dx: self.dx * (self.reduce_order().powf(R::from(-2.0 / 3.0).unwrap()) * (1.0 / 3.0)),
        }
    }
    #[inline]
    fn hypot(self, other: F<D>) -> F<D> {
        Float::sqrt(Float::powi(self, 2) + Float::powi(other, 2))
    }
    #[inline]
    fn sin(self) -> F<D> {
        F {
            x: Float::sin(self.x),
            dx: self.dx * Float::cos(self.reduce_order()),
        }
    }
    #[inline]
    fn cos(self) -> F<D> {
        F {
            x: Float::cos(self.x),
            dx: -self.dx * Float::sin(self.reduce_order()),
        }
    }
    #[inline]
    fn tan(self) -> F<D> {
        let t = Float::tan(self.reduce_order());
        F {
            x: Self::reduced_value(&t),
            dx: self.dx * (t * t + R::one()),
        }
    }
    #[inline]
    fn asin(self) -> F<D> {
        F {
            x: Float::asin(self.x),
            dx: self.dx / Float::sqrt(R::one() - Float::powi(self.reduce_order(), 2)),
        }
    }
    #[inline]
    fn acos(self) -> F<D> {
        F {
            x: Float::acos(self.x),
            dx: -self.dx / Float::sqrt(R::one() - Float::powi(self.reduce_order(), 2)),
        }
    }
    #[inline]
    fn atan(self) -> F<D> {
        F {
            x: Float::atan(self.x),
            dx: self.dx / (Float::powi(self.reduce_order(), 2) + R::one()),
        }
    }
    #[inline]
    fn atan2(self, other: F<D>) -> F<D> {
        let self_r = self.reduce_order();
        let other_r = other.reduce_order();
        F {
            x: Float::atan2(self.x, other.x),
            dx: (self.dx * other_r - other.dx * self_r)
                / (Float::powi(self_r, 2) + Float::powi(other_r, 2)),
        }
    }
    #[inline]
    fn sin_cos(self) -> (F<D>, F<D>) {
        let (s, c) = Float::sin_cos(self.reduce_order());
        let sn = F {
            x: Self::reduced_value(&s),
            dx: self.dx * c,
        };
        let cn = F {
            x: Self::reduced_value(&c),
            dx: self.dx * (-s),
        };
        (sn, cn)
    }
    #[inline]
    fn exp_m1(self) -> F<D> {
        F {
            x: Float::exp_m1(self.x),
            dx: self.dx * Float::exp(self.reduce_order()),
        }
    }
    #[inline]
    fn ln_1p(self) -> F<D> {
        F {
            x: Float::ln_1p(self.x),
            dx: self.dx / (self.reduce_order() + R::one()),
        }
    }
    #[inline]
    fn sinh(self) -> F<D> {
        F {
            x: Float::sinh(self.x),
            dx: self.dx * Float::cosh(self.reduce_order()),
        }
    }
    #[inline]
    fn cosh(self) -> F<D> {
        F {
            x: Float::cosh(self.x),
            dx: self.dx * Float::sinh(self.reduce_order()),
        }
    }
    #[inline]
    fn tanh(self) -> F<D> {
        F {
            x: Float::tanh(self.x),
            dx: self.dx * (R::one() - Float::powi(Float::tanh(self.reduce_order()), 2)),
        }
    }
    #[inline]
    fn asinh(self) -> F<D> {
        F {
            x: Float::asinh(self.x),
            dx: self.dx / (Float::powi(self.reduce_order(), 2) + R::one()).sqrt(),
        }
    }
    #[inline]
    fn acosh(self) -> F<D> {
        F {
            x: Float::acosh(self.x),
            dx: self.dx / (Float::powi(self.reduce_order(), 2) - R::one()).sqrt(),
        }
    }
    #[inline]
    fn atanh(self) -> F<D> {
        F {
            x: Float::atanh(self.x),
            dx: self.dx / (-Float::powi(self.reduce_order(), 2) + R::one()),
        }
    }
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    #[inline]
    fn epsilon() -> F<D> {
        F::cst(f64::EPSILON)
    }
    #[inline]
    fn to_degrees(self) -> F<D> {
        F::cst(Float::to_degrees(self.x))
    }
    #[inline]
    fn to_radians(self) -> F<D> {
        F::cst(Float::to_radians(self.x))
    }
}

impl<D: AddAssign + Zero> std::iter::Sum for F<D> {
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

impl<D: AddAssign + Zero> std::iter::Sum<f64> for F<D> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = f64>,
    {
        iter.map(F::cst).sum()
    }
}

impl<D: Zero> F<D> {
    /// Create a new constant.
    ///
    /// Use this also to convert from a variable to a constant.  This constructor panics if `x`
    /// cannot be converted to `f64`.
    #[inline]
    pub fn cst<T: ToPrimitive>(x: T) -> F<D> {
        F {
            x: x.to_f64().unwrap(),
            dx: D::zero(),
        }
    }
}

impl<D: One> F<D> {
    /// Create a new variable.
    ///
    /// Use this also to convert from a constant to a variable.  This constructor panics if `x`
    /// cannot be converted to `f64`.
    #[inline]
    pub fn var<T: ToPrimitive>(x: T) -> F<D> {
        F {
            x: x.to_f64().unwrap(),
            dx: D::one(),
        }
    }
}

impl<D> F<D> {
    /// Get the value of this variable.
    #[inline]
    pub fn value(&self) -> f64 {
        self.x
    }
}

impl<D: Copy> F<D> {
    /// Get the current derivative of this variable. This will be zero if this `F` is a
    /// constant.
    #[inline]
    pub fn deriv(&self) -> D {
        self.dx
    }
}

impl<D> F<D>
where
    Self: Float,
{
    /// Raise this number to the `n`'th power.
    ///
    /// This is a generic version of `Float::powf`.
    #[inline]
    pub fn pow(self, n: impl Into<F<D>>) -> F<D> {
        self.powf(n.into())
    }
}

/// Evaluates the derivative of `f` at `x0`
///
/// # Examples
///
/// ```rust
/// use autodiff::*;
/// // Define a function `f(x) = e^{-0.5*x^2}`
/// let f = |x: F1| (-x * x / F::cst(2.0)).exp();
///
/// // Differentiate `f` at zero.
/// println!("{}", diff(f, 0.0)); // prints `0`
/// # assert_eq!(diff(f, 0.0), 0.0);
/// ```
pub fn diff<G>(f: G, x0: f64) -> f64
where
    G: FnOnce(F1) -> F1,
{
    f(F1::var(x0)).deriv()
}

/// Evaluates the gradient of `f` at `x0`
///
/// # Examples
///
/// ```rust
/// use autodiff::*;
/// // Define a multivariate function `f(x,y) = x*y^2`
/// let f = |x: &[F1]| x[0] * x[1] * x[1];
///
/// // Differentiate `f` at `(1,2)`.
/// let g = grad(f, &vec![1.0, 2.0]);
/// println!("({}, {})", g[0], g[1]); // prints `(4, 4)`
/// # assert_eq!(g, vec![4.0, 4.0]);
pub fn grad<G>(f: G, x0: &[f64]) -> Vec<f64>
where
    G: Fn(&[F1]) -> F1,
{
    let mut nums: Vec<F1> = x0.iter().map(|&x| F1::cst(x)).collect();

    let mut results = Vec::new();

    for i in 0..nums.len() {
        nums[i] = F1::var(nums[i]);
        results.push(f(&nums).deriv());
        nums[i] = F1::cst(nums[i]);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_arithmetic_test() {
        // Test basic arithmetic on F.
        let mut x = F1::var(1.0);
        let y = F1::var(2.0);

        assert_eq!(-x, F { x: -1.0, dx: -1.0 }); // negation

        assert_eq!(x + y, F { x: 3.0, dx: 2.0 }); // addition
        assert_eq!(x + 2.0, F { x: 3.0, dx: 1.0 }); // addition
        assert_eq!(2.0 + x, F { x: 3.0, dx: 1.0 }); // addition
        x += y;
        assert_eq!(x, F { x: 3.0, dx: 2.0 }); // assign add
        x += 1.0;
        assert_eq!(x, F { x: 4.0, dx: 2.0 }); // assign add

        assert_eq!(x - y, F { x: 2.0, dx: 1.0 }); // subtraction
        assert_eq!(x - 1.0, F { x: 3.0, dx: 2.0 }); // subtraction
        assert_eq!(1.0 - x, F { x: -3.0, dx: -2.0 }); // subtraction
        x -= y;
        assert_eq!(x, F { x: 2.0, dx: 1.0 }); // subtract assign
        x -= 1.0;
        assert_eq!(x, F { x: 1.0, dx: 1.0 }); // subtract assign

        assert_eq!(x * y, F { x: 2.0, dx: 3.0 }); // multiplication
        assert_eq!(x * 2.0, F { x: 2.0, dx: 2.0 }); // multiplication
        assert_eq!(2.0 * x, F { x: 2.0, dx: 2.0 }); // multiplication
        x *= y;
        assert_eq!(x, F { x: 2.0, dx: 3.0 }); // multiply assign
        x *= 2.0;
        assert_eq!(x, F { x: 4.0, dx: 6.0 }); // multiply assign

        assert_eq!(x / y, F { x: 2.0, dx: 2.0 }); // division
        assert_eq!(x / 2.0, F { x: 2.0, dx: 3.0 }); // division
        assert_eq!(2.0 / x, F { x: 0.5, dx: -0.75 }); // division
        x /= y;
        assert_eq!(x, F { x: 2.0, dx: 2.0 }); // divide assign
        x /= 2.0;
        assert_eq!(x, F { x: 1.0, dx: 1.0 }); // divide assign

        assert_eq!(x % y, F { x: 1.0, dx: 1.0 }); // mod
        assert_eq!(x % 2.0, F { x: 1.0, dx: 1.0 }); // mod
        assert_eq!(2.0 % x, F { x: 0.0, dx: -2.0 }); // mod
        x %= y;
        assert_eq!(x, F { x: 1.0, dx: 1.0 }); // mod assign
        x %= 2.0;
        assert_eq!(x, F { x: 1.0, dx: 1.0 }); // mod assign
    }

    // Test the min and max functions
    #[test]
    fn min_max_test() {
        // Test basic arithmetic on F.
        let a = F1::var(1.0);
        let mut b = F::cst(2.0);

        b = b.min(a);
        assert_eq!(b, F { x: 1.0, dx: 1.0 });

        b = F::cst(2.0);
        b = a.min(b);
        assert_eq!(b, F { x: 1.0, dx: 1.0 });

        let b = F::cst(2.0);

        let c = a.max(b);
        assert_eq!(c, F { x: 2.0, dx: 0.0 });

        // Make sure that our min and max are consistent with the internal implementation to avoid
        // inconsistencies in the future. In particular we look at tie breaking.

        let b = F::cst(1.0);
        let minf = a.x.min(b.x);
        assert_eq!(
            a.min(b),
            F {
                x: minf,
                dx: if minf == a.x { a.dx } else { b.dx }
            }
        );

        let maxf = a.x.max(b.x);
        assert_eq!(
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
        let ad_v = vec![F1::var(1.0), F1::var(2.0)].into_iter();
        assert_eq!(ad_v.clone().sum::<F1>(), F { x: 3.0, dx: 2.0 });
        assert_eq!(v.sum::<F1>(), F { x: 3.0, dx: 0.0 });
    }

    // Test the different ways to compute a derivative of a quadratic.
    #[test]
    fn quadratic() {
        let f1 = |x: F1| (x - 1.0f64).pow(2.0);
        let f2 = |x: F1| (x - 1.0f64) * (x - 1.0f64);

        // Derivative at 0
        let dfdx1: F1 = f1(F1::var(0.0));
        let dfdx2: F1 = f2(F1::var(0.0));

        assert_eq!(dfdx1, dfdx2);

        let f1 = |x: F1| x.pow(2.0);
        let f2 = |x: F1| x * x;

        // Derivative at 0
        let dfdx1: F1 = f1(F1::var(0.0));
        let dfdx2: F1 = f2(F1::var(0.0));

        assert_eq!(dfdx1, dfdx2);
    }
}
