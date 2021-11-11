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

//! This module defines forward automatic differentiation.
//! This mode of `autodiff` is most efficient when computing derivatives with more inputs than outputs.
//! It is also useful for computing Jacobian products (see [crate root docs](crate::lib) for examples).

use num_traits::{
    Bounded, Float, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::{f64, fmt};

/// A generic forward differentiation `Dual` number.
///
/// The derivative is generic in `V` to support composition and alternative
/// numeric types and in `D` to support higher order differentiation.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct F<V, D> {
    /// The value of the variable.
    pub x: V,
    /// The derivative of the variable.
    pub dx: D,
}

impl<V: fmt::Debug, D: fmt::Debug> fmt::Debug for F<V, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}V, {:?}D)", self.x, self.dx)
    }
}

impl<V: fmt::Display, D: fmt::Display> fmt::Display for F<V, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}V, {}D)", self.x, self.dx)
    }
}

/// First order dual number.
pub type FT<T> = F<T, T>;

/// First order dual number with `f64`.
///
/// This type is mainly kept for compatibility with autodiff v0.2 and v0.3.
pub type F1 = FT<f64>;

/*
 * IMPORTANT: The reason why PartialEq and PartialOrd need to be implemented in terms of x only
 * (i.e. not compare dx also) is because F is meant to have the same behaviour as a regular float.
 * That is if a and b are floats, then we want `a op b` to be true if and only if `x op y` where
 * x.value() = a and y.value() = b and `op` is one of <, >, <=, >=, or =.
 *
 * Changing this behaviour will change the behaviour of algorithms that expect floats as their
 * parameters, which is unacceptable since it precludes F from being used as a drop-in substitution
 * for floats.
 */

impl<V: PartialEq, D, U> PartialEq<F<V, U>> for F<V, D> {
    #[inline]
    fn eq(&self, rhs: &F<V, U>) -> bool {
        self.x == rhs.x
    }
}

impl<V: PartialOrd, D, U> PartialOrd<F<V, U>> for F<V, D> {
    #[inline]
    fn partial_cmp(&self, other: &F<V, U>) -> Option<::std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.x, &other.x)
    }
}

/// Compare the values and derivatives of two dual numbers for equality.
trait DualEq {
    fn dual_eq(&self, rhs: &Self) -> bool;
}

impl DualEq for f32 {
    /// Compare two single precision floats for equality.
    #[inline]
    fn dual_eq(&self, rhs: &f32) -> bool {
        self == rhs
    }
}

impl DualEq for f64 {
    /// Compare two double precision floats for equality.
    #[inline]
    fn dual_eq(&self, rhs: &f64) -> bool {
        self == rhs
    }
}

impl<V: PartialEq, D: DualEq> DualEq for F<V, D> {
    /// Compare two `F`s in full, including the derivative part.
    #[inline]
    fn dual_eq(&self, rhs: &F<V, D>) -> bool {
        self.x == rhs.x && self.dx.dual_eq(&rhs.dx)
    }
}

impl<V: ToPrimitive, D> Into<f64> for F<V, D> {
    /// Converts the dual number into an `f64`.
    ///
    /// # Panics
    /// This function panics if this conversion fails.
    #[inline]
    fn into(self) -> f64 {
        self.x.to_f64().unwrap()
    }
}

impl<V: ToPrimitive, D> Into<f32> for F<V, D> {
    /// Converts the dual number into an `f32`.
    ///
    /// # Panics
    /// This function panics if this conversion fails.
    #[inline]
    fn into(self) -> f32 {
        self.x.to_f32().unwrap()
    }
}

impl<V: Neg<Output = V>, D: Neg<Output = D>> Neg for F<V, D> {
    type Output = F<V, D>;
    #[inline]
    fn neg(self) -> F<V, D> {
        F {
            x: -self.x,
            dx: -self.dx,
        }
    }
}

impl<V: Add, D: Add<D, Output = D>> Add<F<V, D>> for F<V, D> {
    type Output = F<V::Output, D>;
    #[inline]
    fn add(self, rhs: F<V, D>) -> F<V::Output, D> {
        F {
            x: self.x + rhs.x,
            dx: self.dx + rhs.dx,
        }
    }
}

impl<V: Add<Output = V>, D> Add<V> for F<V, D> {
    type Output = F<V, D>;
    #[inline]
    fn add(self, rhs: V) -> F<V, D> {
        F {
            x: rhs + self.x,
            dx: self.dx,
        }
    }
}

impl Add<FT<f64>> for f64 {
    type Output = FT<f64>;
    #[inline]
    fn add(self, rhs: FT<f64>) -> FT<f64> {
        F {
            x: self + rhs.x,
            dx: rhs.dx,
        }
    }
}

impl Add<FT<f64>> for f32 {
    type Output = FT<f64>;
    #[inline]
    fn add(self, rhs: FT<f64>) -> FT<f64> {
        self as f64 + rhs
    }
}

impl<V: AddAssign, D: AddAssign> AddAssign for F<V, D> {
    #[inline]
    fn add_assign(&mut self, rhs: F<V, D>) {
        self.x += rhs.x;
        self.dx += rhs.dx;
    }
}

impl<V: AddAssign, D> AddAssign<V> for F<V, D> {
    #[inline]
    fn add_assign(&mut self, rhs: V) {
        self.x += rhs;
    }
}

impl<V: Sub<Output = V>, D: Sub<Output = D>> Sub<F<V, D>> for F<V, D> {
    type Output = F<V, D>;
    #[inline]
    fn sub(self, rhs: F<V, D>) -> F<V, D> {
        F {
            x: self.x - rhs.x,
            dx: self.dx - rhs.dx,
        }
    }
}

impl<V: Sub, D> Sub<V> for F<V, D> {
    type Output = F<V::Output, D>;
    #[inline]
    fn sub(self, rhs: V) -> F<V::Output, D> {
        F {
            x: self.x - rhs,
            dx: self.dx,
        }
    }
}

impl Sub<FT<f64>> for f64 {
    type Output = FT<f64>;
    #[inline]
    fn sub(self, rhs: FT<f64>) -> FT<f64> {
        F {
            x: self - rhs.x,
            dx: -rhs.dx,
        }
    }
}

impl Sub<FT<f64>> for f32 {
    type Output = FT<f64>;
    #[inline]
    fn sub(self, rhs: FT<f64>) -> FT<f64> {
        self as f64 - rhs
    }
}

impl<V: SubAssign, D: SubAssign> SubAssign for F<V, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: F<V, D>) {
        self.x -= rhs.x;
        self.dx -= rhs.dx;
    }
}

impl<V: SubAssign, D> SubAssign<V> for F<V, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: V) {
        self.x -= rhs;
    }
}

/*
 * Multiplication
 */

impl<V, D> Mul<F<V, D>> for F<V, D>
where
    V: Clone + Mul,
    //D: Copy + Mul<R>,
    D: Clone + Mul<V>,
    D::Output: Add,
    //F<V,D>: ReduceOrder<Output = R>
{
    type Output = F<V::Output, <<D as Mul<V>>::Output as Add>::Output>;
    #[inline]
    fn mul(self, rhs: F<V, D>) -> Self::Output {
        F {
            x: self.x.clone() * rhs.x.clone(),
            dx: self.dx * rhs.x + rhs.dx * self.x,
            //dx: self.dx * rhs.reduce_order() + rhs.dx * self.reduce_order(),
        }
    }
}

// Multiply by double precision floats (treated as constants)

impl Mul<FT<f64>> for f64 {
    type Output = FT<f64>;
    #[inline]
    fn mul(self, rhs: FT<f64>) -> FT<f64> {
        // self is treated as a constant
        F {
            x: self * rhs.x,
            dx: self * rhs.dx,
        }
    }
}

impl<V: Mul<Output = V> + Clone, D: Mul<V, Output = D>> Mul<V> for F<V, D> {
    type Output = F<V, D>;
    #[inline]
    fn mul(self, rhs: V) -> F<V, D> {
        // rhs is treated as a constant
        F {
            x: self.x * rhs.clone(),
            dx: self.dx * rhs,
        }
    }
}

// Multiply by single precision floats

impl Mul<FT<f64>> for f32 {
    type Output = FT<f64>;
    #[inline]
    fn mul(self, rhs: FT<f64>) -> FT<f64> {
        self as f64 * rhs
    }
}

// Multiply assign operators

impl<V, D> MulAssign for F<V, D>
where
    V: Clone + MulAssign,
    //D: MulAssign<R> + Mul<R, Output = D> + AddAssign,
    D: MulAssign<V> + Mul<V, Output = D> + AddAssign,
    //F<V, D>: ReduceOrder<Reduced = R>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: F<V, D>) {
        // Product rule
        //self.dx *= rhs.reduce_order();
        //self.dx += rhs.dx * self.reduce_order();
        self.dx *= rhs.x.clone();
        self.dx += rhs.dx * self.x.clone();
        self.x *= rhs.x;
    }
}

impl<V: MulAssign + Clone, D: MulAssign<V>> MulAssign<V> for F<V, D> {
    #[inline]
    fn mul_assign(&mut self, rhs: V) {
        // rhs is treated as a constant
        self.x *= rhs.clone();
        self.dx *= rhs;
    }
}

// MulAssign<F> for f64 is not implemented deliberately, because this operation erases the
// tracking of derivative information.

/*
 * Division
 */

impl<V, D> Div<F<V, D>> for F<V, D>
where
    V: Clone + Div<Output = V> + Mul<Output = V>,
    //D: Copy + Div<R, Output = D> + Mul<R, Output = D> + Sub<Output = D>,
    D: Clone + Div<V, Output = D> + Mul<V, Output = D> + Sub<Output = D>,
    //F<V, D>: ReduceOrder<Reduced = R>,
    //R: Copy + Mul<Output = R>,
{
    type Output = F<V, D>;
    #[inline]
    fn div(self, rhs: F<V, D>) -> F<V, D> {
        //let rhs_r = rhs.reduce_order();
        F {
            x: self.x.clone() / rhs.x.clone(),
            dx: (self.dx * rhs.x.clone() - rhs.dx * self.x) / (rhs.x.clone() * rhs.x),
            //dx: (self.dx * rhs_r - rhs.dx * self.reduce_order()) / (rhs_r * rhs_r),
        }
    }
}

impl<V: Div + Clone, D: Div<V>> Div<V> for F<V, D> {
    type Output = F<V::Output, D::Output>;
    #[inline]
    fn div(self, rhs: V) -> Self::Output {
        F {
            x: self.x / rhs.clone(),
            dx: self.dx / rhs,
        }
    }
}

impl Div<FT<f64>> for f64 {
    type Output = FT<f64>;
    #[inline]
    fn div(self, rhs: FT<f64>) -> FT<f64> {
        F {
            x: self / rhs.x,
            dx: -self * rhs.dx / (rhs.x * rhs.x),
        }
    }
}

impl Div<FT<f64>> for f32 {
    type Output = FT<f64>;
    #[inline]
    fn div(self, rhs: FT<f64>) -> FT<f64> {
        self as f64 / rhs
    }
}

impl<V, D> DivAssign for F<V, D>
where
    V: Clone + DivAssign + Mul<Output = V> + Div<Output = V>,
    //D: Mul<R, Output = D> + DivAssign<R> + SubAssign,
    D: Mul<V, Output = D> + DivAssign<V> + SubAssign,
    //F<V, D>: ReduceOrder<Reduced = R>,
    //R: Copy + Mul<Output = R> + Div<R, Output = R>,
{
    #[inline]
    fn div_assign(&mut self, rhs: F<V, D>) {
        //let rhs_r = rhs.reduce_order();
        //self.dx /= rhs_r;
        self.dx /= rhs.x.clone();
        //self.dx -= rhs.dx * (self.reduce_order() / (rhs_r * rhs_r));
        self.dx -= rhs.dx * (self.x.clone() / (rhs.x.clone() * rhs.x.clone()));
        self.x /= rhs.x;
    }
}

impl<V: DivAssign + Clone, D: DivAssign<V>> DivAssign<V> for F<V, D> {
    #[inline]
    fn div_assign(&mut self, rhs: V) {
        self.x /= rhs.clone();
        self.dx /= rhs;
    }
}

// DivAssign<F> for f64 is not implemented deliberately, because this operation erases the
// tracking of the derivative information.

/*
 * Remainder function
 */

impl<V, D> Rem<F<V, D>> for F<V, D>
where
    V: Clone + Rem<Output = V> + Div<Output = V> + Sub<Output = V> + One,
    //D: Copy + Mul<R, Output = D> + Sub<Output = D>,
    D: Clone + Mul<V, Output = D> + Sub<Output = D>,
    //F<V, D>: ReduceOrder<Reduced = R>,
    //R: Float,
{
    type Output = F<<V as Rem>::Output, D>;
    #[inline]
    fn rem(self, rhs: F<V, D>) -> Self::Output {
        // This is an approximation. There are places where the derivative doesn't exist.
        let div = self.x.clone() / rhs.x.clone();
        F {
            x: self.x % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: self.dx - rhs.dx * (div.clone() - div % V::one()),
            //dx: self.dx - rhs.dx * (self.reduce_order() / rhs.reduce_order()).trunc(),
        }
    }
}

impl<V: Rem<f64>, D: Rem<f64>> Rem<f64> for F<V, D> {
    type Output = F<V::Output, D::Output>;
    #[inline]
    fn rem(self, rhs: f64) -> Self::Output {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self.x % rhs, // x % y = x - [x/|y|]*|y|
            dx: self.dx % rhs,
        }
    }
}

impl Rem<FT<f64>> for f64 {
    type Output = FT<f64>;
    #[inline]
    fn rem(self, rhs: FT<f64>) -> FT<f64> {
        // This is an approximation. There are places where the derivative doesn't exist.
        F {
            x: self % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: -(self / rhs.x).trunc() * rhs.dx,
        }
    }
}

impl<V, D> RemAssign for F<V, D>
where
    V: Clone + RemAssign + Div<Output = V> + Sub<Output = V> + Rem<Output = V> + One,
    //D: Copy + Mul<R, Output = D> + SubAssign,
    D: Mul<V, Output = D> + SubAssign,
    //F<V, D>: ReduceOrder<Reduced = R>,
    //R: Float,
{
    #[inline]
    fn rem_assign(&mut self, rhs: F<V, D>) {
        // x % y = x - [x/|y|]*|y|
        //self.dx -= rhs.dx * (self.reduce_order() / rhs.reduce_order()).trunc();
        let div = self.x.clone() / rhs.x.clone();
        self.dx -= rhs.dx * (div.clone() - div % V::one());
        self.x %= rhs.x;
    }
}

impl<V: RemAssign<f64>, D: RemAssign<f64>> RemAssign<f64> for F<V, D> {
    #[inline]
    fn rem_assign(&mut self, rhs: f64) {
        self.x %= rhs;
        self.dx %= rhs;
    }
}

impl<V: RemAssign<f32>, D: RemAssign<f32>> RemAssign<f32> for F<V, D> {
    #[inline]
    fn rem_assign(&mut self, rhs: f32) {
        self.x %= rhs;
        self.dx %= rhs;
    }
}

impl<V: Default, D: Default> Default for F<V, D> {
    #[inline]
    fn default() -> Self {
        F {
            x: V::default(),
            dx: D::default(),
        }
    }
}

impl<V: ToPrimitive, D> ToPrimitive for F<V, D> {
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

impl<V: NumCast, D: Zero> NumCast for F<V, D> {
    fn from<T: ToPrimitive>(n: T) -> Option<F<V, D>> {
        match V::from(n) {
            Some(x) => Some(F { x, dx: D::zero() }),
            None => None,
        }
    }
}

impl<V: FromPrimitive, D: Zero> FromPrimitive for F<V, D> {
    #[inline]
    fn from_isize(n: isize) -> Option<Self> {
        V::from_isize(n).map(F::cst)
    }
    #[inline]
    fn from_i8(n: i8) -> Option<Self> {
        V::from_i8(n).map(F::cst)
    }
    #[inline]
    fn from_i16(n: i16) -> Option<Self> {
        V::from_i16(n).map(F::cst)
    }
    #[inline]
    fn from_i32(n: i32) -> Option<Self> {
        V::from_i32(n).map(F::cst)
    }
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        V::from_i64(n).map(F::cst)
    }
    #[inline]
    fn from_i128(n: i128) -> Option<Self> {
        V::from_i128(n).map(F::cst)
    }
    #[inline]
    fn from_usize(n: usize) -> Option<Self> {
        V::from_usize(n).map(F::cst)
    }
    #[inline]
    fn from_u8(n: u8) -> Option<Self> {
        V::from_u8(n).map(F::cst)
    }
    #[inline]
    fn from_u16(n: u16) -> Option<Self> {
        V::from_u16(n).map(F::cst)
    }
    #[inline]
    fn from_u32(n: u32) -> Option<Self> {
        V::from_u32(n).map(F::cst)
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        V::from_u64(n).map(F::cst)
    }
    #[inline]
    fn from_u128(n: u128) -> Option<Self> {
        V::from_u128(n).map(F::cst)
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        V::from_f32(n).map(F::cst)
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        V::from_f64(n).map(F::cst)
    }
}

impl<V: Zero, D: Zero> Zero for F<V, D> {
    #[inline]
    fn zero() -> F<V, D> {
        F {
            x: V::zero(),
            dx: D::zero(),
        }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.x.is_zero()
    }
}

impl<V, D> One for F<V, D>
where
    V: Clone + One,
    //D: Copy + Zero + std::fmt::Debug + Mul<<F<V, D> as ReduceOrder>::Reduced, Output = D>,
    D: Clone + Zero + std::fmt::Debug + Mul<V, Output = D>,
    //F<V, D>: ReduceOrder,
{
    #[inline]
    fn one() -> F<V, D> {
        F {
            x: V::one(),
            dx: D::zero(),
        }
    }
}

impl<V, D> Num for F<V, D>
where
    V: Clone + Num,
    D: Clone
        + std::fmt::Debug
        + Zero
        + PartialEq
        + Sub<Output = D>
        + Mul<V, Output = D>
        //+ Mul<R, Output = D>
        + Mul<Output = D>
        + Div<V, Output = D>
        //+ Div<R, Output = D>,
        + Div<Output = D>,
    //F<V, D>: ReduceOrder<Reduced = R>,
    //R: Float,
{
    type FromStrRadixErr = V::FromStrRadixErr;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        V::from_str_radix(src, radix).map(F::cst)
    }
}

impl<V, D> Signed for F<V, D>
where
    V: Signed + Clone + Num + PartialOrd,
    D: Signed
        + Clone
        + std::fmt::Debug
        + Zero
        + PartialEq
        + Sub<Output = D>
        + Mul<V, Output = D>
        + Mul<Output = D>
        + Div<V, Output = D>
        + Div<Output = D>,
{
    #[inline]
    fn abs(&self) -> Self {
        if self.is_negative() {
            -self.clone()
        } else {
            self.clone()
        }
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other {
            Self::zero()
        } else {
            self.clone() - other.clone()
        }
    }

    #[inline]
    fn signum(&self) -> Self {
        match self {
            n if n > &Self::zero() => Self::one(),
            n if n == &Self::zero() => Self::zero(),
            _ => -Self::one(),
        }
    }

    #[inline]
    fn is_positive(&self) -> bool {
        self > &Self::zero()
    }

    #[inline]
    fn is_negative(&self) -> bool {
        self < &Self::zero()
    }
}

impl<V, D> Bounded for F<V, D>
where
    V: Bounded,
    D: Bounded,
{
    fn min_value() -> Self {
        F {
            x: V::min_value(),
            dx: D::min_value(),
        }
    }
    fn max_value() -> Self {
        F {
            x: V::max_value(),
            dx: D::max_value(),
        }
    }
}

impl<V: FloatConst, D: Zero> FloatConst for F<V, D> {
    #[inline]
    fn E() -> F<V, D> {
        F::cst(V::E())
    }
    #[inline]
    fn FRAC_1_PI() -> F<V, D> {
        F::cst(V::FRAC_1_PI())
    }
    #[inline]
    fn FRAC_1_SQRT_2() -> F<V, D> {
        F::cst(V::FRAC_1_SQRT_2())
    }
    #[inline]
    fn FRAC_2_PI() -> F<V, D> {
        F::cst(V::FRAC_2_PI())
    }
    #[inline]
    fn FRAC_2_SQRT_PI() -> F<V, D> {
        F::cst(V::FRAC_2_SQRT_PI())
    }
    #[inline]
    fn FRAC_PI_2() -> F<V, D> {
        F::cst(V::FRAC_PI_2())
    }
    #[inline]
    fn FRAC_PI_3() -> F<V, D> {
        F::cst(V::FRAC_PI_3())
    }
    #[inline]
    fn FRAC_PI_4() -> F<V, D> {
        F::cst(V::FRAC_PI_4())
    }
    #[inline]
    fn FRAC_PI_6() -> F<V, D> {
        F::cst(V::FRAC_PI_6())
    }
    #[inline]
    fn FRAC_PI_8() -> F<V, D> {
        F::cst(V::FRAC_PI_8())
    }
    #[inline]
    fn LN_10() -> F<V, D> {
        F::cst(V::LN_10())
    }
    #[inline]
    fn LN_2() -> F<V, D> {
        F::cst(V::LN_2())
    }
    #[inline]
    fn LOG10_E() -> F<V, D> {
        F::cst(V::LOG10_E())
    }
    #[inline]
    fn LOG2_E() -> F<V, D> {
        F::cst(V::LOG2_E())
    }
    #[inline]
    fn PI() -> F<V, D> {
        F::cst(V::PI())
    }
    #[inline]
    fn SQRT_2() -> F<V, D> {
        F::cst(V::SQRT_2())
    }
}

impl<V, D> Float for F<V, D>
where
    V: Float,
    D: std::fmt::Debug
        + Float
        + Zero
        + Neg<Output = D>
        //+ Mul<R, Output = D>
        + Mul<Output = D>
        + Mul<V, Output = D>
        + Add<Output = D>
        + Div<V, Output = D>
        //+ Div<R, Output = D>
        + Div<Output = D>
        + Sub<Output = D>
        + Clone
        + PartialOrd,
    //F<V, D>: ReduceOrder<Reduced = R, ReducedValue = V>,
    //R: Float + Mul<f64, Output = R>,
{
    #[inline]
    fn nan() -> F<V, D> {
        F::cst(V::nan())
    }
    #[inline]
    fn infinity() -> F<V, D> {
        F::cst(V::infinity())
    }
    #[inline]
    fn neg_infinity() -> F<V, D> {
        F::cst(V::neg_infinity())
    }
    #[inline]
    fn neg_zero() -> F<V, D> {
        F::cst(V::neg_zero())
    }
    #[inline]
    fn min_value() -> F<V, D> {
        F::cst(V::min_value())
    }
    #[inline]
    fn min_positive_value() -> F<V, D> {
        F::cst(V::min_positive_value())
    }
    #[inline]
    fn max_value() -> F<V, D> {
        F::cst(V::max_value())
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
    fn floor(self) -> F<V, D> {
        F {
            x: self.x.floor(),
            dx: self.dx,
        }
    }
    #[inline]
    fn ceil(self) -> F<V, D> {
        F {
            x: self.x.ceil(),
            dx: self.dx,
        }
    }
    #[inline]
    fn round(self) -> F<V, D> {
        F {
            x: self.x.round(),
            dx: self.dx,
        }
    }
    #[inline]
    fn trunc(self) -> F<V, D> {
        F {
            x: self.x.trunc(),
            dx: self.dx,
        }
    }
    #[inline]
    fn fract(self) -> F<V, D> {
        F {
            x: self.x.fract(),
            dx: self.dx,
        }
    }
    #[inline]
    fn abs(self) -> F<V, D> {
        F {
            x: self.x.abs(),
            dx: if self.x >= V::zero() {
                self.dx
            } else {
                -self.dx
            },
        }
    }
    #[inline]
    fn signum(self) -> F<V, D> {
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
    fn mul_add(self, a: F<V, D>, b: F<V, D>) -> F<V, D> {
        F {
            x: self.x.mul_add(a.x, b.x),
            dx: self.dx * a.x + a.dx * self.x + b.dx,
        }
    }
    #[inline]
    fn recip(self) -> F<V, D> {
        //let x = self.reduce_order();
        let x = self.x;
        F {
            x: self.x.recip(),
            dx: -self.dx / (x * x),
        }
    }
    #[inline]
    fn powi(self, n: i32) -> F<V, D> {
        F {
            x: self.x.powi(n),
            //dx: self.dx * (self.reduce_order().powi(n - 1) * n as f64),
            dx: self.dx * (self.x.powi(n - 1) * V::from(n).unwrap()),
        }
    }
    #[inline]
    fn powf(self, n: F<V, D>) -> F<V, D> {
        //let self_r = self.reduce_order();
        //let n_r = n.reduce_order();
        let self_r = self.x;
        let n_r = n.x;
        // Avoid imaginary values in the ln.
        let dn = if n.dx.is_zero() {
            D::zero()
        } else {
            n.dx * Float::ln(self_r)
        };

        let x = Float::powf(self_r, n_r);

        // Avoid division by zero.
        let x_df = if self.x.is_zero() && x.is_zero() {
            D::zero()
        } else {
            self.dx * (x * n_r / self_r)
        };

        F {
            //x: *Self::reduced_value(&x),
            x,
            dx: dn * x + x_df,
        }
    }
    #[inline]
    fn sqrt(self) -> F<V, D> {
        F {
            x: self.x.sqrt(),
            dx: {
                //let denom = self.reduce_order().sqrt() * 2.0;
                let denom = self.x.sqrt() * V::from(2.0).unwrap();
                //if denom == R::zero() && self.dx == D::zero() {
                if denom == V::zero() && self.dx == D::zero() {
                    D::zero()
                } else {
                    self.dx / denom
                }
            },
        }
    }

    #[inline]
    fn exp(self) -> F<V, D> {
        let exp = Float::exp(self.x);
        F {
            x: exp,
            //dx: self.dx * Float::exp(self.reduce_order()),
            dx: self.dx * exp,
        }
    }
    #[inline]
    fn exp2(self) -> F<V, D> {
        let exp2 = Float::exp2(self.x);
        F {
            x: exp2,
            //dx: self.dx * Float::ln(2.0) * Float::exp2(self.reduce_order()),
            dx: self.dx * V::from(2.0).unwrap().ln() * exp2,
        }
    }
    #[inline]
    fn ln(self) -> F<V, D> {
        F {
            x: Float::ln(self.x),
            //dx: self.dx * self.reduce_order().recip(),
            dx: self.dx * self.x.recip(),
        }
    }
    #[inline]
    fn log(self, b: F<V, D>) -> F<V, D> {
        //let s_r = self.reduce_order();
        //let b_r = b.reduce_order();
        let s_r = self.x;
        let b_r = b.x;
        let ln_b_r = Float::ln(b_r);
        F {
            x: Float::log(self.x, b.x),
            dx: -b.dx * Float::ln(s_r) / (b_r * ln_b_r * ln_b_r) + self.dx / (s_r * ln_b_r),
        }
    }
    #[inline]
    fn log2(self) -> F<V, D> {
        Float::log(self, F::cst(V::from(2.0).unwrap()))
    }
    #[inline]
    fn log10(self) -> F<V, D> {
        Float::log(self, F::cst(V::from(10.0).unwrap()))
    }
    #[inline]
    fn max(self, other: F<V, D>) -> F<V, D> {
        if self.x < other.x {
            other
        } else {
            self
        }
    }
    #[inline]
    fn min(self, other: F<V, D>) -> F<V, D> {
        if self.x > other.x {
            other
        } else {
            self
        }
    }
    #[inline]
    fn abs_sub(self, other: F<V, D>) -> F<V, D> {
        if self > other {
            F {
                x: Float::abs_sub(self.x, other.x),
                dx: (self - other).dx,
            }
        } else {
            F::cst(V::zero())
        }
    }
    #[inline]
    fn cbrt(self) -> F<V, D> {
        let x_cbrt = Float::cbrt(self.x);
        F {
            x: x_cbrt,
            //dx: self.dx * self.reduce_order().powf(R::from(-2.0 / 3.0).unwrap()) * 1.0 / 3.0,
            dx: {
                let denom = x_cbrt * x_cbrt * V::from(3.0).unwrap();
                if denom == V::zero() && self.dx == D::zero() {
                    D::zero()
                } else {
                    self.dx / denom
                }
            },
        }
    }
    #[inline]
    fn hypot(self, other: F<V, D>) -> F<V, D> {
        Float::sqrt(self.clone() * self + other.clone() * other)
    }
    #[inline]
    fn sin(self) -> F<V, D> {
        F {
            x: Float::sin(self.x),
            //dx: self.dx * Float::cos(self.reduce_order()),
            dx: self.dx * Float::cos(self.x),
        }
    }
    #[inline]
    fn cos(self) -> F<V, D> {
        F {
            x: Float::cos(self.x),
            //dx: -self.dx * Float::sin(self.reduce_order()),
            dx: -self.dx * Float::sin(self.x),
        }
    }
    #[inline]
    fn tan(self) -> F<V, D> {
        //let t = Float::tan(self.reduce_order());
        let t = Float::tan(self.x);
        F {
            //x: *Self::reduced_value(&t),
            x: t,
            //dx: self.dx * (t * t + R::one()),
            dx: self.dx * (t * t + V::one()),
        }
    }
    #[inline]
    fn asin(self) -> F<V, D> {
        F {
            x: Float::asin(self.x.clone()),
            //k;wdx: self.dx / Float::sqrt(R::one() - Float::powi(self.reduce_order(), 2)),
            dx: self.dx / Float::sqrt(V::one() - self.x.clone() * self.x),
        }
    }
    #[inline]
    fn acos(self) -> F<V, D> {
        F {
            x: Float::acos(self.x),
            //dx: -self.dx / Float::sqrt(R::one() - Float::powi(self.reduce_order(), 2)),
            dx: -self.dx / Float::sqrt(V::one() - self.x * self.x),
        }
    }
    #[inline]
    fn atan(self) -> F<V, D> {
        F {
            x: Float::atan(self.x),
            // dx: self.dx / (Float::powi(self.reduce_order(), 2) + R::one()),
            dx: self.dx / (self.x * self.x + V::one()),
        }
    }
    #[inline]
    fn atan2(self, other: F<V, D>) -> F<V, D> {
        self.atan2_impl(other)
    }
    #[inline]
    fn sin_cos(self) -> (F<V, D>, F<V, D>) {
        //let (s, c) = Float::sin_cos(self.reduce_order());
        let (s, c) = Float::sin_cos(self.x);
        let sn = F {
            //x: *Self::reduced_value(&s),
            x: s,
            dx: self.dx * c,
        };
        let cn = F {
            //x: *Self::reduced_value(&c),
            x: c,
            dx: self.dx * (-s),
        };
        (sn, cn)
    }
    #[inline]
    fn exp_m1(self) -> F<V, D> {
        F {
            x: Float::exp_m1(self.x),
            //dx: self.dx * Float::exp(self.reduce_order()),
            dx: self.dx * Float::exp(self.x),
        }
    }
    #[inline]
    fn ln_1p(self) -> F<V, D> {
        F {
            x: Float::ln_1p(self.x),
            // dx: self.dx / (self.reduce_order() + R::one()),
            dx: self.dx / (self.x + V::one()),
        }
    }
    #[inline]
    fn sinh(self) -> F<V, D> {
        F {
            x: Float::sinh(self.x),
            // dx: self.dx * Float::cosh(self.reduce_order()),
            dx: self.dx * Float::cosh(self.x),
        }
    }
    #[inline]
    fn cosh(self) -> F<V, D> {
        F {
            x: Float::cosh(self.x),
            //dx: self.dx * Float::sinh(self.reduce_order()),
            dx: self.dx * Float::sinh(self.x),
        }
    }
    #[inline]
    fn tanh(self) -> F<V, D> {
        let tanhx = Float::tanh(self.x);
        F {
            x: Float::tanh(self.x),
            //dx: self.dx * (R::one() - Float::powi(Float::tanh(self.reduce_order()), 2)),
            dx: self.dx * (V::one() - tanhx * tanhx),
        }
    }
    #[inline]
    fn asinh(self) -> F<V, D> {
        F {
            x: Float::asinh(self.x),
            // dx: self.dx / (Float::powi(self.reduce_order(), 2) + R::one()).sqrt(),
            dx: self.dx / (self.x * self.x + V::one()).sqrt(),
        }
    }
    #[inline]
    fn acosh(self) -> F<V, D> {
        F {
            x: Float::acosh(self.x),
            //dx: self.dx / (Float::powi(self.reduce_order(), 2) - R::one()).sqrt(),
            dx: self.dx / (self.x * self.x - V::one()).sqrt(),
        }
    }
    #[inline]
    fn atanh(self) -> F<V, D> {
        F {
            x: Float::atanh(self.x),
            //dx: self.dx / (-Float::powi(self.reduce_order(), 2) + R::one()),
            dx: self.dx / (-self.x * self.x + V::one()),
        }
    }
    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    #[inline]
    fn epsilon() -> F<V, D> {
        F::cst(V::epsilon())
    }
    #[inline]
    fn to_degrees(self) -> F<V, D> {
        F::cst(Float::to_degrees(self.x))
    }
    #[inline]
    fn to_radians(self) -> F<V, D> {
        F::cst(Float::to_radians(self.x))
    }
}

impl<V: AddAssign + Zero, D: AddAssign + Zero> std::iter::Sum for F<V, D> {
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

impl<V: AddAssign + Zero, D: AddAssign + Zero> std::iter::Sum<V> for F<V, D> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = V>,
    {
        iter.map(F::cst).sum()
    }
}

impl<V, D> F<V, D>
where
    V: Float,
    D: Mul<V, Output = D> + Sub<Output = D> + Div<V, Output = D>,
{
    #[inline]
    pub(crate) fn atan2_impl(self, other: F<V, D>) -> F<V, D> {
        //let self_r = self.reduce_order();
        //let other_r = other.reduce_order();
        let self_r = self.x;
        let other_r = other.x;
        F {
            x: Float::atan2(self.x, other.x),
            dx: (self.dx * other_r - other.dx * self_r) / (self_r * self_r + other_r * other_r),
        }
    }
}

impl<V, D> F<V, D> {
    /// Create a new dual number with value `x` and initial derivative `d`.
    ///
    /// This is equivalent to setting the fields of `F` directly.
    #[inline]
    pub fn new(x: V, dx: D) -> F<V, D> {
        F { x, dx }
    }
}

impl<V, D: Zero> F<V, D> {
    /// Create a new constant.
    ///
    /// Use this also to convert from a variable to a constant.
    #[inline]
    pub fn cst(x: impl Into<V>) -> F<V, D> {
        F {
            x: x.into(),
            dx: D::zero(),
        }
    }
}

impl<V, D: One> F<V, D> {
    /// Create a new variable.
    ///
    /// Use this also to convert from a constant to a variable.
    #[inline]
    pub fn var(x: impl Into<V>) -> F<V, D> {
        F::new(x.into(), D::one())
    }
}

impl<V: Clone, D> F<V, D> {
    /// Get the value of this variable.
    #[inline]
    pub fn value(&self) -> V {
        self.x.clone()
    }
}

impl<V, D: Clone> F<V, D> {
    /// Get the current derivative of this variable.
    ///
    /// This will be zero if this `F` is a constant.
    #[inline]
    pub fn deriv(&self) -> D {
        self.dx.clone()
    }
}

impl<V: Zero> From<V> for F<V, V> {
    fn from(x: V) -> Self {
        F::cst(x)
    }
}

impl<V, D> F<V, D>
where
    Self: Float,
{
    /// Raise this number to the `n`'th power.
    ///
    /// This is a generic version of `Float::powf`.
    #[inline]
    pub fn pow(self, n: impl Into<F<V, D>>) -> F<V, D> {
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
/// let f = |x: FT<f64>| (-x * x / F::cst(2.0)).exp();
///
/// // Differentiate `f` at zero.
/// println!("{}", diff(f, 0.0)); // prints `0`
/// # assert_eq!(diff(f, 0.0), 0.0);
/// ```
pub fn diff<G>(f: G, x0: f64) -> f64
where
    G: FnOnce(FT<f64>) -> FT<f64>,
{
    f(F1::var(x0)).deriv()
}

/// Evaluates the gradient of `f` at `x0`
///
/// Note that it is much more efficient to use Backward or Reverse-mode automatic
/// differentiation for computing gradients of scalar valued functions.
///
/// # Examples
///
/// ```rust
/// use autodiff::*;
/// // Define a multivariate function `f(x,y) = x*y^2`
/// let f = |x: &[FT<f64>]| x[0] * x[1] * x[1];
///
/// // Differentiate `f` at `(1,2)`.
/// let g = grad(f, &vec![1.0, 2.0]);
/// println!("({}, {})", g[0], g[1]); // prints `(4, 4)`
/// # assert_eq!(g, vec![4.0, 4.0]);
pub fn grad<G>(f: G, x0: &[f64]) -> Vec<f64>
where
    G: Fn(&[FT<f64>]) -> FT<f64>,
{
    let mut nums: Vec<FT<f64>> = x0.iter().map(|&x| F1::cst(x)).collect();

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

    /// Convenience macro for comparing `F`s in full.
    macro_rules! assert_dual_eq {
        ($x:expr, $y:expr $(,)?) => {
            {
                let x = &$x;
                let y = &$y;
                assert!(F::dual_eq(x, y), "\nleft:  {:?}\nright: {:?}\n", x, y);
            }
        };
        ($x:expr, $y:expr, $($args:tt)+) => {
            assert!(F::dual_eq(&$x, &$y), $($args)+);
        };
    }

    #[test]
    fn basic_arithmetic_test() {
        // Test basic arithmetic on F.
        let mut x = F1::var(1.0);
        let y = F1::var(2.0);

        assert_dual_eq!(-x, F::new(-1.0, -1.0)); // negation

        assert_dual_eq!(x + y, F::new(3.0, 2.0)); // addition
        assert_dual_eq!(x + 2.0, F::new(3.0, 1.0)); // addition
        assert_dual_eq!(2.0 + x, F::new(3.0, 1.0)); // addition
        x += y;
        assert_dual_eq!(x, F::new(3.0, 2.0)); // assign add
        x += 1.0;
        assert_dual_eq!(x, F::new(4.0, 2.0)); // assign add

        assert_dual_eq!(x - y, F::new(2.0, 1.0)); // subtraction
        assert_dual_eq!(x - 1.0, F::new(3.0, 2.0)); // subtraction
        assert_dual_eq!(1.0 - x, F::new(-3.0, -2.)); // subtraction
        x -= y;
        assert_dual_eq!(x, F::new(2.0, 1.0)); // subtract assign
        x -= 1.0;
        assert_dual_eq!(x, F::new(1.0, 1.0)); // subtract assign

        assert_dual_eq!(x * y, F::new(2.0, 3.0)); // multiplication
        assert_dual_eq!(x * 2.0, F::new(2.0, 2.0)); // multiplication
        assert_dual_eq!(2.0 * x, F::new(2.0, 2.0)); // multiplication
        x *= y;
        assert_dual_eq!(x, F::new(2.0, 3.0)); // multiply assign
        x *= 2.0;
        assert_dual_eq!(x, F::new(4.0, 6.0)); // multiply assign

        assert_dual_eq!(x / y, F::new(2.0, 2.0)); // division
        assert_dual_eq!(x / 2.0, F::new(2.0, 3.0)); // division
        assert_dual_eq!(2.0 / x, F::new(0.5, -0.75)); // division
        x /= y;
        assert_dual_eq!(x, F::new(2.0, 2.0)); // divide assign
        x /= 2.0;
        assert_dual_eq!(x, F::new(1.0, 1.0)); // divide assign

        assert_dual_eq!(x % y, F::new(1.0, 1.0)); // mod
        assert_dual_eq!(x % 2.0, F::new(1.0, 1.0)); // mod
        assert_dual_eq!(2.0 % x, F::new(0.0, -2.0)); // mod
        x %= y;
        assert_dual_eq!(x, F::new(1.0, 1.0)); // mod assign
        x %= 2.0;
        assert_dual_eq!(x, F::new(1.0, 1.0)); // mod assign
    }

    // Test the min and max functions
    #[test]
    fn min_max_test() {
        // Test basic arithmetic on F.
        let a = F1::var(1.0);
        let mut b = F::cst(2.0);

        b = b.min(a);
        assert_dual_eq!(b, F::new(1.0, 1.0));

        b = F::cst(2.0);
        b = a.min(b);
        assert_dual_eq!(b, F::new(1.0, 1.0));

        let b = F::cst(2.0);

        let c = a.max(b);
        assert_dual_eq!(c, F::new(2.0, 0.0));

        // Make sure that our min and max are consistent with the internal implementation to avoid
        // inconsistencies in the future. In particular we look at tie breaking.

        let b = F::cst(1.0);
        let minf = a.x.min(b.x);
        assert_dual_eq!(
            a.min(b),
            F {
                x: minf,
                dx: if minf == a.x { a.dx } else { b.dx }
            }
        );

        let maxf = a.x.max(b.x);
        assert_dual_eq!(
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
        assert_dual_eq!(ad_v.clone().sum::<FT<f64>>(), F::new(3.0, 2.0));
        assert_dual_eq!(v.sum::<FT<f64>>(), F::new(3.0, 0.0));
    }

    // Test the different ways to compute a derivative of a quadratic.
    #[test]
    fn quadratic() {
        let f1 = |x: FT<f64>| (x - 1.0f64).pow(2.0);
        let f2 = |x: FT<f64>| (x - 1.0f64) * (x - 1.0f64);

        // Derivative at 0
        let dfdx1: FT<f64> = f1(F1::var(0.0));
        let dfdx2: FT<f64> = f2(F1::var(0.0));

        assert_dual_eq!(dfdx1, dfdx2);

        let f1 = |x: FT<f64>| x.pow(2.0);
        let f2 = |x: FT<f64>| x * x;

        // Derivative at 0
        let dfdx1: FT<f64> = f1(F1::var(0.0));
        let dfdx2: FT<f64> = f2(F1::var(0.0));

        assert_dual_eq!(dfdx1, dfdx2);
    }

    #[test]
    fn sqrt() {
        let x = F1::var(0.2).sqrt();
        assert_dual_eq!(x, F1::new(0.2.sqrt(), 0.5 / 0.2.sqrt()), "{:?}", x);

        // Test that taking a square root of zero does not produce NaN.
        // By convention we take 0/0 = 0 here.
        let x = F1::cst(0.0).sqrt();
        assert_dual_eq!(x, F1::new(0.0, 0.0), "{:?}", x);
    }

    #[test]
    fn cbrt() {
        let x = F1::var(0.2).cbrt();
        assert_dual_eq!(
            x,
            F1::new(0.2.cbrt(), 1.0 / (3.0 * 0.2.cbrt() * 0.2.cbrt())),
            "{:?}",
            x
        );

        // Test that taking a cube root of zero does not produce NaN.
        // By convention we take 0/0 = 0 here.
        let x = F1::cst(0.0).cbrt();
        assert_dual_eq!(x, F1::new(0.0, 0.0), "{:?}", x);
    }
}
