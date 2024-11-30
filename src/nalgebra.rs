use nalgebra::{convert, ComplexField, Field, RealField, SimdBool, SimdValue};
use num_traits::{FloatConst, NumCast, Zero};
use std::ops::{Div, Mul};

use crate::autofloat::float_impl::*;
use crate::{binary_op, unary_op, AutoFloat};

impl<T, const N: usize> RealField for AutoFloat<T, N>
where
    T: RealField + NumCast + FloatConst + Default + Copy,
{
    fn is_sign_positive(&self) -> bool {
        self.x.is_sign_positive()
    }
    fn is_sign_negative(&self) -> bool {
        self.x.is_sign_negative()
    }
    fn copysign(self, to: Self) -> Self {
        AutoFloat {
            x: RealField::copysign(self.x, to.x),
            dx: to.dx,
        }
    }
    fn max(self, other: Self) -> Self {
        max_impl!(self, other)
    }

    fn min(self, other: Self) -> Self {
        min_impl!(self, other)
    }

    fn clamp(self, low: Self, high: Self) -> Self {
        self.min(high).max(low)
    }

    fn atan2(self, other: Self) -> Self {
        atan2_impl!(self, other)
    }

    fn pi() -> Self {
        Self::PI()
    }
    fn two_pi() -> Self {
        Self::TAU()
    }
    fn frac_pi_2() -> Self {
        Self::FRAC_PI_2()
    }
    fn frac_pi_3() -> Self {
        Self::FRAC_PI_3()
    }
    fn frac_pi_4() -> Self {
        Self::FRAC_PI_4()
    }
    fn frac_pi_6() -> Self {
        Self::FRAC_PI_6()
    }
    fn frac_pi_8() -> Self {
        Self::FRAC_PI_8()
    }
    fn frac_1_pi() -> Self {
        Self::FRAC_1_PI()
    }
    fn frac_2_pi() -> Self {
        Self::FRAC_2_PI()
    }
    fn frac_2_sqrt_pi() -> Self {
        Self::FRAC_2_SQRT_PI()
    }
    fn e() -> Self {
        Self::E()
    }
    fn log2_e() -> Self {
        Self::LOG2_E()
    }
    fn log10_e() -> Self {
        Self::LOG10_E()
    }
    fn ln_2() -> Self {
        Self::LN_2()
    }
    fn ln_10() -> Self {
        Self::LN_10()
    }

    fn min_value() -> Option<Self> {
        <T as RealField>::min_value().map(AutoFloat::constant)
    }

    fn max_value() -> Option<Self> {
        <T as RealField>::max_value().map(AutoFloat::constant)
    }
}

impl<T, const N: usize> ComplexField for AutoFloat<T, N>
where
    T: ComplexField
        + Mul<T::RealField, Output = T>
        + Div<T::RealField, Output = T>
        + Copy
        + Default
        + NumCast,
    T::RealField: RealField + FloatConst + Copy + Default + NumCast + Zero,
{
    type RealField = AutoFloat<T::RealField, N>;

    fn from_real(re: Self::RealField) -> Self {
        AutoFloat {
            x: T::from_real(re.x),
            dx: unary_op(re.dx, T::from_real),
        }
    }

    fn real(self) -> Self::RealField {
        AutoFloat {
            x: self.x.real(),
            dx: unary_op(self.dx, |v| v.real()),
        }
    }
    fn imaginary(self) -> Self::RealField {
        AutoFloat {
            x: self.x.imaginary(),
            dx: unary_op(self.dx, |v| v.imaginary()),
        }
    }

    fn modulus(self) -> Self::RealField {
        let x = self.x.modulus();
        AutoFloat {
            x: x,
            dx: unary_op(self.dx, |v| {
                (v.real() * v.real() + v.imaginary() * self.x.imaginary()) / x
            }),
        }
    }

    fn modulus_squared(self) -> Self::RealField {
        AutoFloat {
            x: self.x.modulus_squared(),
            dx: unary_op(self.dx, |v| {
                convert::<f64, T::RealField>(2.0)
                    * (v.real() * self.x.real() + v.imaginary() * self.x.imaginary())
            }),
        }
    }

    fn argument(self) -> Self::RealField {
        RealField::atan2(self.imaginary(), self.real())
    }

    fn norm1(self) -> Self::RealField {
        ComplexField::abs(self.real()) + ComplexField::abs(self.imaginary())
    }

    fn scale(self, factor: Self::RealField) -> Self {
        self * Self::from_real(factor)
    }

    fn unscale(self, factor: Self::RealField) -> Self {
        self / Self::from_real(factor)
    }

    fn floor(self) -> Self {
        floor_impl!(self)
    }

    fn ceil(self) -> Self {
        ceil_impl!(self)
    }

    fn round(self) -> Self {
        round_impl!(self)
    }

    fn trunc(self) -> Self {
        trunc_impl!(self)
    }

    fn fract(self) -> Self {
        fract_impl!(self)
    }

    fn abs(self) -> Self::RealField {
        self.modulus()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        mul_add_impl!(self, a, b)
    }

    fn recip(self) -> Self {
        recip_impl!(self)
    }

    fn sin(self) -> Self {
        sin_impl!(self)
    }

    fn cos(self) -> Self {
        cos_impl!(self)
    }

    fn tan(self) -> Self {
        tan_impl!(self)
    }

    fn asin(self) -> Self {
        asin_impl!(self)
    }

    fn acos(self) -> Self {
        acos_impl!(self)
    }

    fn atan(self) -> Self {
        atan_impl!(self)
    }

    fn sin_cos(self) -> (Self, Self) {
        sin_cos_impl!(self)
    }

    fn sinh(self) -> Self {
        sinh_impl!(self)
    }

    fn cosh(self) -> Self {
        cosh_impl!(self)
    }

    fn tanh(self) -> Self {
        tanh_impl!(self)
    }

    fn asinh(self) -> Self {
        asinh_impl!(self)
    }

    fn acosh(self) -> Self {
        acosh_impl!(self)
    }

    fn atanh(self) -> Self {
        atanh_impl!(self)
    }

    fn log(self, b: Self::RealField) -> Self {
        log_impl!(self, b)
    }

    fn log2(self) -> Self {
        log2_impl!(self, <T::RealField as NumCast>::from(2).unwrap())
    }

    fn log10(self) -> Self {
        log10_impl!(self, <T::RealField as NumCast>::from(10).unwrap())
    }

    fn ln(self) -> Self {
        ln_impl!(self)
    }

    fn ln_1p(self) -> Self {
        ln_1p_impl!(self)
    }

    fn sqrt(self) -> Self {
        sqrt_impl!(self)
    }

    fn cbrt(self) -> Self {
        cbrt_impl!(self)
    }

    fn exp(self) -> Self {
        exp_impl!(self)
    }

    fn exp2(self) -> Self {
        exp2_impl!(self)
    }

    fn powi(self, n: i32) -> Self {
        powi_impl!(self, n)
    }

    fn hypot(self, other: Self) -> Self::RealField {
        ComplexField::sqrt(
            self.clone().real() * self.clone().real() - self.clone().imaginary() * self.imaginary()
                + other.clone().real() * other.clone().real()
                - other.clone().imaginary() * other.imaginary(),
        )
    }

    fn conjugate(self) -> Self {
        AutoFloat {
            x: self.x.conjugate(),
            dx: unary_op(self.dx, |v| v.conjugate()),
        }
    }

    fn powf(self, n: Self::RealField) -> Self {
        let x = self.x.powf(n.x);
        powf_impl!(self, n, x)
    }

    fn exp_m1(self) -> Self {
        exp_m1_impl!(self)
    }

    fn powc(self, n: Self) -> Self {
        let x = self.x.powc(n.x);
        powf_impl!(self, n, x)
    }

    fn is_finite(&self) -> bool {
        self.x.is_finite() && Iterator::all(&mut self.dx.iter(), |v| v.is_finite())
    }

    fn try_sqrt(self) -> Option<Self> {
        let sqrt = ComplexField::sqrt(self.x);
        let denom = sqrt * convert::<f64, T>(2.0);
        if denom.is_zero() && Iterator::all(&mut self.dx.iter(), |v| v.is_zero()) {
            None
        } else {
            let factor = T::one() / denom;
            Some(AutoFloat {
                x: sqrt.clone(),
                dx: unary_op(self.dx, |v| v * factor),
            })
        }
    }
}

impl<T, B, const N: usize> Field for AutoFloat<T, N>
where
    B: SimdBool,
    T: Copy + Default + Field<SimdBool = B, Element: Copy + Default>,
{
}

impl<T, B, const N: usize> SimdValue for AutoFloat<T, N>
where
    B: SimdBool,
    T: SimdValue<SimdBool = B, Element: Copy + Default> + Copy + Default,
{
    type Element = AutoFloat<T::Element, N>;
    type SimdBool = B;

    const LANES: usize = T::LANES;

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        AutoFloat {
            x: T::splat(val.x),
            dx: unary_op(val.dx, T::splat),
        }
    }

    #[inline(always)]
    fn extract(&self, i: usize) -> Self::Element {
        AutoFloat {
            x: self.x.extract(i),
            dx: unary_op(self.dx, |v| v.extract(i)),
        }
    }

    #[inline(always)]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        AutoFloat {
            x: self.x.extract_unchecked(i),
            dx: unary_op(self.dx, |v| v.extract_unchecked(i)),
        }
    }

    #[inline(always)]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.x.replace(i, val.x);
        self.dx
            .iter_mut()
            .zip(val.dx.into_iter())
            .for_each(|(l, r)| l.replace(i, r));
    }

    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.x.replace_unchecked(i, val.x);
        self.dx
            .iter_mut()
            .zip(val.dx.into_iter())
            .for_each(|(l, r)| l.replace_unchecked(i, r));
    }

    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        AutoFloat {
            x: self.x.select(cond, other.x),
            dx: binary_op(self.dx, other.dx, |l, r| l.select(cond, r)),
        }
    }
}

#[cfg(test)]
mod tests {
    // use crate::{AutoFloat, F1};
    // use nalgebra::{ComplexField, Matrix3, Vector3};
    // fn make_mtx() -> Matrix3<AutoFloat<f64, f64>> {
    //     [
    //         [
    //             AutoFloat::cst(1.0),
    //             AutoFloat::cst(2.0),
    //             AutoFloat::cst(3.0),
    //         ],
    //         [
    //             AutoFloat::cst(4.0),
    //             AutoFloat::cst(5.0),
    //             AutoFloat::cst(6.0),
    //         ],
    //         [
    //             AutoFloat::cst(7.0),
    //             AutoFloat::cst(8.0),
    //             AutoFloat::cst(9.0),
    //         ],
    //     ]
    //     .into()
    // }

    // // Generic multiply. This tests that AutoFloat is a realfield
    // fn mul<T: ComplexField>(m: Matrix3<T>, v: Vector3<T>) -> Vector3<T> {
    //     m * v
    // }

    // #[test]
    // fn mtx_mul() {
    //     let mtx = make_mtx();
    //     let v = Vector3::from([F1::var(1.0); 3]);
    //     assert_eq!(
    //         mul(mtx, v),
    //         Vector3::from([
    //             AutoFloat { x: 12.0, dx: 12.0 },
    //             AutoFloat { x: 15.0, dx: 15.0 },
    //             AutoFloat { x: 18.0, dx: 18.0 }
    //         ])
    //     );
    // }
}
