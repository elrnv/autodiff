use nalgebra::{convert, ComplexField, Field, RealField, SimdBool, SimdValue};
use num_traits::{Float, FloatConst, Zero};
use std::ops::{Add, Div, DivAssign, Mul, MulAssign, Neg, Sub};

use crate::{binary_op, unary_op, AutoFloat};

impl<T, const N: usize> RealField for AutoFloat<T, N>
where
    T: RealField + Float + FloatConst,
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
        if self.x < other.x {
            other
        } else {
            self
        }
    }
    fn min(self, other: Self) -> Self {
        if self.x > other.x {
            other
        } else {
            self
        }
    }
    fn clamp(self, min: Self, max: Self) -> Self {
        RealField::max(RealField::min(self, max), min)
    }
    fn atan2(self, other: Self) -> Self {
        self.atan2_impl(other)
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
        let v_min = <T as RealField>::min_value();
        let d_min = <D as RealField>::min_value();
        if v_min.is_some() && d_min.is_some() {
            Some(Self {
                x: v_min.unwrap(),
                dx: d_min.unwrap(),
            })
        } else {
            None
        }
    }

    fn max_value() -> Option<Self> {
        let v_max = <T as RealField>::max_value();
        let d_max = <D as RealField>::max_value();
        if v_max.is_some() && d_max.is_some() {
            Some(Self {
                x: v_max.unwrap(),
                dx: d_max.unwrap(),
            })
        } else {
            None
        }
    }
}

impl<T, const N: usize> ComplexField for AutoFloat<T, N>
where
    T: ComplexField
        + Mul<T::RealField, Output = T>
        + Div<T::RealField, Output = T>
        + Copy
        + Default
        + Float,
    T::RealField: RealField + Float + FloatConst + Default,
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
        let x = self.x.clone().modulus();
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
        RealField::atan2(self.clone().imaginary(), self.real())
    }
    fn norm1(self) -> Self::RealField {
        ComplexField::abs(self.clone().real()) + ComplexField::abs(self.imaginary())
    }
    fn scale(self, factor: Self::RealField) -> Self {
        self * factor
    }
    fn unscale(self, factor: Self::RealField) -> Self {
        self / factor
    }
    fn floor(self) -> Self {
        Float::floor(self)
    }
    fn ceil(self) -> Self {
        Float::ceil(self)
    }
    fn round(self) -> Self {
        Float::round(self)
    }
    fn trunc(self) -> Self {
        Float::trunc(self)
    }
    fn fract(self) -> Self {
        Float::fract(self)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        Float::mul_add(self, a, b)
    }
    fn abs(self) -> Self::RealField {
        self.modulus()
    }
    fn hypot(self, other: Self) -> Self::RealField {
        ComplexField::sqrt(
            self.clone().real() * self.clone().real() - self.clone().imaginary() * self.imaginary()
                + other.clone().real() * other.clone().real()
                - other.clone().imaginary() * other.imaginary(),
        )
    }
    fn recip(self) -> Self {
        Float::recip(self)
    }
    fn conjugate(self) -> Self {
        AutoFloat {
            x: self.x.conjugate(),
            dx: unary_op(self.dx, |v| v.conjugate()),
        }
    }
    fn sin(self) -> Self {
        Float::sin(self)
    }
    fn cos(self) -> Self {
        Float::cos(self)
    }
    fn sin_cos(self) -> (Self, Self) {
        Float::sin_cos(self)
    }
    fn tan(self) -> Self {
        Float::tan(self)
    }
    fn asin(self) -> Self {
        Float::asin(self)
    }
    fn acos(self) -> Self {
        Float::acos(self)
    }
    fn atan(self) -> Self {
        Float::atan(self)
    }
    fn sinh(self) -> Self {
        Float::sinh(self)
    }
    fn cosh(self) -> Self {
        Float::cosh(self)
    }
    fn tanh(self) -> Self {
        Float::tanh(self)
    }
    fn asinh(self) -> Self {
        Float::asinh(self)
    }
    fn acosh(self) -> Self {
        Float::acosh(self)
    }
    fn atanh(self) -> Self {
        Float::atanh(self)
    }
    fn log(self, b: Self::RealField) -> Self {
        let s_r = self.x.clone();
        let b_r = b.x;
        let ln_b_r = ComplexField::ln(b_r);
        AutoFloat {
            x: ComplexField::log(self.x, b.x),
            dx: -b.dx * ComplexField::ln(s_r.clone()) / (b_r * ln_b_r * ln_b_r)
                + self.dx / (s_r * ln_b_r),
        }
    }
    fn log2(self) -> Self {
        Float::log2(self)
    }
    fn log10(self) -> Self {
        Float::log10(self)
    }
    fn ln(self) -> Self {
        Float::ln(self)
    }
    fn ln_1p(self) -> Self {
        Float::ln_1p(self)
    }
    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }
    fn exp(self) -> Self {
        Float::exp(self)
    }
    fn exp2(self) -> Self {
        Float::exp2(self)
    }
    fn exp_m1(self) -> Self {
        Float::exp_m1(self)
    }
    fn powi(self, n: i32) -> Self {
        Float::powi(self, n)
    }

    // TODO: Fix the following implementations for complex values.
    fn powf(self, n: Self::RealField) -> Self {
        let self_r = self.x.clone();
        let n_r = n.x;
        // Avoid imaginary values in the ln.
        let dn = if n.dx.is_zero() {
            D::zero()
        } else {
            n.dx * ComplexField::ln(self_r.clone())
        };

        let x = ComplexField::powf(self_r.clone(), n_r.clone());

        // Avoid division by zero.
        let x_df = if self.x.is_zero() && x.is_zero() {
            D::zero()
        } else {
            self.dx * (x.clone() * n_r / self_r)
        };

        AutoFloat {
            x: x.clone(),
            dx: dn * x + x_df,
        }
    }

    fn powc(self, n: Self) -> Self {
        let self_r = self.x.clone();
        let n_r = n.x;
        // Avoid imaginary values in the ln.
        let dn = if n.dx.is_zero() {
            D::zero()
        } else {
            n.dx * ComplexField::ln(self_r.clone())
        };

        let x = ComplexField::powc(self_r.clone(), n_r.clone());

        // Avoid division by zero.
        let x_df = if self.x.clone().is_zero() && x.is_zero() {
            D::zero()
        } else {
            self.dx * (x.clone() * n_r / self_r)
        };

        AutoFloat {
            x: x.clone(),
            dx: dn * x + x_df,
        }
    }
    fn cbrt(self) -> Self {
        Float::cbrt(self)
    }
    fn is_finite(&self) -> bool {
        self.x.is_finite() && self.dx.iter().all(|v| v.is_finite())
    }

    fn try_sqrt(self) -> Option<Self> {
        let sqrt = ComplexField::sqrt(self.x);
        let denom = sqrt * convert::<f64, T>(2.0);
        if denom.is_zero() && self.dx.into_iter().all(|v| v.is_zero()) {
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
    use crate::{AutoFloat, F1};
    use nalgebra::{ComplexField, Matrix3, Vector3};
    fn make_mtx() -> Matrix3<AutoFloat<f64, f64>> {
        [
            [
                AutoFloat::cst(1.0),
                AutoFloat::cst(2.0),
                AutoFloat::cst(3.0),
            ],
            [
                AutoFloat::cst(4.0),
                AutoFloat::cst(5.0),
                AutoFloat::cst(6.0),
            ],
            [
                AutoFloat::cst(7.0),
                AutoFloat::cst(8.0),
                AutoFloat::cst(9.0),
            ],
        ]
        .into()
    }

    // Generic multiply. This tests that AutoFloat is a realfield
    fn mul<T: ComplexField>(m: Matrix3<T>, v: Vector3<T>) -> Vector3<T> {
        m * v
    }

    #[test]
    fn mtx_mul() {
        let mtx = make_mtx();
        let v = Vector3::from([F1::var(1.0); 3]);
        assert_eq!(
            mul(mtx, v),
            Vector3::from([
                AutoFloat { x: 12.0, dx: 12.0 },
                AutoFloat { x: 15.0, dx: 15.0 },
                AutoFloat { x: 18.0, dx: 18.0 }
            ])
        );
    }
}
