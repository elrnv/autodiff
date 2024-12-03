use nalgebra::{
    convert, ComplexField, DVector, Field, RealField, SVector, Scalar, SimdBool, SimdValue,
};
use num_traits::{FloatConst, NumCast, One, Zero};
use std::ops::{Div, Mul};

use crate::autofloat::float_impl::*;
use crate::{
    autofloat::{binary_op, unary_op},
    AutoFloat,
};

impl<T, const N: usize> RealField for AutoFloat<T, N>
where
    T: RealField + NumCast + FloatConst + Clone,
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
    T: ComplexField + Mul<T::RealField, Output = T> + Div<T::RealField, Output = T> + NumCast,
    T::RealField: RealField + FloatConst + NumCast + Zero,
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
            x: x.clone(),
            dx: unary_op(self.dx, |v| {
                (v.clone().real() * v.clone().real() + v.imaginary() * self.x.clone().imaginary())
                    / x.clone()
            }),
        }
    }

    fn modulus_squared(self) -> Self::RealField {
        AutoFloat {
            x: self.x.clone().modulus_squared(),
            dx: unary_op(self.dx, |v| {
                convert::<f64, T::RealField>(2.0)
                    * (v.clone().real() * self.x.clone().real()
                        + v.imaginary() * self.x.clone().imaginary())
            }),
        }
    }

    fn argument(self) -> Self::RealField {
        RealField::atan2(self.clone().imaginary(), self.clone().real())
    }

    fn norm1(self) -> Self::RealField {
        ComplexField::abs(self.clone().real()) + ComplexField::abs(self.clone().imaginary())
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
        let r = self.real();
        if r >= Self::RealField::zero() {
            r
        } else {
            -r
        }
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
        let x = self.x.clone().powf(n.x.clone());
        powf_impl!(self, n, x)
    }

    fn exp_m1(self) -> Self {
        exp_m1_impl!(self)
    }

    fn powc(self, n: Self) -> Self {
        let x = self.x.clone().powc(n.x.clone());
        powf_impl!(self, n, x)
    }

    fn is_finite(&self) -> bool {
        self.x.is_finite() && Iterator::all(&mut self.dx.iter(), |v| v.is_finite())
    }

    fn try_sqrt(self) -> Option<Self> {
        let sqrt = ComplexField::sqrt(self.x);
        let denom = sqrt.clone() * convert::<f64, T>(2.0);
        if denom.is_zero() && Iterator::all(&mut self.dx.iter(), |v| v.is_zero()) {
            None
        } else {
            let factor = T::one() / denom;
            Some(AutoFloat {
                x: sqrt,
                dx: unary_op(self.dx, |v| v * factor.clone()),
            })
        }
    }
}

impl<T, B, const N: usize> Field for AutoFloat<T, N>
where
    B: SimdBool,
    T: Clone + Field<SimdBool = B, Element: Clone>,
{
}

impl<T, B, const N: usize> SimdValue for AutoFloat<T, N>
where
    B: SimdBool,
    T: SimdValue<SimdBool = B, Element: Clone> + Clone,
{
    type Element = AutoFloat<T::Element, N>;
    type SimdBool = B;

    const LANES: usize = T::LANES;

    fn splat(val: Self::Element) -> Self {
        AutoFloat {
            x: T::splat(val.x),
            dx: unary_op(val.dx, T::splat),
        }
    }

    fn extract(&self, i: usize) -> Self::Element {
        AutoFloat {
            x: self.x.extract(i),
            dx: unary_op(self.dx.clone(), |v| v.extract(i)),
        }
    }

    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        AutoFloat {
            x: self.x.extract_unchecked(i),
            dx: unary_op(self.dx.clone(), |v| v.extract_unchecked(i)),
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

    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.x.replace_unchecked(i, val.x);
        self.dx
            .iter_mut()
            .zip(val.dx.into_iter())
            .for_each(|(l, r)| l.replace_unchecked(i, r));
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        AutoFloat {
            x: self.x.select(cond, other.x),
            dx: binary_op(self.dx, other.dx, |l, r| l.select(cond, r)),
        }
    }
}

impl<T, const N: usize> AutoFloat<T, N>
where
    T: Scalar + Zero + Clone,
{
    pub fn constant_svector<const R: usize>(vec: &SVector<T, R>) -> SVector<Self, R> {
        SVector::from_iterator(vec.iter().map(|v| Self::constant(v.clone())))
    }

    #[cfg(feature = "std")]
    pub fn constant_dvector(vec: &DVector<T>) -> DVector<Self> {
        DVector::from_iterator(vec.len(), vec.iter().map(|v| Self::constant(v.clone())))
    }
}

impl<T, const N: usize> AutoFloat<T, N>
where
    T: Scalar + Zero + One + Clone,
{
    pub fn variable_svector(vec: &SVector<T, N>) -> SVector<Self, N> {
        SVector::from_iterator(
            vec.iter()
                .enumerate()
                .map(|(i, v)| Self::variable(v.clone(), i)),
        )
    }

    #[cfg(feature = "std")]
    pub fn variable_dvector(vec: &DVector<T>) -> DVector<Self> {
        assert_eq!(vec.len(), N);
        DVector::from_iterator(
            vec.len(),
            vec.iter()
                .enumerate()
                .map(|(i, v)| Self::variable(v.clone(), i)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        test::{
            assert_autofloat_eq, assert_autofloat_near, assert_near, compute_numeric_derivative,
            execute_numeric_test,
        },
        AutoFloat, AutoFloat2,
    };
    use nalgebra::{Matrix3, Vector3};

    fn make_matrix() -> Matrix3<AutoFloat<f64, 3>> {
        [
            [
                AutoFloat::constant(1.0),
                AutoFloat::constant(2.0),
                AutoFloat::constant(3.0),
            ],
            [
                AutoFloat::constant(4.0),
                AutoFloat::constant(5.0),
                AutoFloat::constant(6.0),
            ],
            [
                AutoFloat::constant(7.0),
                AutoFloat::constant(8.0),
                AutoFloat::constant(9.0),
            ],
        ]
        .into()
    }

    // Generic multiply. This tests that AutoFloat is a realfield
    fn mul<T: ComplexField>(m: Matrix3<T>, v: Vector3<T>) -> Vector3<T> {
        m * v
    }

    #[test]
    fn matrix_mul() {
        let matrix = make_matrix();
        let vector = Vector3::from(std::array::from_fn(|i| {
            AutoFloat::variable((i + 2) as f64, i)
        }));
        let actual = mul(matrix, vector);

        let expected = Vector3::from([
            AutoFloat::new(42.0, [1.0, 4.0, 7.0]),
            AutoFloat::new(51.0, [2.0, 5.0, 8.0]),
            AutoFloat::new(60.0, [3.0, 6.0, 9.0]),
        ]);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_autofloat_eq!(e, a);
        }
    }

    #[test]
    fn floor() {
        let x = AutoFloat::new(0.2, [1.0, 2.0]).floor();
        assert_autofloat_eq!(x, AutoFloat::new(0.2.floor(), [0.0, 0.0]));
    }

    #[test]
    fn ceil() {
        let x = AutoFloat::new(0.2, [1.0, 2.0]).ceil();
        assert_autofloat_eq!(x, AutoFloat::new(0.2.ceil(), [0.0, 0.0]));
    }

    #[test]
    fn round() {
        let x = AutoFloat::new(0.2, [1.0, 2.0]).round();
        assert_autofloat_eq!(x, AutoFloat::new(0.2.round(), [0.0, 0.0]));
    }

    #[test]
    fn trunc() {
        let x = AutoFloat::new(0.2, [1.0, 2.0]).trunc();
        assert_autofloat_eq!(x, AutoFloat::new(0.2.trunc(), [0.0, 0.0]));
    }

    #[test]
    fn fract() {
        let x = AutoFloat::new(0.2, [1.0, 2.0]).fract();
        assert_autofloat_eq!(x, AutoFloat::new(0.2.fract(), [1.0, 2.0]));
    }

    #[test]
    fn abs() {
        let x: f64 = -2.5;
        execute_numeric_test!(x, abs);
    }

    #[test]
    fn mul_add() {
        let eps: f64 = 1e-6;
        let a = -2.5;
        let b = 1.2;
        let c = 2.7;
        let actual =
            AutoFloat2::variable(a, 0).mul_add(AutoFloat2::constant(b), AutoFloat2::constant(c));
        let deriv = compute_numeric_derivative(a, |v| v.mul_add(b, c));
        assert_autofloat_near!(actual, AutoFloat::new(a.mul_add(b, c), [deriv, 0.0]), eps);
    }

    #[test]
    fn recip() {
        let x: f64 = -2.5;
        execute_numeric_test!(x, recip);
    }

    #[test]
    fn sin() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, sin);
    }

    #[test]
    fn cos() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, cos);
    }

    #[test]
    fn tan() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, tan);
    }

    #[test]
    fn asin() {
        let x: f64 = 0.634;
        execute_numeric_test!(x, asin);
    }

    #[test]
    fn acos() {
        let x: f64 = 0.634;
        execute_numeric_test!(x, acos);
    }

    #[test]
    fn atan() {
        let x: f64 = 0.634;
        execute_numeric_test!(x, atan);
    }

    #[test]
    fn atan2() {
        let x = AutoFloat::new(2.0, [1.3, 0.5]);
        let y = AutoFloat::new(3.0, [-1.0, 1.9]);
        let v1 = y.atan2(x);
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(3.0.atan2(2.0), [-0.4538461538461539, 0.17692307692307693])
        );
    }
    #[test]
    fn sin_cos() {}

    #[test]
    fn sinh() {
        let x: f64 = 0.253;
        execute_numeric_test!(x, sinh);
    }

    #[test]
    fn cosh() {
        let x: f64 = 0.253;
        execute_numeric_test!(x, cosh);
    }

    #[test]
    fn tanh() {
        let x: f64 = 0.253;
        execute_numeric_test!(x, tanh);
    }

    #[test]
    fn asinh() {
        let x: f64 = 0.253;
        execute_numeric_test!(x, asinh);
    }

    #[test]
    fn acosh() {
        let x: f64 = 1.533;
        execute_numeric_test!(x, acosh);
    }

    #[test]
    fn atanh() {
        let x: f64 = 0.253;
        execute_numeric_test!(x, atanh);
    }

    #[test]
    fn log() {
        let eps: f64 = 1e-6;
        let x = 2.38;
        let n = 1.2;
        let actual = AutoFloat2::variable(x, 0).log(AutoFloat2::constant(n));
        let deriv = compute_numeric_derivative(x, |v| v.log(n));
        assert_autofloat_near!(actual, AutoFloat::new(x.log(n), [deriv, 0.0]), eps);
    }

    #[test]
    fn log2() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, log2);
    }

    #[test]
    fn log10() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, log10);
    }

    #[test]
    fn ln() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, ln);
    }

    #[test]
    fn ln_1p() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, ln_1p);
    }

    #[test]
    fn sqrt() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, sqrt);
    }

    #[test]
    fn cbrt() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, cbrt);
    }

    #[test]
    fn exp() {
        let v1 = AutoFloat::new(2.0, [1.3, 0.5]).exp();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(2.0.exp(), [9.605772928609845, 3.694528049465325])
        );

        let x: f64 = 2.38;
        execute_numeric_test!(x, exp);
    }

    #[test]
    fn exp2() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, exp2);
    }

    #[test]
    fn exp_m1() {
        let x: f64 = 2.38;
        execute_numeric_test!(x, exp_m1);
    }

    #[test]
    fn powi() {
        let eps: f64 = 1e-6;
        let x = 2.38;
        let n = 5;
        let actual = AutoFloat2::variable(x, 0).powi(n);
        let deriv = compute_numeric_derivative(x, |v| v.powi(n));
        assert_autofloat_near!(actual, AutoFloat::new(x.powi(n), [deriv, 0.0]), eps);
    }

    #[test]
    fn powf() {
        let eps: f64 = 1e-6;
        let x = 2.38;
        let n = 5.2;
        let actual = AutoFloat2::variable(x, 0).powf(AutoFloat::constant(n));
        let deriv = compute_numeric_derivative(x, |v| v.powf(n));
        assert_autofloat_near!(actual, AutoFloat::new(x.powf(n), [deriv, 0.0]), eps);
    }
}
