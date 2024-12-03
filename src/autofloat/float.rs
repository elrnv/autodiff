use std::num::FpCategory;

use num_traits::{Float, Zero};

use super::float_impl::*;
use crate::{
    autofloat::{binary_op, unary_op},
    AutoFloat,
};

impl<T, const N: usize> Float for AutoFloat<T, N>
where
    T: Float + Zero + Clone,
{
    fn nan() -> Self {
        Self::constant(T::nan())
    }

    fn infinity() -> Self {
        Self::constant(T::infinity())
    }

    fn neg_infinity() -> Self {
        Self::constant(T::neg_infinity())
    }

    fn neg_zero() -> Self {
        Self::constant(T::neg_zero())
    }

    fn min_value() -> Self {
        Self::constant(T::min_value())
    }

    fn min_positive_value() -> Self {
        Self::constant(T::min_positive_value())
    }

    fn max_value() -> Self {
        Self::constant(T::max_value())
    }

    fn is_nan(self) -> bool {
        self.x.is_nan() || self.dx.into_iter().any(|v| v.is_nan())
    }

    fn is_infinite(self) -> bool {
        self.x.is_infinite() || self.dx.into_iter().any(|v| v.is_infinite())
    }

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.dx.into_iter().all(|v| v.is_finite())
    }

    fn is_normal(self) -> bool {
        self.x.is_normal() && self.dx.into_iter().all(|v| v.is_normal())
    }

    fn classify(self) -> FpCategory {
        self.x.classify()
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

    fn abs(self) -> Self {
        abs_impl!(self)
    }

    fn mul_add(self, a: AutoFloat<T, N>, b: AutoFloat<T, N>) -> Self {
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

    fn atan2(self, other: Self) -> Self {
        atan2_impl!(self, other)
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

    fn log(self, b: Self) -> Self {
        log_impl!(self, b)
    }

    fn log2(self) -> Self {
        log2_impl!(self, T::from(2).unwrap())
    }

    fn log10(self) -> Self {
        log10_impl!(self, T::from(10).unwrap())
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

    fn exp_m1(self) -> Self {
        exp_m1_impl!(self)
    }

    fn powi(self, n: i32) -> Self {
        powi_impl!(self, n)
    }

    fn powf(self, n: AutoFloat<T, N>) -> Self {
        let x = self.x.powf(n.x);
        powf_impl!(self, n, x)
    }

    fn signum(self) -> Self {
        AutoFloat::constant(self.x.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.x.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.x.is_sign_negative()
    }

    fn max(self, other: AutoFloat<T, N>) -> Self {
        max_impl!(self, other)
    }

    fn min(self, other: AutoFloat<T, N>) -> Self {
        min_impl!(self, other)
    }

    fn abs_sub(self, other: AutoFloat<T, N>) -> Self {
        if self > other {
            AutoFloat {
                x: Float::abs_sub(self.x, other.x),
                dx: (self - other).dx,
            }
        } else {
            AutoFloat::constant(T::zero())
        }
    }

    fn hypot(self, other: AutoFloat<T, N>) -> Self {
        Float::sqrt(self.clone() * self + other.clone() * other)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    fn epsilon() -> Self {
        AutoFloat::constant(T::epsilon())
    }

    fn to_degrees(self) -> Self {
        to_degrees_impl!(self)
    }

    fn to_radians(self) -> Self {
        to_radians_impl!(self)
    }
}

impl<T, const N: usize> AutoFloat<T, N>
where
    Self: Float,
{
    /// Raise this number to the `n`'th power.
    ///
    /// This is a generic version of `Float::powf`.
    pub fn pow<U>(self, n: U) -> AutoFloat<T, N>
    where
        U: Into<AutoFloat<T, N>>,
    {
        self.powf(n.into())
    }
}

#[cfg(test)]
mod test {
    use num_traits::Signed;

    use super::*;
    use crate::{
        test::{
            assert_autofloat_eq, assert_autofloat_near, assert_near, compute_numeric_derivative,
            execute_numeric_test,
        },
        AutoFloat, AutoFloat1, AutoFloat2,
    };

    #[test]
    fn nan() {
        assert!(AutoFloat2::<f32>::nan().is_nan());
        assert!(AutoFloat2::new(f32::nan(), [0.0, 0.0]).is_nan());
    }

    #[test]
    fn infinity() {
        assert!(AutoFloat2::<f32>::infinity().is_infinite());
        assert!(AutoFloat2::<f32>::infinity().is_positive());
        assert!(AutoFloat2::new(f32::infinity(), [0.0, 0.0]).is_infinite());
    }

    #[test]
    fn neg_infinity() {
        assert!(AutoFloat2::<f32>::neg_infinity().is_infinite());
        assert!(AutoFloat2::<f32>::neg_infinity().is_negative());
        assert!(AutoFloat2::new(f32::neg_infinity(), [0.0, 0.0]).is_infinite());
    }

    #[test]
    fn neg_zero() {
        assert!(AutoFloat2::<f32>::neg_zero().is_zero());
        assert!(AutoFloat2::<f32>::neg_zero().is_negative());
        assert!(AutoFloat2::new(f32::neg_zero(), [0.0, 0.0]).is_zero());
    }

    #[test]
    fn min_value() {
        assert_autofloat_eq!(
            AutoFloat2::<f32>::min_value(),
            AutoFloat2::new(f32::min_value(), [0.0, 0.0])
        );
    }

    #[test]
    fn min_positive_value() {
        assert_autofloat_eq!(
            AutoFloat2::<f32>::min_positive_value(),
            AutoFloat2::new(f32::min_positive_value(), [0.0, 0.0])
        );
    }

    #[test]
    fn max_value() {
        assert_autofloat_eq!(
            AutoFloat2::<f32>::max_value(),
            AutoFloat2::new(f32::max_value(), [0.0, 0.0])
        );
    }

    #[test]
    fn float_to_degrees() {
        let x = Float::to_degrees(AutoFloat1::variable(0.2, 0));
        assert_autofloat_eq!(
            x,
            AutoFloat::new(Float::to_degrees(0.2), [Float::to_degrees(1.0)])
        );
    }

    #[test]
    fn float_to_radians() {
        let x = Float::to_radians(AutoFloat1::variable(0.2, 0));
        assert_autofloat_eq!(
            x,
            AutoFloat::new(Float::to_radians(0.2), [Float::to_radians(1.0)])
        );
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
        let v1 = AutoFloat::new(2.0, [1.3, 0.5]).sin();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(2.0.sin(), [1.3 * 2.0.cos(), 0.5 * 2.0.cos()])
        );

        let x: f64 = 2.38;
        execute_numeric_test!(x, sin);
    }

    #[test]
    fn cos() {
        let v1 = AutoFloat::new(2.0, [1.3, 0.5]).cos();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(2.0.cos(), [1.3 * -2.0.sin(), 0.5 * -2.0.sin()])
        );

        let x: f64 = 2.38;
        execute_numeric_test!(x, cos);
    }

    #[test]
    fn tan() {
        let v1 = AutoFloat::new(2.0, [1.3, 0.5]).tan();
        let factor = 1.0 / (2.0.cos() * 2.0.cos());
        assert_autofloat_eq!(v1, AutoFloat::new(2.0.tan(), [1.3 * factor, 0.5 * factor]));

        let x: f64 = 2.38;
        execute_numeric_test!(x, tan);
    }

    #[test]
    fn asin() {
        let v1 = AutoFloat::new(0.25, [1.3, 0.5]).asin();
        let factor = 1.0 / (1.0 - (0.25 * 0.25)).sqrt();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(0.25.asin(), [1.3 * factor, 0.5 * factor])
        );

        let x: f64 = 0.634;
        execute_numeric_test!(x, asin);
    }

    #[test]
    fn acos() {
        let v1 = AutoFloat::new(0.25, [1.3, 0.5]).acos();
        let factor = -1.0 / (1.0 - (0.25 * 0.25)).sqrt();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(0.25.acos(), [1.3 * factor, 0.5 * factor])
        );

        let x: f64 = 0.634;
        execute_numeric_test!(x, acos);
    }

    #[test]
    fn atan() {
        let v1 = AutoFloat::new(0.25, [1.3, 0.5]).atan();
        let factor = 1.0 / ((0.25 * 0.25) + 1.0);
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(0.25.atan(), [1.3 * factor, 0.5 * factor])
        );

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
        let x = AutoFloat1::variable(0.2, 0).sqrt();
        assert_autofloat_eq!(x, AutoFloat::new(0.2.sqrt(), [0.5 / 0.2.sqrt()]));

        // Test that taking a square root of zero does not produce NaN.
        // By convention we take 0/0 = 0 here.
        let x = AutoFloat1::constant(0.0).sqrt();
        assert_autofloat_eq!(x, AutoFloat::new(0.0, [0.0]));

        let x: f64 = 2.38;
        execute_numeric_test!(x, sqrt);
    }

    #[test]
    fn cbrt() {
        let x = AutoFloat1::variable(0.2, 0).cbrt();
        assert_autofloat_eq!(
            x,
            AutoFloat::new(0.2.cbrt(), [1.0 / (3.0 * 0.2.cbrt() * 0.2.cbrt())])
        );

        // Test that taking a cube root of zero does not produce NaN.
        // By convention we take 0/0 = 0 here.
        let x = AutoFloat1::constant(0.0).cbrt();
        assert_autofloat_eq!(x, AutoFloat::new(0.0, [0.0]));

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

    #[test]
    fn float_min_max_value() {
        // Test basic arithmetic on F.
        let a = AutoFloat1::variable(1.0, 0);
        let mut b = AutoFloat1::constant(2.0);

        b = Float::min(b, a);
        assert_autofloat_eq!(AutoFloat::new(1.0, [1.0]), b);

        b = AutoFloat::constant(2.0);
        b = Float::min(a, b);
        assert_autofloat_eq!(AutoFloat::new(1.0, [1.0]), b);

        let b = AutoFloat::constant(2.0);

        let c = Float::max(a, b);
        assert_autofloat_eq!(AutoFloat::new(2.0, [0.0]), c);

        // Make sure that our min and max are consistent with the internal implementation to avoid
        // inconsistencies in the future. In particular we look at tie breaking.

        let b = AutoFloat::constant(1.0);
        let minf = Float::min(a.x, b.x);
        assert_autofloat_eq!(
            AutoFloat::new(minf, if minf == a.x { a.dx } else { b.dx }),
            Float::min(a, b)
        );

        let maxf = Float::max(a.x, b.x);
        assert_autofloat_eq!(
            AutoFloat::new(maxf, if maxf == a.x { a.dx } else { b.dx }),
            Float::max(a, b)
        );
    }
}
