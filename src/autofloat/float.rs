use std::num::FpCategory;

use num_traits::{Float, Zero};

use super::float_impl::*;
use crate::{binary_op, unary_op, AutoFloat};

impl<T, const N: usize> Float for AutoFloat<T, N>
where
    T: Float + Zero + Copy + Default,
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
    use super::*;
    use crate::{test::assert_autofloat_eq, AutoFloat, AutoFloat1};

    #[test]
    fn sin() {
        let v1 = AutoFloat::new(2.0, [1.3, 0.5]).sin();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(2.0.sin(), [1.3 * 2.0.cos(), 0.5 * 2.0.cos()])
        );
    }

    #[test]
    fn cos() {
        let v1 = AutoFloat::new(2.0, [1.3, 0.5]).cos();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(2.0.cos(), [1.3 * -2.0.sin(), 0.5 * -2.0.sin()])
        );
    }

    #[test]
    fn tan() {
        let v1 = AutoFloat::new(2.0, [1.3, 0.5]).tan();
        let factor = 1.0 / (2.0.cos() * 2.0.cos());
        assert_autofloat_eq!(v1, AutoFloat::new(2.0.tan(), [1.3 * factor, 0.5 * factor]));
    }

    #[test]
    fn asin() {
        let v1 = AutoFloat::new(0.25, [1.3, 0.5]).asin();
        let factor = 1.0 / (1.0 - (0.25 * 0.25)).sqrt();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(0.25.asin(), [1.3 * factor, 0.5 * factor])
        );
    }

    #[test]
    fn acos() {
        let v1 = AutoFloat::new(0.25, [1.3, 0.5]).acos();
        let factor = -1.0 / (1.0 - (0.25 * 0.25)).sqrt();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(0.25.acos(), [1.3 * factor, 0.5 * factor])
        );
    }

    #[test]
    fn atan() {
        let v1 = AutoFloat::new(0.25, [1.3, 0.5]).atan();
        let factor = 1.0 / ((0.25 * 0.25) + 1.0);
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(0.25.atan(), [1.3 * factor, 0.5 * factor])
        );
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
    fn exp() {
        let v1 = AutoFloat::new(2.0, [1.3, 0.5]).exp();
        assert_autofloat_eq!(
            v1,
            AutoFloat::new(2.0.exp(), [9.605772928609845, 3.694528049465325])
        );
    }

    #[test]
    fn sqrt() {
        let x = AutoFloat1::variable(0.2, 0).sqrt();
        assert_autofloat_eq!(x, AutoFloat::new(0.2.sqrt(), [0.5 / 0.2.sqrt()]));

        // Test that taking a square root of zero does not produce NaN.
        // By convention we take 0/0 = 0 here.
        let x = AutoFloat1::constant(0.0).sqrt();
        assert_autofloat_eq!(x, AutoFloat::new(0.0, [0.0]));
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
            Float::min(a, b),
        );

        let maxf = Float::max(a.x, b.x);
        assert_autofloat_eq!(
            AutoFloat::new(maxf, if maxf == a.x { a.dx } else { b.dx }),
            Float::max(a, b),
        );
    }
}
