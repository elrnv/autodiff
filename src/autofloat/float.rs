use std::num::FpCategory;

use num_traits::{float::FloatCore, Float, FloatConst, Zero};

use super::{binary_op, unary_op, AutoFloat};

impl<T, const N: usize> FloatConst for AutoFloat<T, N>
where
    T: FloatConst + Zero + Copy,
{
    fn E() -> Self {
        AutoFloat::constant(T::E())
    }

    fn FRAC_1_PI() -> Self {
        AutoFloat::constant(T::FRAC_1_PI())
    }

    fn FRAC_1_SQRT_2() -> Self {
        AutoFloat::constant(T::FRAC_1_SQRT_2())
    }

    fn FRAC_2_PI() -> Self {
        AutoFloat::constant(T::FRAC_2_PI())
    }

    fn FRAC_2_SQRT_PI() -> Self {
        AutoFloat::constant(T::FRAC_2_SQRT_PI())
    }

    fn FRAC_PI_2() -> Self {
        AutoFloat::constant(T::FRAC_PI_2())
    }

    fn FRAC_PI_3() -> Self {
        AutoFloat::constant(T::FRAC_PI_3())
    }

    fn FRAC_PI_4() -> Self {
        AutoFloat::constant(T::FRAC_PI_4())
    }

    fn FRAC_PI_6() -> Self {
        AutoFloat::constant(T::FRAC_PI_6())
    }

    fn FRAC_PI_8() -> Self {
        AutoFloat::constant(T::FRAC_PI_8())
    }

    fn LN_10() -> Self {
        AutoFloat::constant(T::LN_10())
    }

    fn LN_2() -> Self {
        AutoFloat::constant(T::LN_2())
    }

    fn LOG10_E() -> Self {
        AutoFloat::constant(T::LOG10_E())
    }

    fn LOG2_E() -> Self {
        AutoFloat::constant(T::LOG2_E())
    }

    fn PI() -> Self {
        AutoFloat::constant(T::PI())
    }

    fn SQRT_2() -> Self {
        AutoFloat::constant(T::SQRT_2())
    }
}

impl<T, const N: usize> FloatCore for AutoFloat<T, N>
where
    T: FloatCore,
{
    fn infinity() -> Self {
        Self::constant(T::infinity())
    }

    fn neg_infinity() -> Self {
        Self::constant(T::neg_infinity())
    }

    fn nan() -> Self {
        Self::constant(T::nan())
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

    fn epsilon() -> Self {
        Self::constant(T::epsilon())
    }

    fn max_value() -> Self {
        Self::constant(T::max_value())
    }

    fn classify(self) -> FpCategory {
        self.x.classify()
    }

    fn to_degrees(self) -> Self {
        AutoFloat {
            x: FloatCore::to_degrees(self.x),
            dx: unary_op(self.dx, FloatCore::to_degrees),
        }
    }

    fn to_radians(self) -> Self {
        AutoFloat {
            x: FloatCore::to_radians(self.x),
            dx: unary_op(self.dx, FloatCore::to_radians),
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    fn max(self, other: AutoFloat<T, N>) -> Self {
        if self.x < other.x {
            other
        } else {
            self
        }
    }

    fn min(self, other: AutoFloat<T, N>) -> Self {
        if self.x > other.x {
            other
        } else {
            self
        }
    }
}

#[cfg(feature = "std")]
impl<T, const N: usize> Float for AutoFloat<T, N>
where
    T: Float + Zero + Copy,
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
        AutoFloat {
            x: self.x.floor(),
            dx: self.dx,
        }
    }

    fn ceil(self) -> Self {
        AutoFloat {
            x: self.x.ceil(),
            dx: self.dx,
        }
    }

    fn round(self) -> Self {
        AutoFloat {
            x: self.x.round(),
            dx: self.dx,
        }
    }

    fn trunc(self) -> Self {
        AutoFloat {
            x: self.x.trunc(),
            dx: self.dx,
        }
    }

    fn fract(self) -> Self {
        AutoFloat {
            x: self.x.fract(),
            dx: self.dx,
        }
    }

    fn abs(self) -> Self {
        if self.x >= T::zero() {
            self
        } else {
            -self
        }
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

    fn mul_add(self, a: AutoFloat<T, N>, b: AutoFloat<T, N>) -> Self {
        AutoFloat {
            x: self.x.mul_add(a.x, b.x),
            dx: binary_op(
                binary_op(self.dx, a.dx, |l, r| l * a.x + r * self.x),
                b.dx,
                |l, r| l + r,
            ),
        }
    }

    fn recip(self) -> Self {
        let factor = -T::one() / self.x * self.x;
        AutoFloat {
            x: self.x.recip(),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn powi(self, n: i32) -> Self {
        let factor = self.x.powi(n - 1) * T::from(n).unwrap();
        AutoFloat {
            x: self.x.powi(n),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn powf(self, n: AutoFloat<T, N>) -> Self {
        let x = Float::powf(self.x, n.x);

        // Avoid division by zero.
        let x_df_factor = if self.x.is_zero() && x.is_zero() {
            T::zero()
        } else {
            x * n.x / self.x
        };

        AutoFloat {
            x,
            dx: unary_op(n.dx, |v| {
                // Avoid imaginary values in the ln
                let dn = if v.is_zero() {
                    T::zero()
                } else {
                    v * Float::ln(self.x)
                };

                dn * x + v * x_df_factor
            }),
        }
    }

    fn sqrt(self) -> Self {
        let denom = self.x.sqrt() * T::from(2.0).unwrap();
        let factor = if denom.is_zero() {
            T::zero()
        } else {
            T::one() / denom
        };

        AutoFloat {
            x: self.x.sqrt(),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn exp(self) -> Self {
        let exp = Float::exp(self.x);
        AutoFloat {
            x: exp,
            dx: unary_op(self.dx, |v| v * exp),
        }
    }

    fn exp2(self) -> Self {
        let exp2 = Float::exp2(self.x);
        let factor = T::from(2.0).unwrap().ln() * exp2;
        AutoFloat {
            x: exp2,
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn ln(self) -> Self {
        let factor = self.x.recip();
        AutoFloat {
            x: Float::ln(self.x),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn log(self, b: AutoFloat<T, N>) -> Self {
        let ln_bx = Float::ln(b.x);
        let factor_bdx = -Float::ln(self.x) / (b.x * ln_bx * ln_bx);
        let factor_sdx = T::one() / self.x * ln_bx;

        AutoFloat {
            x: Float::log(self.x, b.x),
            dx: binary_op(self.dx, b.dx, |l, r| r * factor_bdx + l * factor_sdx),
        }
    }

    fn log2(self) -> Self {
        Float::log(self, AutoFloat::constant(T::from(2.0).unwrap()))
    }

    fn log10(self) -> Self {
        Float::log(self, AutoFloat::constant(T::from(10.0).unwrap()))
    }

    fn max(self, other: AutoFloat<T, N>) -> Self {
        if self.x < other.x {
            other
        } else {
            self
        }
    }

    fn min(self, other: AutoFloat<T, N>) -> Self {
        if self.x > other.x {
            other
        } else {
            self
        }
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

    fn cbrt(self) -> Self {
        let x_cbrt = Float::cbrt(self.x);
        let denom = x_cbrt * x_cbrt * T::from(3.0).unwrap();
        let factor = if denom.is_zero() {
            T::zero()
        } else {
            T::one() / denom
        };

        AutoFloat {
            x: x_cbrt,
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn hypot(self, other: AutoFloat<T, N>) -> Self {
        Float::sqrt(self.clone() * self + other.clone() * other)
    }

    fn sin(self) -> Self {
        let cos_x = Float::cos(self.x);
        AutoFloat {
            x: Float::sin(self.x),
            dx: unary_op(self.dx, |v| v * cos_x),
        }
    }

    fn cos(self) -> Self {
        let sin_x = -Float::sin(self.x);
        AutoFloat {
            x: Float::cos(self.x),
            dx: unary_op(self.dx, |v| v * sin_x),
        }
    }

    fn tan(self) -> Self {
        let tan_x = Float::tan(self.x);
        let factor = tan_x * tan_x + T::one();
        AutoFloat {
            x: tan_x,
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn asin(self) -> Self {
        let factor = T::one() / (T::one() - self.x.clone() * self.x).sqrt();
        AutoFloat {
            x: Float::asin(self.x.clone()),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn acos(self) -> Self {
        let factor = -T::one() / (T::one() - self.x * self.x).sqrt();
        AutoFloat {
            x: Float::acos(self.x),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn atan(self) -> Self {
        let factor = T::one() / (self.x * self.x + T::one());
        AutoFloat {
            x: Float::atan(self.x),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn atan2(self, other: AutoFloat<T, N>) -> Self {
        let factor = T::one() / (self.x * self.x + other.x * other.x);
        AutoFloat {
            x: Float::atan2(self.x, other.x),
            dx: binary_op(self.dx, other.dx, |l, r| {
                (l * other.x - r * self.x) * factor
            }),
        }
    }

    fn sin_cos(self) -> (AutoFloat<T, N>, AutoFloat<T, N>) {
        let (s, c) = Float::sin_cos(self.x);
        let sn = AutoFloat {
            x: s,
            dx: unary_op(self.dx, |v| v * c),
        };
        let s_neg = -s;
        let cn = AutoFloat {
            x: c,
            dx: unary_op(self.dx, |v| v * s_neg),
        };
        (sn, cn)
    }

    fn exp_m1(self) -> Self {
        let exp_x = Float::exp(self.x);
        AutoFloat {
            x: Float::exp_m1(self.x),
            dx: unary_op(self.dx, |v| v * exp_x),
        }
    }

    fn ln_1p(self) -> Self {
        let factor = T::one() / (self.x + T::one());
        AutoFloat {
            x: Float::ln_1p(self.x),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn sinh(self) -> Self {
        let cosh_x = Float::cosh(self.x);
        AutoFloat {
            x: Float::sinh(self.x),
            dx: unary_op(self.dx, |v| v * cosh_x),
        }
    }

    fn cosh(self) -> Self {
        let sinh_x = Float::sinh(self.x);
        AutoFloat {
            x: Float::cosh(self.x),
            dx: unary_op(self.dx, |v| v * sinh_x),
        }
    }

    fn tanh(self) -> Self {
        let tanhx = Float::tanh(self.x);
        let factor = T::one() - tanhx * tanhx;
        AutoFloat {
            x: tanhx,
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn asinh(self) -> Self {
        let factor = T::one() / (self.x * self.x + T::one()).sqrt();
        AutoFloat {
            x: Float::asinh(self.x),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn acosh(self) -> Self {
        let factor = T::one() / (self.x * self.x - T::one()).sqrt();
        AutoFloat {
            x: Float::acosh(self.x),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn atanh(self) -> Self {
        let factor = T::one() / (-self.x * self.x + T::one());
        AutoFloat {
            x: Float::atanh(self.x),
            dx: unary_op(self.dx, |v| v * factor),
        }
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    fn epsilon() -> Self {
        AutoFloat::constant(T::epsilon())
    }

    fn to_degrees(self) -> Self {
        AutoFloat {
            x: Float::to_degrees(self.x),
            dx: unary_op(self.dx, Float::to_degrees),
        }
    }

    fn to_radians(self) -> Self {
        AutoFloat {
            x: Float::to_radians(self.x),
            dx: unary_op(self.dx, Float::to_radians),
        }
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
    use crate::{autofloat::test::assert_autofloat_eq, AutoFloat, AutoFloat1};

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

    #[test]
    fn float_core_to_degrees() {
        let x = FloatCore::to_degrees(AutoFloat1::variable(0.2, 0));
        assert_autofloat_eq!(
            x,
            AutoFloat::new(FloatCore::to_degrees(0.2), [FloatCore::to_degrees(1.0)])
        );
    }

    #[test]
    fn float_core_to_radians() {
        let x = FloatCore::to_radians(AutoFloat1::variable(0.2, 0));
        assert_autofloat_eq!(
            x,
            AutoFloat::new(FloatCore::to_radians(0.2), [FloatCore::to_radians(1.0)])
        );
    }

    #[test]
    fn float_core_min_max_value() {
        // Test basic arithmetic on F.
        let a = AutoFloat1::variable(1.0, 0);
        let mut b = AutoFloat1::constant(2.0);

        b = FloatCore::min(b, a);
        assert_autofloat_eq!(AutoFloat::new(1.0, [1.0]), b);

        b = AutoFloat::constant(2.0);
        b = FloatCore::min(a, b);
        assert_autofloat_eq!(AutoFloat::new(1.0, [1.0]), b);

        let b = AutoFloat::constant(2.0);

        let c = FloatCore::max(a, b);
        assert_autofloat_eq!(AutoFloat::new(2.0, [0.0]), c);

        // Make sure that our min and max are consistent with the internal implementation to avoid
        // inconsistencies in the future. In particular we look at tie breaking.

        let b = AutoFloat::constant(1.0);
        let minf = FloatCore::min(a.x, b.x);
        assert_autofloat_eq!(
            AutoFloat::new(minf, if minf == a.x { a.dx } else { b.dx }),
            FloatCore::min(a, b),
        );

        let maxf = FloatCore::max(a.x, b.x);
        assert_autofloat_eq!(
            AutoFloat::new(maxf, if maxf == a.x { a.dx } else { b.dx }),
            FloatCore::max(a, b),
        );
    }
}
