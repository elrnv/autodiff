use std::num::FpCategory;

use num_traits::{Float, FloatConst, Zero};

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

impl<T, const N: usize> Float for AutoFloat<T, N>
where
    T: Float + Zero + Copy,
{
    fn nan() -> Self {
        AutoFloat::constant(T::nan())
    }

    fn infinity() -> Self {
        AutoFloat::constant(T::infinity())
    }

    fn neg_infinity() -> Self {
        AutoFloat::constant(T::neg_infinity())
    }

    fn neg_zero() -> Self {
        AutoFloat::constant(T::neg_zero())
    }

    fn min_value() -> Self {
        AutoFloat::constant(T::min_value())
    }

    fn min_positive_value() -> Self {
        AutoFloat::constant(T::min_positive_value())
    }

    fn max_value() -> Self {
        AutoFloat::constant(T::max_value())
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
    pub fn pow(self, n: impl Into<AutoFloat<T, N>>) -> AutoFloat<T, N> {
        self.powf(n.into())
    }
}
