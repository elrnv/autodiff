use super::F;
use nalgebra::{convert, ComplexField, Field, RealField, SimdBool, SimdValue};
use num_traits::{Float, FloatConst, Zero};
use std::ops::{Add, Div, DivAssign, Mul, MulAssign, Neg, Sub};

impl<V, D> RealField for F<V, D>
where
    V: RealField + Float + FloatConst,
    D: RealField
        + std::fmt::Debug
        + Float
        + Zero
        + Neg<Output = D>
        + Mul<Output = D>
        + Mul<V, Output = D>
        + Add<Output = D>
        + DivAssign<V>
        + MulAssign<V>
        + Div<V, Output = D>
        + Div<Output = D>
        + Sub<Output = D>
        + Clone
        + PartialOrd,
{
    fn is_sign_positive(&self) -> bool {
        self.x.is_sign_positive()
    }
    fn is_sign_negative(&self) -> bool {
        self.x.is_sign_negative()
    }
    fn copysign(self, to: Self) -> Self {
        F {
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
        let v_min = <V as RealField>::min_value();
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
        let v_max = <V as RealField>::max_value();
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

impl<V, D> ComplexField for F<V, D>
where
    V: ComplexField + Mul<V::RealField, Output = V> + Div<V::RealField, Output = V>,
    D: ComplexField
        + std::fmt::Debug
        + Zero
        + Neg<Output = D>
        + Mul<Output = D>
        + Mul<V::RealField, Output = D>
        + Div<V::RealField, Output = D>
        + Mul<V, Output = D>
        + Add<Output = D>
        + DivAssign<V>
        + MulAssign<V>
        + Div<V, Output = D>
        + Div<Output = D>
        + Sub<Output = D>
        + Clone,
    V::RealField: RealField + Float + FloatConst,
    D::RealField: RealField
        + std::fmt::Debug
        + Float
        + Zero
        + Neg<Output = D::RealField>
        + Mul<Output = D::RealField>
        + Mul<V::RealField, Output = D::RealField>
        + Mul<V, Output = D>
        + Add<Output = D::RealField>
        + DivAssign<V::RealField>
        + MulAssign<V::RealField>
        + Div<V::RealField, Output = D::RealField>
        + Div<Output = D::RealField>
        + Sub<Output = D::RealField>
        + Clone
        + PartialOrd,
{
    type RealField = F<V::RealField, D::RealField>;
    fn from_real(re: Self::RealField) -> Self {
        F {
            x: V::from_real(re.x),
            dx: D::from_real(re.dx),
        }
    }
    fn real(self) -> Self::RealField {
        F {
            x: self.x.real(),
            dx: self.dx.real(),
        }
    }
    fn imaginary(self) -> Self::RealField {
        F {
            x: self.x.imaginary(),
            dx: self.dx.imaginary(),
        }
    }
    fn modulus(self) -> Self::RealField {
        let x = self.x.clone().modulus();
        F {
            x: x.clone(),
            dx: (self.dx.clone().real() * self.x.clone().real()
                + self.dx.imaginary() * self.x.imaginary())
                / x,
        }
    }
    fn modulus_squared(self) -> Self::RealField {
        F {
            x: self.x.clone().modulus_squared(),
            dx: convert::<f64, D::RealField>(2.0)
                * (self.dx.clone().real() * self.x.clone().real()
                    + self.dx.imaginary() * self.x.imaginary()),
        }
    }
    fn argument(self) -> Self::RealField {
        RealField::atan2(self.clone().imaginary(), self.real())
    }
    fn norm1(self) -> Self::RealField {
        ComplexField::abs(self.clone().real()) + ComplexField::abs(self.imaginary())
    }
    fn scale(self, factor: Self::RealField) -> Self {
        F {
            x: self.x.clone() * factor.x.clone(),
            dx: self.dx * factor.x + factor.dx * self.x,
        }
    }
    fn unscale(self, factor: Self::RealField) -> Self {
        F {
            x: self.x.clone() / factor.x.clone(),
            dx: (self.dx * factor.x.clone() - factor.dx * self.x) / (factor.x.clone() * factor.x),
        }
    }
    fn floor(self) -> Self {
        F {
            x: self.x.floor(),
            dx: self.dx,
        }
    }
    fn ceil(self) -> Self {
        F {
            x: self.x.ceil(),
            dx: self.dx,
        }
    }
    fn round(self) -> Self {
        F {
            x: self.x.round(),
            dx: self.dx,
        }
    }
    fn trunc(self) -> Self {
        F {
            x: self.x.trunc(),
            dx: self.dx,
        }
    }
    fn fract(self) -> Self {
        F {
            x: self.x.fract(),
            dx: self.dx,
        }
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        F {
            x: self.x.clone().mul_add(a.x.clone(), b.x),
            dx: self.dx * a.x + a.dx * self.x + b.dx,
        }
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
        let x = self.x.clone();
        F {
            x: self.x.recip(),
            dx: -self.dx / (x.clone() * x),
        }
    }
    fn conjugate(self) -> Self {
        F {
            x: self.x.conjugate(),
            dx: self.dx.conjugate(),
        }
    }
    fn sin(self) -> Self {
        F {
            x: ComplexField::sin(self.x.clone()),
            dx: self.dx * ComplexField::cos(self.x),
        }
    }
    fn cos(self) -> Self {
        F {
            x: ComplexField::cos(self.x.clone()),
            dx: -self.dx * ComplexField::sin(self.x),
        }
    }
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = ComplexField::sin_cos(self.x);
        let sn = F {
            x: s.clone(),
            dx: self.dx.clone() * c.clone(),
        };
        let cn = F {
            x: c,
            dx: self.dx * (-s),
        };
        (sn, cn)
    }
    fn tan(self) -> Self {
        let t = ComplexField::tan(self.x);
        F {
            x: t.clone(),
            dx: self.dx * (t.clone() * t + V::one()),
        }
    }
    fn asin(self) -> Self {
        F {
            x: ComplexField::asin(self.x.clone()),
            dx: self.dx / ComplexField::sqrt(V::one() - self.x.clone() * self.x),
        }
    }
    fn acos(self) -> Self {
        F {
            x: ComplexField::acos(self.x.clone()),
            dx: -self.dx / ComplexField::sqrt(V::one() - self.x.clone() * self.x),
        }
    }
    fn atan(self) -> Self {
        F {
            x: ComplexField::atan(self.x.clone()),
            dx: self.dx / (self.x.clone() * self.x + V::one()),
        }
    }
    fn sinh(self) -> Self {
        F {
            x: ComplexField::sinh(self.x.clone()),
            dx: self.dx * ComplexField::cosh(self.x),
        }
    }
    fn cosh(self) -> Self {
        F {
            x: ComplexField::cosh(self.x.clone()),
            dx: self.dx * ComplexField::sinh(self.x),
        }
    }
    fn tanh(self) -> Self {
        let tanhx = ComplexField::tanh(self.x.clone());
        F {
            x: ComplexField::tanh(self.x),
            dx: self.dx * (V::one() - tanhx.clone() * tanhx),
        }
    }
    fn asinh(self) -> Self {
        F {
            x: ComplexField::asinh(self.x.clone()),
            dx: self.dx / (self.x.clone() * self.x + V::one()).sqrt(),
        }
    }
    fn acosh(self) -> Self {
        F {
            x: ComplexField::acosh(self.x.clone()),
            dx: self.dx / (self.x.clone() * self.x - V::one()).sqrt(),
        }
    }
    fn atanh(self) -> Self {
        F {
            x: ComplexField::atanh(self.x.clone()),
            dx: self.dx / (-self.x.clone() * self.x + V::one()),
        }
    }
    fn log(self, b: Self::RealField) -> Self {
        let s_r = self.x.clone();
        let b_r = b.x;
        let ln_b_r = ComplexField::ln(b_r);
        F {
            x: ComplexField::log(self.x, b.x),
            dx: -b.dx * ComplexField::ln(s_r.clone()) / (b_r * ln_b_r * ln_b_r)
                + self.dx / (s_r * ln_b_r),
        }
    }
    fn log2(self) -> Self {
        let s_r = self.x.clone();
        let b = convert::<f64, V::RealField>(2.0);
        let ln_b = ComplexField::ln(b);
        F {
            x: ComplexField::log(self.x, b),
            dx: self.dx / (s_r * ln_b),
        }
    }
    fn log10(self) -> Self {
        let s_r = self.x.clone();
        let b = convert::<f64, V::RealField>(10.0);
        let ln_b = ComplexField::ln(b);
        F {
            x: ComplexField::log(self.x, b),
            dx: self.dx / (s_r * ln_b),
        }
    }
    fn ln(self) -> Self {
        F {
            x: ComplexField::ln(self.x.clone()),
            dx: self.dx * self.x.recip(),
        }
    }
    fn ln_1p(self) -> Self {
        F {
            x: ComplexField::ln_1p(self.x.clone()),
            dx: self.dx / (self.x + V::one()),
        }
    }
    fn sqrt(self) -> Self {
        F {
            x: self.x.clone().sqrt(),
            dx: {
                let denom = self.x.sqrt() * convert::<f64, V>(2.0);
                if denom == V::zero() && self.dx == D::zero() {
                    D::zero()
                } else {
                    self.dx / denom
                }
            },
        }
    }
    fn exp(self) -> Self {
        let exp = ComplexField::exp(self.x);
        F {
            x: exp.clone(),
            dx: self.dx * exp,
        }
    }
    fn exp2(self) -> Self {
        let exp2 = ComplexField::exp2(self.x);
        F {
            x: exp2.clone(),
            dx: self.dx * convert::<f64, V>(2.0).ln() * exp2,
        }
    }
    fn exp_m1(self) -> Self {
        F {
            x: ComplexField::exp_m1(self.x.clone()),
            dx: self.dx * ComplexField::exp(self.x),
        }
    }
    fn powi(self, n: i32) -> Self {
        F {
            x: self.x.clone().powi(n),
            dx: self.dx * (self.x.powi(n - 1) * convert::<f64, V>(f64::from(n))),
        }
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

        F {
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

        F {
            x: x.clone(),
            dx: dn * x + x_df,
        }
    }
    fn cbrt(self) -> Self {
        let x_cbrt = ComplexField::cbrt(self.x);
        F {
            x: x_cbrt.clone(),
            dx: {
                let denom = x_cbrt.clone() * x_cbrt * convert::<f64, V>(3.0);
                if denom == V::zero() && self.dx == D::zero() {
                    D::zero()
                } else {
                    self.dx / denom
                }
            },
        }
    }
    fn is_finite(&self) -> bool {
        self.x.is_finite() && self.dx.is_finite()
    }
    fn try_sqrt(self) -> Option<Self> {
        let sqrt = self.x.sqrt();
        Some(F {
            x: sqrt.clone(),
            dx: {
                let denom = sqrt * convert::<f64, V>(2.0);
                if denom == V::zero() && self.dx == D::zero() {
                    return None;
                } else {
                    self.dx / denom
                }
            },
        })
    }
}

impl<V, D, B> Field for F<V, D>
where
    B: SimdBool,
    V: Clone + Field<SimdBool = B>,
    D: Clone
        + Field<SimdBool = B>
        + std::fmt::Debug
        + Zero
        + Neg<Output = D>
        + Mul<Output = D>
        + Mul<V, Output = D>
        + Add<Output = D>
        + DivAssign<V>
        + MulAssign<V>
        + Div<V, Output = D>
        + Div<Output = D>
        + Sub<Output = D>,
{
}

//impl<V, D, B> Field for F<V, D>
//where
//    B: SimdBool,
//    V: Clone + Field<SimdBool = B> + RemAssign + Div<Output = V> + Float,
//    D: Clone
//        + Field<SimdBool = B>
//        + Mul<V, Output = D>
//        + SubAssign
//        + std::ops::DivAssign<V>
//        + std::ops::MulAssign<V>
//        + std::fmt::Debug
//        + Div<V>
//        + Div<V, Output = D>,
//{
//}

impl<V, D, B> SimdValue for F<V, D>
where
    B: SimdBool,
    V: SimdValue<SimdBool = B>,
    D: SimdValue<SimdBool = B>,
{
    type Element = F<V::Element, D::Element>;
    type SimdBool = B;

    #[inline(always)]
    fn lanes() -> usize {
        V::lanes()
    }

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        F {
            x: V::splat(val.x),
            dx: D::splat(val.dx),
        }
    }

    #[inline(always)]
    fn extract(&self, i: usize) -> Self::Element {
        F {
            x: self.x.extract(i),
            dx: self.dx.extract(i),
        }
    }

    #[inline(always)]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        F {
            x: self.x.extract_unchecked(i),
            dx: self.dx.extract_unchecked(i),
        }
    }

    #[inline(always)]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.x.replace(i, val.x);
        self.dx.replace(i, val.dx);
    }

    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.x.replace_unchecked(i, val.x);
        self.dx.replace_unchecked(i, val.dx);
    }

    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        F {
            x: self.x.select(cond, other.x),
            dx: self.dx.select(cond, other.dx),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{F, F1};
    use nalgebra::{ComplexField, Matrix3, Vector3};
    fn make_mtx() -> Matrix3<F<f64, f64>> {
        [
            [F::cst(1.0), F::cst(2.0), F::cst(3.0)],
            [F::cst(4.0), F::cst(5.0), F::cst(6.0)],
            [F::cst(7.0), F::cst(8.0), F::cst(9.0)],
        ]
        .into()
    }

    // Generic multiply. This tests that F is a realfield
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
                F { x: 12.0, dx: 12.0 },
                F { x: 15.0, dx: 15.0 },
                F { x: 18.0, dx: 18.0 }
            ])
        );
    }
}
