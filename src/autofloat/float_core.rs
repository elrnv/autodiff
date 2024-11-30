use std::num::FpCategory;

use num_traits::{float::FloatCore, FloatConst, Zero};

use crate::unary_op;

use super::float_impl::*;
use super::AutoFloat;

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
    T: FloatCore + Copy + Default,
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
        to_degrees_impl!(self)
    }

    fn to_radians(self) -> Self {
        to_radians_impl!(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.x.integer_decode()
    }

    fn max(self, other: AutoFloat<T, N>) -> Self {
        max_impl!(self, other)
    }

    fn min(self, other: AutoFloat<T, N>) -> Self {
        min_impl!(self, other)
    }
}

#[cfg(test)]
mod test {
    use crate::{test::assert_autofloat_eq, AutoFloat1};

    use super::*;

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
