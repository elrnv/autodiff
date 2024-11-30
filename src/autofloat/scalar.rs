use num_traits::{One, ToPrimitive, Zero};
use std::{fmt::Display, ops::Neg};

use crate::unary_op;

/// A generic forward differentiation `Dual` number.
#[derive(Debug, Copy, Clone)]
pub struct AutoFloat<T, const N: usize> {
    /// The value of the variable.
    pub x: T,
    /// The gradient of the variable.
    pub dx: [T; N],
}

pub type AutoFloat1<T> = AutoFloat<T, 1>;
pub type AutoFloat2<T> = AutoFloat<T, 2>;
pub type AutoFloat3<T> = AutoFloat<T, 3>;
pub type AutoFloat4<T> = AutoFloat<T, 4>;
pub type AutoFloat5<T> = AutoFloat<T, 5>;
pub type AutoFloat6<T> = AutoFloat<T, 6>;

impl<T, const N: usize> AutoFloat<T, N> {
    /// Creates an `AutoFloat` with the given value and gradients.
    pub fn new(x: T, dx: [T; N]) -> Self {
        Self { x, dx }
    }
}

impl<T, const N: usize> Display for AutoFloat<T, N> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<T, const N: usize> AutoFloat<T, N>
where
    T: Zero + One + Copy,
{
    /// Creates an `AutoFloat` as variable where the given dimension is set to one.
    ///
    /// # Panics
    ///
    /// Panics if the given dimension is greater than the gradient dimension.
    pub fn variable(x: T, dim: usize) -> Self {
        let mut dx = [T::zero(); N];
        dx[dim] = T::one();
        Self { x, dx }
    }
}

impl<T, const N: usize> AutoFloat<T, N>
where
    T: Zero + Copy,
{
    /// Creates an `AutoFloat` as constant with the given value and zero gradients.
    pub fn constant(x: T) -> Self {
        Self {
            x,
            dx: [T::zero(); N],
        }
    }
}

impl<T, const N: usize> Default for AutoFloat<T, N>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Self::new(T::default(), [T::default(); N])
    }
}

/// Implement partial equality in terms of the scalar value `x``.
impl<T: PartialEq, const N: usize> PartialEq<AutoFloat<T, N>> for AutoFloat<T, N> {
    fn eq(&self, rhs: &AutoFloat<T, N>) -> bool {
        self.x == rhs.x
    }
}

/// Implement partial order in terms of the scalar value `x`
impl<T: PartialOrd, const N: usize> PartialOrd<AutoFloat<T, N>> for AutoFloat<T, N> {
    fn partial_cmp(&self, other: &AutoFloat<T, N>) -> Option<::std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.x, &other.x)
    }
}

impl<T, const N: usize> From<T> for AutoFloat<T, N>
where
    T: Zero + Copy,
{
    fn from(x: T) -> Self {
        AutoFloat::constant(x)
    }
}

impl<T, const N: usize> Into<f64> for AutoFloat<T, N>
where
    T: ToPrimitive,
{
    /// Converts the `AutoFloat` into an `f64`.
    ///
    /// # Panics
    ///
    /// This function panics if this conversion fails.
    fn into(self) -> f64 {
        self.x.to_f64().unwrap()
    }
}

impl<T, const N: usize> Into<f32> for AutoFloat<T, N>
where
    T: ToPrimitive,
{
    /// Converts the `AutoFloat` into an `f32`.
    ///
    /// # Panics
    ///
    /// This function panics if this conversion fails.
    fn into(self) -> f32 {
        self.x.to_f32().unwrap()
    }
}

impl<T, const N: usize> Neg for AutoFloat<T, N>
where
    T: Neg<Output = T> + Copy + Default,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        AutoFloat {
            x: -self.x,
            dx: unary_op(self.dx, |v| -v),
        }
    }
}

#[cfg(test)]
mod test {

    use crate::test::assert_autofloat_eq;

    use super::*;

    #[test]
    fn create_constant() {
        let c1 = AutoFloat::<f32, 4>::constant(2.5);
        assert_eq!(2.5, c1.x);
        assert!(c1.dx.iter().all(|&v| v.is_zero()));

        let c2 = AutoFloat::<f64, 6>::constant(-3.2);
        assert_eq!(-3.2, c2.x);
        assert!(c2.dx.iter().all(|&v| v.is_zero()));
    }

    #[test]
    fn create_variable() {
        let v1 = AutoFloat::<f32, 4>::variable(2.5, 2);
        assert_autofloat_eq!(AutoFloat::new(2.5, [0.0, 0.0, 1.0, 0.0]), v1);

        let v2 = AutoFloat::<f64, 6>::variable(-3.2, 0);
        assert_autofloat_eq!(AutoFloat::new(-3.2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],), v2);
    }

    #[test]
    fn create_default() {
        let v1 = AutoFloat::<f32, 4>::default();
        assert_eq!(0.0, v1.x);
        assert!(v1.dx.iter().all(|v| v.is_zero()));

        let v2 = AutoFloat::<f64, 6>::default();
        assert_eq!(0.0, v2.x);
        assert!(v2.dx.iter().all(|v| v.is_zero()));
    }

    #[test]
    fn partial_equality() {
        let val1 = AutoFloat::<f32, 3>::new(1.0, [2.0, 3.0, 4.0]);
        let val2 = AutoFloat::<f32, 3>::new(1.0, [-2.0, -3.0, -4.0]);
        let val3 = AutoFloat::<f32, 3>::new(-1.0, [2.0, 3.0, 4.0]);

        // equal although gradients are not equal
        assert_eq!(val1, val2);

        // not equal although gradients are equal
        assert_ne!(val1, val3);
    }

    #[test]
    fn partial_order() {
        let val1 = AutoFloat::<f32, 3>::new(1.0, [2.0, 3.0, 4.0]);
        let val2 = AutoFloat::<f32, 3>::new(2.0, [-2.0, -3.0, -4.0]);
        let val3 = AutoFloat::<f32, 3>::new(-1.0, [-2.0, -3.0, -4.0]);
        assert!(val1 < val2);
        assert!(val1 > val3);
    }

    #[test]
    fn into_f32() {
        let val = AutoFloat::<f32, 3>::new(1.0, [2.0, 3.0, 4.0]);
        assert_eq!(1.0f32, val.into());

        let val = AutoFloat::<f64, 3>::new(1.0, [2.0, 3.0, 4.0]);
        assert_eq!(1.0f32, val.into());
    }

    #[test]
    fn into_f64() {
        let val = AutoFloat::<f32, 3>::new(1.0, [2.0, 3.0, 4.0]);
        assert_eq!(1.0f64, val.into());

        let val = AutoFloat::<f64, 3>::new(1.0, [2.0, 3.0, 4.0]);
        assert_eq!(1.0f64, val.into());
    }

    #[test]
    fn negate() {
        let v1 = AutoFloat::<f32, 3>::new(1.0, [2.0, 3.0, 4.0]);
        let v1_neg = -v1;
        assert_eq!(-1.0, v1_neg.x);
        assert_eq!([-2.0, -3.0, -4.0], v1_neg.dx);

        let v2 = AutoFloat::<f64, 3>::new(-3.0, [1.0, -2.0, -8.0]);
        let v2_neg = -v2;
        assert_eq!(3.0, v2_neg.x);
        assert_eq!([-1.0, 2.0, 8.0], v2_neg.dx);
    }
}
