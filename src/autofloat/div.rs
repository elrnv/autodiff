use std::ops::{Div, DivAssign, Mul, Sub, SubAssign};

use num_traits::One;

use crate::{
    autofloat::{binary_op, unary_op},
    AutoFloat,
};

impl<T, const N: usize> Div<AutoFloat<T, N>> for AutoFloat<T, N>
where
    T: Div<Output = T> + Mul<Output = T> + Sub<Output = T> + One + Copy + Default,
{
    type Output = Self;

    fn div(self, rhs: AutoFloat<T, N>) -> Self::Output {
        let factor = T::one() / (rhs.x.clone() * rhs.x.clone());
        AutoFloat {
            x: self.x.clone() / rhs.x.clone(),
            dx: binary_op(self.dx, rhs.dx, |l, r| {
                (l * rhs.x.clone() - r * self.x.clone()) * factor.clone()
            }),
        }
    }
}

impl<T, const N: usize> Div<T> for AutoFloat<T, N>
where
    T: Div<Output = T> + Copy + Default,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        AutoFloat {
            x: self.x / rhs,
            dx: unary_op(self.dx, |v| v / rhs),
        }
    }
}

impl<const N: usize> Div<AutoFloat<f64, N>> for f64 {
    type Output = AutoFloat<f64, N>;

    fn div(self, rhs: AutoFloat<f64, N>) -> Self::Output {
        let denom_inv = 1.0 / (rhs.x * rhs.x);
        let negated = -self;
        AutoFloat {
            x: self / rhs.x,
            dx: unary_op(rhs.dx, |v| negated * v * denom_inv),
        }
    }
}

impl<const N: usize> Div<AutoFloat<f32, N>> for f32 {
    type Output = AutoFloat<f32, N>;

    fn div(self, rhs: AutoFloat<f32, N>) -> Self::Output {
        let denom_inv = 1.0 / (rhs.x * rhs.x);
        let negated = -self;
        AutoFloat {
            x: self / rhs.x,
            dx: unary_op(rhs.dx, |v| negated * v * denom_inv),
        }
    }
}

impl<T, const N: usize> DivAssign for AutoFloat<T, N>
where
    T: DivAssign + SubAssign + Mul<Output = T> + Div<Output = T> + Clone,
{
    fn div_assign(&mut self, rhs: AutoFloat<T, N>) {
        self.dx.iter_mut().for_each(|v| (*v) /= rhs.x.clone());
        let factor = self.x.clone() / (rhs.x.clone() * rhs.x.clone());
        self.dx
            .iter_mut()
            .zip(rhs.dx.into_iter())
            .for_each(|(v, u)| (*v) -= u * factor.clone());
        self.x /= rhs.x;
    }
}

impl<T, const N: usize> DivAssign<T> for AutoFloat<T, N>
where
    T: DivAssign + Clone,
{
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs.clone();
        self.dx.iter_mut().for_each(|v| (*v) /= rhs.clone());
    }
}

#[cfg(test)]
mod test {

    use crate::test::assert_autofloat_eq;

    use super::*;

    #[test]
    fn div_autofloats() {
        let v1 = AutoFloat::new(3.0, [1.0, 3.0]);
        let v2 = AutoFloat::new(2.0, [-2.0, 1.0]);
        let r1 = v1 / v2;

        assert_autofloat_eq!(AutoFloat::new(1.5, [2.0, 0.75]), r1);

        let mut r2 = v1;
        r2 /= v2;

        assert_autofloat_eq!(r1, r2);
    }

    #[test]
    fn div_autofloat_f32() {
        let v1 = AutoFloat::<f32, 2>::new(2.0, [1.0, 3.0]);
        let c1: f32 = 4.0;

        let r1 = v1 / c1;
        assert_autofloat_eq!(AutoFloat::new(0.5, [0.25, 0.75]), r1);

        let r2 = c1 / v1;
        assert_autofloat_eq!(AutoFloat::new(2.0, [-1.0, -3.0]), r2);

        let mut r3 = v1;
        r3 /= c1;
        assert_autofloat_eq!(r1, r3);
    }

    #[test]
    fn div_autofloat_f64() {
        let v1 = AutoFloat::<f64, 2>::new(2.0, [1.0, 3.0]);
        let c1: f64 = 4.0;

        let r1 = v1 / c1;
        assert_autofloat_eq!(AutoFloat::new(0.5, [0.25, 0.75]), r1);

        let r2 = c1 / v1;
        assert_autofloat_eq!(AutoFloat::new(2.0, [-1.0, -3.0]), r2);

        let mut r3 = v1;
        r3 /= c1;
        assert_autofloat_eq!(r1, r3);
    }
}
