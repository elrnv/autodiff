use std::ops::{Div, Rem, RemAssign, Sub, SubAssign};

use num_traits::One;

use crate::{binary_op, unary_op, AutoFloat};

impl<T, const N: usize> Rem<AutoFloat<T, N>> for AutoFloat<T, N>
where
    T: Copy + Default + Rem<Output = T> + Div<Output = T> + Sub<Output = T> + One,
{
    type Output = Self;

    fn rem(self, rhs: AutoFloat<T, N>) -> Self::Output {
        // This is an approximation. There are places where the derivative doesn't exist.
        let div = self.x.clone() / rhs.x.clone();
        let factor = div.clone() - div % T::one();
        AutoFloat {
            x: self.x % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: binary_op(self.dx, rhs.dx, |l, r| l - r * factor.clone()),
        }
    }
}

impl<T, const N: usize> Rem<T> for AutoFloat<T, N>
where
    T: Rem<Output = T> + Copy + Default,
{
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        AutoFloat {
            x: self.x % rhs,
            dx: unary_op(self.dx, |v| v % rhs),
        }
    }
}

impl<const N: usize> Rem<AutoFloat<f64, N>> for f64 {
    type Output = AutoFloat<f64, N>;

    fn rem(self, rhs: AutoFloat<f64, N>) -> Self::Output {
        // This is an approximation. There are places where the derivative doesn't exist.
        AutoFloat {
            x: self % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: unary_op(rhs.dx, |v| -(self / rhs.x).trunc() * v),
        }
    }
}

impl<const N: usize> Rem<AutoFloat<f32, N>> for f32 {
    type Output = AutoFloat<f32, N>;

    fn rem(self, rhs: AutoFloat<f32, N>) -> Self::Output {
        // This is an approximation. There are places where the derivative doesn't exist.
        AutoFloat {
            x: self % rhs.x, // x % y = x - [x/|y|]*|y|
            dx: unary_op(rhs.dx, |v| -(self / rhs.x).trunc() * v),
        }
    }
}

impl<T, const N: usize> RemAssign for AutoFloat<T, N>
where
    T: RemAssign + SubAssign + Div<Output = T> + Sub<Output = T> + Rem<Output = T> + One + Clone,
{
    fn rem_assign(&mut self, rhs: AutoFloat<T, N>) {
        // x % y = x - [x/|y|]*|y|
        let div = self.x.clone() / rhs.x.clone();
        let factor = div.clone() - (div % T::one());
        self.dx
            .iter_mut()
            .zip(rhs.dx.into_iter())
            .for_each(|(v, u)| (*v) -= u * factor.clone());
        self.x %= rhs.x;
    }
}

impl<T, const N: usize> RemAssign<f64> for AutoFloat<T, N>
where
    T: RemAssign<f64>,
{
    fn rem_assign(&mut self, rhs: f64) {
        self.x %= rhs;
        self.dx.iter_mut().for_each(|v| (*v) %= rhs);
    }
}

impl<T, const N: usize> RemAssign<f32> for AutoFloat<T, N>
where
    T: RemAssign<f32>,
{
    fn rem_assign(&mut self, rhs: f32) {
        self.x %= rhs;
        self.dx.iter_mut().for_each(|v| (*v) %= rhs);
    }
}

#[cfg(test)]
mod test {

    use crate::test::assert_autofloat_eq;

    use super::*;

    #[test]
    fn rem_autofloats() {
        let v1 = AutoFloat::new(3.0, [1.0, 3.0]);
        let v2 = AutoFloat::new(2.0, [-2.0, 1.0]);
        let r1 = v1 % v2;

        assert_autofloat_eq!(AutoFloat::new(1.0, [3.0, 2.0]), r1);

        let mut r2 = v1;
        r2 %= v2;

        assert_autofloat_eq!(r1, r2);
    }

    #[test]
    fn div_autofloat_f32() {
        let v1 = AutoFloat::<f32, 2>::new(2.0, [1.0, 3.0]);
        let c1: f32 = 6.0;

        let r1 = v1 % c1;
        assert_autofloat_eq!(AutoFloat::new(2.0, [1.0, 3.0]), r1);

        let r2 = c1 % v1;
        assert_autofloat_eq!(AutoFloat::new(0.0, [-3.0, -9.0]), r2);

        let mut r3 = v1;
        r3 %= c1;
        assert_autofloat_eq!(r1, r3);
    }

    #[test]
    fn div_autofloat_f64() {
        let v1 = AutoFloat::<f64, 2>::new(2.0, [1.0, 3.0]);
        let c1: f64 = 6.0;

        let r1 = v1 % c1;
        assert_autofloat_eq!(AutoFloat::new(2.0, [1.0, 3.0]), r1);

        let r2 = c1 % v1;
        assert_autofloat_eq!(AutoFloat::new(0.0, [-3.0, -9.0]), r2);

        let mut r3 = v1;
        r3 %= c1;
        assert_autofloat_eq!(r1, r3);
    }
}
