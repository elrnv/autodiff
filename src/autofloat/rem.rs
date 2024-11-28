use std::ops::{Div, Rem, RemAssign, Sub, SubAssign};

use num_traits::One;

use super::{binary_op, unary_op, AutoFloat};

impl<T, const N: usize> Rem<AutoFloat<T, N>> for AutoFloat<T, N>
where
    T: Clone + Rem<Output = T> + Div<Output = T> + Sub<Output = T> + One,
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
