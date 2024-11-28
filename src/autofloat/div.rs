use std::ops::{Div, DivAssign, Mul, Sub, SubAssign};

use super::{binary_op, unary_op, AutoFloat};

impl<T, const N: usize> Div<AutoFloat<T, N>> for AutoFloat<T, N>
where
    T: Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, rhs: AutoFloat<T, N>) -> Self::Output {
        let denom = rhs.x.clone() * rhs.x.clone();
        AutoFloat {
            x: self.x.clone() / rhs.x.clone(),
            dx: binary_op(self.dx, rhs.dx, |l, r| {
                (l * rhs.x.clone() - r * self.x.clone()) / denom.clone()
            }),
        }
    }
}

impl<T, const N: usize> Div<T> for AutoFloat<T, N>
where
    T: Div<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        AutoFloat {
            x: self.x / rhs.clone(),
            dx: unary_op(self.dx, |v| v / rhs.clone()),
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
