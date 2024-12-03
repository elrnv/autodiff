use std::ops::{Add, AddAssign, Mul, MulAssign};

use crate::{
    autofloat::{binary_op, unary_op},
    AutoFloat,
};

impl<T, const N: usize> Mul<AutoFloat<T, N>> for AutoFloat<T, N>
where
    T: Mul<Output = T> + Add<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: AutoFloat<T, N>) -> Self::Output {
        AutoFloat {
            x: self.x.clone() * rhs.x.clone(),
            dx: binary_op(self.dx, rhs.dx, |l, r| {
                l * rhs.x.clone() + r * self.x.clone()
            }),
        }
    }
}

impl<const N: usize> Mul<AutoFloat<f64, N>> for f64 {
    type Output = AutoFloat<f64, N>;

    fn mul(self, rhs: AutoFloat<f64, N>) -> Self::Output {
        AutoFloat {
            x: self * rhs.x,
            dx: unary_op(rhs.dx, |v| self * v),
        }
    }
}

impl<const N: usize> Mul<AutoFloat<f32, N>> for f32 {
    type Output = AutoFloat<f32, N>;

    fn mul(self, rhs: AutoFloat<f32, N>) -> Self::Output {
        AutoFloat {
            x: self * rhs.x,
            dx: unary_op(rhs.dx, |v| self * v),
        }
    }
}

impl<T, const N: usize> Mul<T> for AutoFloat<T, N>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        AutoFloat {
            x: self.x * rhs.clone(),
            dx: unary_op(self.dx, |v| v * rhs.clone()),
        }
    }
}

impl<T, const N: usize> MulAssign for AutoFloat<T, N>
where
    T: Clone + MulAssign + AddAssign + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: AutoFloat<T, N>) {
        self.dx.iter_mut().for_each(|v| (*v) *= rhs.x.clone());
        self.dx
            .iter_mut()
            .zip(rhs.dx.into_iter())
            .for_each(|(v, u)| (*v) += u * self.x.clone());

        self.x *= rhs.x;
    }
}

impl<T, const N: usize> MulAssign<T> for AutoFloat<T, N>
where
    T: MulAssign + Clone,
{
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs.clone();
        self.dx.iter_mut().for_each(|v| (*v) *= rhs.clone());
    }
}

#[cfg(test)]
mod test {

    use crate::test::assert_autofloat_eq;

    use super::*;

    #[test]
    fn mul_autofloats() {
        let v1 = AutoFloat::new(2.0, [1.0, 3.0]);
        let v2 = AutoFloat::new(3.0, [-2.0, 1.0]);
        let r1 = v1 * v2;

        assert_autofloat_eq!(AutoFloat::new(6.0, [-1.0, 11.0]), r1);

        let mut r2 = v1;
        r2 *= v2;

        assert_autofloat_eq!(r1, r2);
    }

    #[test]
    fn mul_autofloat_f32() {
        let v1 = AutoFloat::<f32, 2>::new(2.0, [1.0, 3.0]);
        let c1: f32 = 4.0;

        let r1 = v1 * c1;
        assert_autofloat_eq!(AutoFloat::new(8.0, [4.0, 12.0]), r1);

        let r2 = c1 * v1;
        assert_autofloat_eq!(r1, r2);

        let mut r3 = v1;
        r3 *= c1;
        assert_autofloat_eq!(r1, r3);
    }

    #[test]
    fn mul_autofloat_f64() {
        let v1 = AutoFloat::<f64, 2>::new(2.0, [1.0, 3.0]);
        let c1: f64 = 4.0;

        let r1 = v1 * c1;
        assert_autofloat_eq!(AutoFloat::new(8.0, [4.0, 12.0]), r1);

        let r2 = c1 * v1;
        assert_autofloat_eq!(r1, r2);

        let mut r3 = v1;
        r3 *= c1;
        assert_autofloat_eq!(r1, r3);
    }
}
