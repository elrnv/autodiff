use num_traits::Zero;
use std::{
    iter::Sum,
    ops::{Add, AddAssign},
};

use crate::{autofloat::binary_op, AutoFloat};

impl<T, const N: usize> Add<AutoFloat<T, N>> for AutoFloat<T, N>
where
    T: Add<Output = T> + Copy + Default,
{
    type Output = Self;

    fn add(self, rhs: AutoFloat<T, N>) -> Self::Output {
        AutoFloat {
            x: self.x + rhs.x,
            dx: binary_op(self.dx, rhs.dx, |l, r| l + r),
        }
    }
}

impl<T, const N: usize> Add<T> for AutoFloat<T, N>
where
    T: Add<Output = T>,
{
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        AutoFloat {
            x: rhs + self.x,
            dx: self.dx,
        }
    }
}

impl<const N: usize> Add<AutoFloat<f64, N>> for f64 {
    type Output = AutoFloat<f64, N>;

    fn add(self, rhs: AutoFloat<f64, N>) -> Self::Output {
        AutoFloat {
            x: self + rhs.x,
            dx: rhs.dx,
        }
    }
}

impl<const N: usize> Add<AutoFloat<f32, N>> for f32 {
    type Output = AutoFloat<f32, N>;

    fn add(self, rhs: AutoFloat<f32, N>) -> Self::Output {
        AutoFloat {
            x: self + rhs.x,
            dx: rhs.dx,
        }
    }
}

impl<T, const N: usize> AddAssign for AutoFloat<T, N>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: AutoFloat<T, N>) {
        self.x += rhs.x;
        for (dst, src) in self.dx.iter_mut().zip(rhs.dx.into_iter()) {
            *dst += src;
        }
    }
}

impl<T, const N: usize> AddAssign<T> for AutoFloat<T, N>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: T) {
        self.x += rhs;
    }
}

impl<T, const N: usize> Sum for AutoFloat<T, N>
where
    T: AddAssign + Zero + Copy + Default,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut res = Self::zero();
        for x in iter {
            res += x;
        }
        res
    }
}

impl<T, const N: usize> std::iter::Sum<T> for AutoFloat<T, N>
where
    T: AddAssign + Copy + Default + Zero,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        iter.map(AutoFloat::constant).sum()
    }
}

#[cfg(test)]
mod test {

    use crate::test::assert_autofloat_eq;

    use super::*;

    #[test]
    fn add_autofloats() {
        let v1 = AutoFloat::new(2.0, [1.0, 3.0]);
        let v2 = AutoFloat::new(3.0, [-2.0, 1.0]);
        let r1 = v1 + v2;

        assert_autofloat_eq!(AutoFloat::new(5.0, [-1.0, 4.0]), r1);

        let mut r2 = v1;
        r2 += v2;

        assert_autofloat_eq!(r1, r2);
    }

    #[test]
    fn add_autofloat_f32() {
        let v1 = AutoFloat::<f32, 2>::new(2.0, [1.0, 3.0]);
        let c1: f32 = 4.0;

        let r1 = v1 + c1;
        assert_autofloat_eq!(AutoFloat::new(6.0, [1.0, 3.0]), r1);

        let r2 = c1 + v1;
        assert_autofloat_eq!(r1, r2);

        let mut r3 = v1;
        r3 += c1;
        assert_autofloat_eq!(r1, r3);
    }

    #[test]
    fn add_autofloat_f64() {
        let v1 = AutoFloat::<f64, 2>::new(2.0, [1.0, 3.0]);
        let c1: f64 = 4.0;

        let r1 = v1 + c1;
        assert_autofloat_eq!(AutoFloat::new(6.0, [1.0, 3.0]), r1);

        let r2 = c1 + v1;
        assert_autofloat_eq!(r1, r2);

        let mut r3 = v1;
        r3 += c1;
        assert_autofloat_eq!(r1, r3);
    }

    #[test]
    fn sum_autofloats() {
        let v1 = AutoFloat::new(2.0, [1.0, 3.0]);
        let v2 = AutoFloat::new(3.0, [-2.0, 1.0]);
        let v3 = AutoFloat::new(4.0, [3.0, -10.0]);
        let vals = [v1, v2, v3];

        let r: AutoFloat<f32, 2> = vals.into_iter().sum();

        assert_autofloat_eq!(AutoFloat::new(9.0, [2.0, -6.0]), r);
    }

    #[test]
    fn sum_constants() {
        let vals = [1.0, 4.0, 3.0];

        let r: AutoFloat<f32, 2> = vals.into_iter().sum();

        assert_autofloat_eq!(AutoFloat::new(8.0, [0.0, 0.0]), r);
    }
}
