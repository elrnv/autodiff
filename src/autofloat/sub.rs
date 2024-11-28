use std::ops::{Sub, SubAssign};

use super::{binary_op, unary_op, AutoFloat};

impl<T, const N: usize> Sub<AutoFloat<T, N>> for AutoFloat<T, N>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Self;

    fn sub(self, rhs: AutoFloat<T, N>) -> Self::Output {
        AutoFloat {
            x: self.x - rhs.x,
            dx: binary_op(self.dx, rhs.dx, |l, r| l - r),
        }
    }
}

impl<T, const N: usize> Sub<T> for AutoFloat<T, N>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        AutoFloat {
            x: self.x - rhs,
            dx: self.dx,
        }
    }
}

impl<const N: usize> Sub<AutoFloat<f64, N>> for f64 {
    type Output = AutoFloat<f64, N>;

    fn sub(self, rhs: AutoFloat<f64, N>) -> Self::Output {
        AutoFloat {
            x: self - rhs.x,
            dx: unary_op(rhs.dx, |v| -v),
        }
    }
}

impl<const N: usize> Sub<AutoFloat<f32, N>> for f32 {
    type Output = AutoFloat<f32, N>;

    fn sub(self, rhs: AutoFloat<f32, N>) -> Self::Output {
        AutoFloat {
            x: self - rhs.x,
            dx: unary_op(rhs.dx, |v| -v),
        }
    }
}

impl<T, const N: usize> SubAssign for AutoFloat<T, N>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, rhs: AutoFloat<T, N>) {
        self.x -= rhs.x;
        for (dst, src) in self.dx.iter_mut().zip(rhs.dx.into_iter()) {
            *dst -= src;
        }
    }
}

impl<T, const N: usize> SubAssign<T> for AutoFloat<T, N>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, rhs: T) {
        self.x -= rhs;
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn sub_autofloats() {
        let v1 = AutoFloat::new(2.0, [1.0, 3.0]);
        let v2 = AutoFloat::new(3.0, [-2.0, 1.0]);
        let r1 = v1 - v2;

        assert_eq!(-1.0, r1.x);
        assert_eq!([3.0, 2.0], r1.dx);

        let mut r2 = v1;
        r2 -= v2;

        assert_eq!(r1.x, r2.x);
        assert_eq!(r1.dx, r2.dx);
    }

    #[test]
    fn sub_autofloat_f32() {
        let v1 = AutoFloat::<f32, 2>::new(2.0, [1.0, 3.0]);
        let c1: f32 = 4.0;

        let r1 = v1 - c1;
        assert_eq!(-2.0, r1.x);
        assert_eq!([1.0, 3.0], r1.dx);

        let r2 = c1 - v1;
        assert_eq!(2.0, r2.x);
        assert_eq!([-1.0, -3.0], r2.dx);

        let mut r3 = v1;
        r3 -= c1;
        assert_eq!(r1.x, r3.x);
        assert_eq!(r1.dx, r3.dx);
    }

    #[test]
    fn sub_autofloat_f64() {
        let v1 = AutoFloat::<f64, 2>::new(2.0, [1.0, 3.0]);
        let c1: f64 = 4.0;

        let r1 = v1 - c1;
        assert_eq!(-2.0, r1.x);
        assert_eq!([1.0, 3.0], r1.dx);

        let r2 = c1 - v1;
        assert_eq!(2.0, r2.x);
        assert_eq!([-1.0, -3.0], r2.dx);

        let mut r3 = v1;
        r3 -= c1;
        assert_eq!(r1.x, r3.x);
        assert_eq!(r1.dx, r3.dx);
    }
}
