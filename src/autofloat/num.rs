use num_traits::{Bounded, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};

use crate::AutoFloat;

impl<T, const N: usize> ToPrimitive for AutoFloat<T, N>
where
    T: ToPrimitive,
{
    fn to_i64(&self) -> Option<i64> {
        self.x.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.x.to_u64()
    }

    fn to_isize(&self) -> Option<isize> {
        self.x.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self.x.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.x.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.x.to_i32()
    }

    fn to_usize(&self) -> Option<usize> {
        self.x.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self.x.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self.x.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self.x.to_u32()
    }

    fn to_f32(&self) -> Option<f32> {
        self.x.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.x.to_f64()
    }
}

impl<T, const N: usize> FromPrimitive for AutoFloat<T, N>
where
    T: FromPrimitive + Copy + Zero,
{
    fn from_isize(n: isize) -> Option<Self> {
        T::from_isize(n).map(AutoFloat::constant)
    }

    fn from_i8(n: i8) -> Option<Self> {
        T::from_i8(n).map(AutoFloat::constant)
    }

    fn from_i16(n: i16) -> Option<Self> {
        T::from_i16(n).map(AutoFloat::constant)
    }

    fn from_i32(n: i32) -> Option<Self> {
        T::from_i32(n).map(AutoFloat::constant)
    }

    fn from_i64(n: i64) -> Option<Self> {
        T::from_i64(n).map(AutoFloat::constant)
    }

    fn from_i128(n: i128) -> Option<Self> {
        T::from_i128(n).map(AutoFloat::constant)
    }

    fn from_usize(n: usize) -> Option<Self> {
        T::from_usize(n).map(AutoFloat::constant)
    }

    fn from_u8(n: u8) -> Option<Self> {
        T::from_u8(n).map(AutoFloat::constant)
    }

    fn from_u16(n: u16) -> Option<Self> {
        T::from_u16(n).map(AutoFloat::constant)
    }

    fn from_u32(n: u32) -> Option<Self> {
        T::from_u32(n).map(AutoFloat::constant)
    }

    fn from_u64(n: u64) -> Option<Self> {
        T::from_u64(n).map(AutoFloat::constant)
    }

    fn from_u128(n: u128) -> Option<Self> {
        T::from_u128(n).map(AutoFloat::constant)
    }

    fn from_f32(n: f32) -> Option<Self> {
        T::from_f32(n).map(AutoFloat::constant)
    }

    fn from_f64(n: f64) -> Option<Self> {
        T::from_f64(n).map(AutoFloat::constant)
    }
}

impl<T, const N: usize> NumCast for AutoFloat<T, N>
where
    T: NumCast + Zero + Copy,
{
    fn from<U: ToPrimitive>(n: U) -> Option<Self> {
        T::from(n).map(AutoFloat::constant)
    }
}

impl<T, const N: usize> Zero for AutoFloat<T, N>
where
    T: Zero + Copy + Default,
{
    fn zero() -> Self {
        AutoFloat::constant(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.x.is_zero()
    }
}

impl<T, const N: usize> One for AutoFloat<T, N>
where
    T: Copy + Default + Zero + One,
{
    fn one() -> Self {
        AutoFloat::constant(T::one())
    }
}

impl<T, const N: usize> Num for AutoFloat<T, N>
where
    T: Copy + Default + Zero + Num,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(src, radix).map(AutoFloat::constant)
    }
}

impl<T, const N: usize> Signed for AutoFloat<T, N>
where
    T: Signed + Zero + Copy + Default + Num + PartialOrd,
{
    fn abs(&self) -> Self {
        if self.is_negative() {
            -self.clone()
        } else {
            self.clone()
        }
    }

    fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other {
            Self::zero()
        } else {
            self.clone() - other.clone()
        }
    }

    fn signum(&self) -> Self {
        Self::constant(self.x.signum())
    }

    fn is_positive(&self) -> bool {
        self.x.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.x.is_negative()
    }
}

impl<T, const N: usize> Bounded for AutoFloat<T, N>
where
    T: Bounded + Zero + Copy,
{
    fn min_value() -> Self {
        AutoFloat::constant(T::min_value())
    }
    fn max_value() -> Self {
        AutoFloat::constant(T::max_value())
    }
}

#[cfg(test)]
mod test {
    use core::f32;

    use crate::autofloat::test::assert_autofloat_eq;

    use super::*;

    #[test]
    fn to_primitive() {
        let v1 = AutoFloat::<f32, 2>::new(3.0, [1.0, 2.0]);

        assert_eq!(v1.to_i8(), Some(3));
        assert_eq!(v1.to_i16(), Some(3));
        assert_eq!(v1.to_i32(), Some(3));
        assert_eq!(v1.to_i64(), Some(3));
        assert_eq!(v1.to_i128(), Some(3));
        assert_eq!(v1.to_isize(), Some(3));

        assert_eq!(v1.to_u8(), Some(3));
        assert_eq!(v1.to_u16(), Some(3));
        assert_eq!(v1.to_u32(), Some(3));
        assert_eq!(v1.to_u64(), Some(3));
        assert_eq!(v1.to_u128(), Some(3));
        assert_eq!(v1.to_usize(), Some(3));

        assert_eq!(v1.to_f32(), Some(3.0));
        assert_eq!(v1.to_f64(), Some(3.0));
    }

    #[test]
    fn from_primitive() {
        let v1 = AutoFloat::<f32, 2>::new(3.0, [0.0, 0.0]);

        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_i8(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_i16(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_i32(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_i64(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_i128(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_isize(3).unwrap(), v1);

        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_u8(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_u16(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_u32(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_u64(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_u128(3).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_usize(3).unwrap(), v1);

        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_f32(3.0).unwrap(), v1);
        assert_autofloat_eq!(AutoFloat::<f32, 2>::from_f64(3.0).unwrap(), v1);
    }

    #[test]
    fn zero() {
        assert_autofloat_eq!(
            AutoFloat::<f32, 2>::zero(),
            AutoFloat::<f32, 2>::new(0.0, [0.0, 0.0])
        );

        let v1 = AutoFloat::<f32, 2>::new(0.0, [1.0, 2.0]);
        let v2 = AutoFloat::<f32, 2>::new(-2.0, [4.0, 3.0]);

        assert!(v1.is_zero());
        assert!(!v2.is_zero());
    }

    #[test]
    fn one() {
        assert_autofloat_eq!(
            AutoFloat::<f32, 2>::one(),
            AutoFloat::<f32, 2>::new(1.0, [0.0, 0.0])
        );

        let v1 = AutoFloat::<f32, 2>::new(1.0, [1.0, 2.0]);
        let v2 = AutoFloat::<f32, 2>::new(-2.0, [4.0, 3.0]);

        assert!(v1.is_one());
        assert!(!v2.is_one());
    }

    #[test]
    fn signed() {
        let v1 = AutoFloat::<f32, 2>::new(1.0, [1.0, 2.0]);
        let v2 = AutoFloat::<f32, 2>::new(-2.0, [4.0, 3.0]);
        let v3 = AutoFloat::<f32, 2>::new(0.0, [-1.0, 2.0]);

        assert!(!v1.is_negative());
        assert!(v2.is_negative());
        assert!(!v3.is_negative());

        assert!(v1.is_positive());
        assert!(!v2.is_positive());
        assert!(v3.is_positive());

        assert_autofloat_eq!(v1.signum(), AutoFloat::new(1.0, [0.0, 0.0]));
        assert_autofloat_eq!(v2.signum(), AutoFloat::new(-1.0, [0.0, 0.0]));
        assert_autofloat_eq!(v3.signum(), AutoFloat::new(1.0, [0.0, 0.0]));

        assert_autofloat_eq!(v1.abs(), AutoFloat::new(1.0, [1.0, 2.0]));
        assert_autofloat_eq!(v2.abs(), AutoFloat::new(2.0, [-4.0, -3.0]));
        assert_autofloat_eq!(v3.abs(), AutoFloat::new(0.0, [-1.0, 2.0]));
    }

    #[test]
    fn bounded() {
        assert_autofloat_eq!(
            AutoFloat::<f32, 2>::min_value(),
            AutoFloat::new(f32::MIN, [0.0, 0.0])
        );
        assert_autofloat_eq!(
            AutoFloat::<f32, 2>::max_value(),
            AutoFloat::new(f32::MAX, [0.0, 0.0])
        );
    }
}
