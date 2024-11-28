use num_traits::{Bounded, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};

use super::AutoFloat;

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
    T: Zero + Copy,
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
    T: Copy + Zero + One,
{
    fn one() -> Self {
        AutoFloat::constant(T::one())
    }
}

impl<T, const N: usize> Num for AutoFloat<T, N>
where
    T: Copy + Zero + Num,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(src, radix).map(AutoFloat::constant)
    }
}

impl<T, const N: usize> Signed for AutoFloat<T, N>
where
    T: Signed + Zero + Copy + Num + PartialOrd,
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
