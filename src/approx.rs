use super::AutoFloat;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::Zero;

impl<T, const N: usize> AbsDiffEq for AutoFloat<T, N>
where
    T: PartialEq + AbsDiffEq<Epsilon = T> + Zero + Clone,
{
    type Epsilon = Self;
    fn default_epsilon() -> Self::Epsilon {
        AutoFloat::constant(T::default_epsilon())
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon.x)
    }
}
impl<T, const N: usize> RelativeEq for AutoFloat<T, N>
where
    T: PartialEq + RelativeEq<Epsilon = T> + Zero + Clone,
{
    fn default_max_relative() -> Self::Epsilon {
        AutoFloat::constant(T::default_max_relative())
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.x.relative_eq(&other.x, epsilon.x, max_relative.x)
    }
}
impl<T, const N: usize> UlpsEq for AutoFloat<T, N>
where
    T: PartialEq + UlpsEq<Epsilon = T> + Zero + Clone,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.x.ulps_eq(&other.x, epsilon.x, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use crate::AutoFloat1;

    use super::*;

    #[test]
    fn abs_diff_eq() {
        let a = AutoFloat1::constant(1.0);
        let b = AutoFloat1::constant(1.0);
        assert!(a.abs_diff_eq(&b, AutoFloat1::constant(1e-4)));
    }

    #[test]
    fn default_max_ulps() {
        assert_eq!(
            AutoFloat::<f32, 3>::default_max_ulps(),
            f32::default_max_ulps()
        );
    }

    #[test]
    fn ulps_eq() {
        let v1 = AutoFloat::<f32, 2>::new(1.0, [1.0, 2.0]);
        let v2 = AutoFloat::<f32, 2>::new(2.0, [4.0, 1.0]);
        let ulps = f32::default_max_ulps();
        assert_eq!(
            v1.ulps_eq(&v2, AutoFloat::default_epsilon(), ulps),
            v1.x.ulps_eq(&v2.x, f32::default_epsilon(), ulps)
        );
    }
}
