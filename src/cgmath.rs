use super::F;
use cgmath::{AbsDiffEq, RelativeEq, UlpsEq};

impl AbsDiffEq for F {
    type Epsilon = <f64 as AbsDiffEq>::Epsilon;
    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon)
    }
}
impl RelativeEq for F {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }
    
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon
    ) -> bool {
        self.x.relative_eq(&other.x, epsilon, max_relative)
    }
}
impl UlpsEq for F {
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }
    fn ulps_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_ulps: u32
    ) -> bool {
        self.x.ulps_eq(&other.x, epsilon, max_ulps)
    }
}
