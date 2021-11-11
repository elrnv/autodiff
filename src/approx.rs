use super::F;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::Zero;

impl<V: PartialEq + AbsDiffEq<Epsilon = V>, D: PartialEq + Zero> AbsDiffEq for F<V, D> {
    type Epsilon = Self;
    fn default_epsilon() -> Self::Epsilon {
        F::cst(V::default_epsilon())
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon.x)
    }
}
impl<V: PartialEq + RelativeEq<Epsilon = V>, D: PartialEq + Zero> RelativeEq for F<V, D> {
    fn default_max_relative() -> Self::Epsilon {
        F::cst(V::default_max_relative())
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
impl<V: PartialEq + UlpsEq<Epsilon = V>, D: PartialEq + Zero> UlpsEq for F<V, D> {
    fn default_max_ulps() -> u32 {
        V::default_max_ulps()
    }
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.x.ulps_eq(&other.x, epsilon.x, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use crate::F1;
    use approx::AbsDiffEq;

    #[test]
    fn abs_diff_eq() {
        let a = F1::cst(1.0);
        let b = F1::cst(1.0);
        assert!(a.abs_diff_eq(&b, F1::cst(1e-4)));
    }
}
