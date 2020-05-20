use super::F;
use cgmath::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::Zero;

impl<D: PartialEq + Zero> AbsDiffEq for F<D> {
    type Epsilon = Self;
    fn default_epsilon() -> Self::Epsilon {
        F::cst(f64::default_epsilon())
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon.x)
    }
}
impl<D: PartialEq + Zero> RelativeEq for F<D> {
    fn default_max_relative() -> Self::Epsilon {
        F::cst(f64::default_max_relative())
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
impl<D: PartialEq + Zero> UlpsEq for F<D> {
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.x.ulps_eq(&other.x, epsilon.x, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use crate::{F, F1};
    use cgmath::{Matrix3, Vector3};
    #[test]
    fn mtx_mul() {
        let data = [
            [F::cst(1.0), F::cst(2.0), F::cst(3.0)],
            [F::cst(4.0), F::cst(5.0), F::cst(6.0)],
            [F::cst(7.0), F::cst(8.0), F::cst(9.0)],
        ];
        let mtx = Matrix3::from(data);
        let v = Vector3::from([F1::var(1.0); 3]);
        assert_eq!(
            mtx * v,
            Vector3::from([
                F { x: 12.0, dx: 12.0 },
                F { x: 15.0, dx: 15.0 },
                F { x: 18.0, dx: 18.0 }
            ])
        );
    }
}
