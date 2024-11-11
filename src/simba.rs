use super::F;
use num_traits::Zero;
use simba::scalar::{SubsetOf, SupersetOf};

impl<V: SubsetOf<V>, D: SubsetOf<D>> SubsetOf<F<V, D>> for F<V, D> {
    fn to_superset(&self) -> F<V, D> {
        F {
            x: self.x.to_superset(),
            dx: self.dx.to_superset(),
        }
    }
    fn from_superset_unchecked(element: &F<V, D>) -> Self {
        F {
            x: V::from_superset_unchecked(&element.x),
            dx: D::from_superset_unchecked(&element.dx),
        }
    }
    fn is_in_subset(element: &F<V, D>) -> bool {
        V::is_in_subset(&element.x) && D::is_in_subset(&element.dx)
    }
}

impl<V: SubsetOf<V>, D: SubsetOf<D> + num_traits::Zero> SubsetOf<V> for F<V, D> {
    fn to_superset(&self) -> V {
        self.x.to_superset()
    }
    fn from_superset_unchecked(element: &V) -> Self {
        F::cst(V::from_superset_unchecked(&element))
    }
    fn is_in_subset(element: &V) -> bool {
        V::is_in_subset(element)
    }
}

impl<V, D> SupersetOf<f32> for F<V, D>
where
    V: SupersetOf<f32>,
    D: Zero,
{
    fn is_in_subset(&self) -> bool {
        self.dx.is_zero() && self.x.is_in_subset()
    }

    fn to_subset_unchecked(&self) -> f32 {
        self.x.to_subset_unchecked()
    }

    fn from_subset(element: &f32) -> Self {
        Self::cst(V::from_subset(element))
    }
}

impl<V, D> SupersetOf<f64> for F<V, D>
where
    V: SupersetOf<f64>,
    D: Zero,
{
    fn is_in_subset(&self) -> bool {
        self.dx.is_zero() && self.x.is_in_subset()
    }

    fn to_subset_unchecked(&self) -> f64 {
        self.x.to_subset_unchecked()
    }

    fn from_subset(element: &f64) -> Self {
        Self::cst(V::from_subset(element))
    }
}

#[cfg(test)]
mod test {
    use simba::scalar::SupersetOf;

    use crate::F;

    macro_rules! create_superset_of_test {
        ($scalar: ty,  $name: ident) => {
            #[test]
            fn $name() {
                let variable = F::<$scalar, $scalar>::var(2.3);
                let constant = F::<$scalar, $scalar>::cst(4.1);

                assert!(!SupersetOf::<$scalar>::is_in_subset(&variable));
                assert_eq!(None, SupersetOf::<$scalar>::to_subset(&variable));

                assert!(SupersetOf::<$scalar>::is_in_subset(&constant));
                assert_eq!(
                    Some(constant.x),
                    SupersetOf::<$scalar>::to_subset(&constant)
                );

                let from_subset: F<$scalar, $scalar> = SupersetOf::<$scalar>::from_subset(&3.1);
                assert_eq!(3.1, from_subset.x);
                assert_eq!(0.0, from_subset.dx);
            }
        };
    }

    create_superset_of_test!(f32, superset_of_f32);
    create_superset_of_test!(f64, superset_of_f64);
}
