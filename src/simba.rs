use num_traits::Zero;
use simba::scalar::{SubsetOf, SupersetOf};

use crate::{autofloat::unary_op, AutoFloat};

impl<T, const N: usize> SubsetOf<AutoFloat<T, N>> for AutoFloat<T, N>
where
    T: SubsetOf<T> + Clone,
{
    fn to_superset(&self) -> Self {
        AutoFloat {
            x: self.x.to_superset(),
            dx: unary_op(self.dx.clone(), |v| v.to_superset()),
        }
    }
    fn from_superset_unchecked(element: &AutoFloat<T, N>) -> Self {
        AutoFloat {
            x: T::from_superset_unchecked(&element.x),
            dx: unary_op(element.dx.clone(), |v| T::from_superset_unchecked(&v)),
        }
    }
    fn is_in_subset(element: &AutoFloat<T, N>) -> bool {
        T::is_in_subset(&element.x) && element.dx.iter().all(T::is_in_subset)
    }
}

impl<T, const N: usize> SubsetOf<T> for AutoFloat<T, N>
where
    T: SubsetOf<T> + Zero + Clone,
{
    fn to_superset(&self) -> T {
        self.x.to_superset()
    }
    fn from_superset_unchecked(element: &T) -> Self {
        AutoFloat::constant(T::from_superset_unchecked(&element))
    }
    fn is_in_subset(element: &T) -> bool {
        T::is_in_subset(element)
    }
}

impl<T, const N: usize> SupersetOf<f32> for AutoFloat<T, N>
where
    T: SupersetOf<f32> + Zero + Clone,
{
    fn is_in_subset(&self) -> bool {
        self.dx.iter().all(T::is_zero) && self.x.is_in_subset()
    }

    fn to_subset_unchecked(&self) -> f32 {
        self.x.to_subset_unchecked()
    }

    fn from_subset(element: &f32) -> Self {
        Self::constant(T::from_subset(element))
    }
}

impl<T, const N: usize> SupersetOf<f64> for AutoFloat<T, N>
where
    T: SupersetOf<f64> + Zero + Clone,
{
    fn is_in_subset(&self) -> bool {
        self.dx.iter().all(T::is_zero) && self.x.is_in_subset()
    }

    fn to_subset_unchecked(&self) -> f64 {
        self.x.to_subset_unchecked()
    }

    fn from_subset(element: &f64) -> Self {
        Self::constant(T::from_subset(element))
    }
}

#[cfg(test)]
mod test {
    use simba::scalar::SupersetOf;

    use crate::AutoFloat;

    macro_rules! create_superset_of_test {
        ($scalar: ty,  $name: ident) => {
            #[test]
            fn $name() {
                let v = AutoFloat::<$scalar, 2>::variable(2.3, 0);
                let c = AutoFloat::<$scalar, 2>::constant(4.1);

                assert!(!SupersetOf::<$scalar>::is_in_subset(&v));
                assert_eq!(None, SupersetOf::<$scalar>::to_subset(&v));

                assert!(SupersetOf::<$scalar>::is_in_subset(&c));
                assert_eq!(Some(c.x), SupersetOf::<$scalar>::to_subset(&c));

                let from_subset: AutoFloat<$scalar, 2> = SupersetOf::<$scalar>::from_subset(&3.1);
                assert_eq!(3.1, from_subset.x);
                assert_eq!([0.0, 0.0], from_subset.dx);
            }
        };
    }

    create_superset_of_test!(f32, superset_of_f32);
    create_superset_of_test!(f64, superset_of_f64);
}
