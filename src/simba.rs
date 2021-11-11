use super::F;
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

impl<V: SupersetOf<f64>, D: num_traits::Zero> SupersetOf<f64> for F<V, D> {
    fn is_in_subset(&self) -> bool {
        self.x.is_in_subset()
    }
    fn to_subset_unchecked(&self) -> f64 {
        self.x.to_subset_unchecked()
    }
    fn from_subset(element: &f64) -> Self {
        F::cst(V::from_subset(element))
    }
}
