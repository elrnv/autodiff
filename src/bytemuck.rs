use super::F;
use bytemuck::{Pod, Zeroable};

// Implement unsafe bytemuck traits for common known types.
// SAFETY: F is repr(C) and F<T,T> is identical to a pair [T;2] which is Pod.

unsafe impl<T: Pod> Pod for F<T, T> {}
unsafe impl<T: Zeroable> Zeroable for F<T, T> {}

#[cfg(test)]
mod tests {
    use crate::F1;

    #[test]
    fn cast() {
        let autodiff_data: &[F1] = &[F1::var(1.0), F1::cst(2.0), F1::cst(3.0)][..];

        let pairs: &[[f64; 2]] = bytemuck::cast_slice(autodiff_data);

        assert_eq!(pairs, &[[1.0, 1.0], [2.0, 0.0], [3.0, 0.0]]);
    }
}
