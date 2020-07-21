use super::{F1, F2, F3};
use bytemuck::{Pod, Zeroable};

// Implement unsafe bytemuck traits for common known types.
// SAFETY: F is repr(C) and F1, F2, F3 are byte aligned f64s and hence will have no padding bytes.

unsafe impl Pod for F1 {}
unsafe impl Zeroable for F1 {}
unsafe impl Pod for F2 {}
unsafe impl Zeroable for F2 {}
unsafe impl Pod for F3 {}
unsafe impl Zeroable for F3 {}

#[cfg(test)]
mod tests {
    use crate::F1;
    #[test]
    fn cast() {
        let autodiff_data = &[F1::var(1.0), F1::cst(2.0), F1::cst(3.0)][..];

        let pairs: &[[f64; 2]] = bytemuck::cast_slice(autodiff_data);

        assert_eq!(pairs, &[[1.0, 1.0], [2.0, 0.0], [3.0, 0.0]]);
    }
}
