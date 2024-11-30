mod add;
mod div;
mod float_core;
pub mod float_impl;
mod mul;
mod num;
mod rem;
mod scalar;
mod sub;

#[cfg(feature = "std")]
mod float;

pub use scalar::*;

pub(crate) fn unary_op<T, V, F, const N: usize>(array: [T; N], func: F) -> [V; N]
where
    T: Clone,
    F: Fn(T) -> V,
    V: Copy + Default,
{
    let mut result = [V::default(); N];
    for (dst, src) in result.iter_mut().zip(array.into_iter()) {
        *dst = func(src);
    }
    result
}

pub(crate) fn binary_op<T, U, V, F, const N: usize>(lhs: [T; N], rhs: [U; N], func: F) -> [V; N]
where
    F: Fn(T, U) -> V,
    V: Copy + Default,
{
    let mut result = [V::default(); N];
    for (dst, (l, r)) in result.iter_mut().zip(lhs.into_iter().zip(rhs.into_iter())) {
        *dst = func(l, r);
    }
    result
}
