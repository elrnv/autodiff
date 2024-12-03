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
{
    std::array::from_fn(|i| {
        // safety: the arrays have the same length by compile-time constant
        let x = unsafe { array.get_unchecked(i) };
        func(x.clone())
    })
}

pub(crate) fn binary_op<T, U, V, F, const N: usize>(lhs: [T; N], rhs: [U; N], func: F) -> [V; N]
where
    T: Clone,
    U: Clone,
    F: Fn(T, U) -> V,
{
    std::array::from_fn(|i| {
        // safety: the arrays have the same length by compile-time constant
        let l = unsafe { lhs.get_unchecked(i) };
        let r = unsafe { rhs.get_unchecked(i) };
        func(l.clone(), r.clone())
    })
}
