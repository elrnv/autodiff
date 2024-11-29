mod add;
mod div;
mod float;
mod mul;
mod num;
mod rem;
mod scalar;
mod sub;

pub use scalar::*;

fn unary_op<T, F, const N: usize>(array: [T; N], func: F) -> [T; N]
where
    T: Clone,
    F: Fn(T) -> T,
{
    let mut result = array;
    for dst in result.iter_mut() {
        *dst = func(dst.clone());
    }
    result
}

fn binary_op<T, F, const N: usize>(lhs: [T; N], rhs: [T; N], func: F) -> [T; N]
where
    T: Clone,
    F: Fn(T, T) -> T,
{
    let mut result = lhs;
    for (dst, src) in result.iter_mut().zip(rhs.into_iter()) {
        *dst = func(dst.clone(), src);
    }
    result
}

// impl<V, D> AutoFloat<V, D>
// where
//     V: Float,
//     D: Mul<V, Output = D> + Sub<Output = D> + Div<V, Output = D>,
// {
//     #[inline]
//     pub(crate) fn atan2_impl(self, other: AutoFloat<V, D>) -> AutoFloat<V, D> {
//         //let self_r = self.reduce_order();
//         //let other_r = other.reduce_order();
//         let self_r = self.x;
//         let other_r = other.x;
//         AutoFloat {
//             x: Float::atan2(self.x, other.x),
//             dx: (self.dx * other_r - other.dx * self_r) / (self_r * self_r + other_r * other_r),
//         }
//     }
// }

#[cfg(test)]
mod test {
    /// Convenience macro for comparing `AutoFloats`s in full.
    macro_rules! assert_autofloat_eq {
        ($lhs:expr, $rhs:expr $(,)?) => {
            assert_eq!($lhs.x, $rhs.x);
            assert_eq!($lhs.dx, $rhs.dx);
        };
    }
    pub(crate) use assert_autofloat_eq;
}
