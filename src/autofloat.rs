mod add;
mod div;
mod float;
mod mul;
mod num;
mod rem;
mod scalar;
mod sub;

pub use scalar::*;

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
