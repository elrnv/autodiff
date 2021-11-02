use std::{iter::FromIterator, ops::*};

use num_traits::Zero;

/*
 * CwiseVec
 */

#[derive(Clone, PartialEq, Debug, PartialOrd)]
#[repr(transparent)]
pub struct CwiseVec<T>(pub Vec<T>);

impl<T> CwiseVec<T> {
    #[inline]
    pub fn new(v: Vec<T>) -> Self {
        v.into()
    }

    #[inline]
    pub fn as_slice(&self) -> &CwiseSlice<T> {
        self.0.as_slice().into()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut CwiseSlice<T> {
        self.0.as_mut_slice().into()
    }
}

impl<T> From<Vec<T>> for CwiseVec<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        CwiseVec(v)
    }
}

impl<T> FromIterator<T> for CwiseVec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<T>>().into()
    }
}

impl<T> AsRef<CwiseSlice<T>> for CwiseVec<T> {
    #[inline]
    fn as_ref(&self) -> &CwiseSlice<T> {
        self.as_slice()
    }
}
impl<T> AsMut<CwiseSlice<T>> for CwiseVec<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut CwiseSlice<T> {
        self.as_mut_slice()
    }
}

impl<T> Deref for CwiseVec<T> {
    type Target = CwiseSlice<T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for CwiseVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Zero + Copy> Zero for CwiseVec<T> {
    fn zero() -> Self {
        CwiseVec(Vec::new())
    }
    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
}

/*
 * CwiseSlice
 */

#[derive(PartialEq, Debug, PartialOrd)]
#[repr(transparent)]
pub struct CwiseSlice<T>([T]);

impl<T> AsRef<[T]> for CwiseSlice<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}
impl<T> AsMut<[T]> for CwiseSlice<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> Deref for CwiseSlice<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for CwiseSlice<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T> From<&'a [T]> for &'a CwiseSlice<T> {
    #[inline]
    fn from(slice: &'a [T]) -> Self {
        // SAFETY: CwiseSlice is a transparent [T], so this is a safe transmute.
        unsafe { std::mem::transmute(slice) }
    }
}

impl<'a, T> From<&'a mut [T]> for &'a mut CwiseSlice<T> {
    #[inline]
    fn from(slice: &'a mut [T]) -> Self {
        // SAFETY: CwiseSlice is a transparent [T], so this is a safe transmute.
        unsafe { std::mem::transmute(slice) }
    }
}

/*
 * Cwise operators
 */
macro_rules! bin_op_alloc_impl {
    (impl{$($params:tt)*} $op_trait:ident<$r_type:ty> for $l_type:ty where {$($bounds:tt)*} { type Output = $output:ty; $op_fn:ident($op_expr:expr) }) => {
        impl<$($params)*> $op_trait<$r_type> for $l_type
        where
            $($bounds)*
        {
            type Output = $output;
            #[inline]
            fn $op_fn(self, rhs: $r_type) -> Self::Output {
                self.0
                    .iter()
                    .zip(rhs.0.iter())
                    .map($op_expr)
                    .collect()
            }
        }
    }
}

macro_rules! bin_op_self_mut_impl {
    (impl{$($params:tt)*} $op_trait:ident<$r_type:ty> for $l_type:ty where {$($bounds:tt)*} { type Output = $output:ty; $op_fn:ident($op_expr:expr) }) => {
        impl<$($params)*> $op_trait<$r_type> for $l_type
        where
            $($bounds)*
        {
            type Output = $output;
            #[inline]
            fn $op_fn(mut self, rhs: $r_type) -> Self::Output {
                self.0
                    .iter_mut()
                    .zip(rhs.0.iter())
                    .for_each($op_expr);
                self
            }
        }
    }
}

macro_rules! bin_op_rhs_mut_impl {
    (impl{$($params:tt)*} $op_trait:ident<$r_type:ty> for $l_type:ty where {$($bounds:tt)*} { type Output = $output:ty; $op_fn:ident($op_expr:expr) }) => {
        impl<$($params)*> $op_trait<$r_type> for $l_type
        where
            $($bounds)*
        {
            type Output = $output;
            #[inline]
            fn $op_fn(self, mut rhs: $r_type) -> Self::Output {
                self.0
                    .iter()
                    .zip(rhs.0.iter_mut())
                    .for_each($op_expr);
                rhs
            }
        }
    }
}

macro_rules! bin_op_scalar_self_mut_impl {
    (impl{$($params:tt)*} $op_trait:ident<$r_type:ty> for $l_type:ty where {$($bounds:tt)*} { type Output = $output:ty; $op_fn:ident($op_expr:expr) }) => {
        impl<$($params)*> $op_trait<$r_type> for $l_type
        where
            $($bounds)*
        {
            type Output = $output;
            #[inline]
            fn $op_fn(mut self, rhs: $r_type) -> Self::Output {
                self.0
                    .iter_mut()
                    .zip(std::iter::repeat(rhs))
                    .for_each($op_expr);
                rhs
            }
        }
    }
}

/* CwiseSlice ops */

bin_op_alloc_impl!(impl{'r, 'l, L, R} Mul<&'r CwiseSlice<R>> for &'l CwiseSlice<L>
where {&'l L: Mul<&'r R>,}
{ type Output = CwiseVec<<&'l L as Mul<&'r R>>::Output>; mul(|(a,b)| a*b) });
bin_op_alloc_impl!(impl{'r, 'l, L, R} Div<&'r CwiseSlice<R>> for &'l CwiseSlice<L>
where {&'l L: Div<&'r R>,}
{ type Output = CwiseVec<<&'l L as Div<&'r R>>::Output>; div(|(a,b)| a/b) });
bin_op_alloc_impl!(impl{'r, 'l, L, R} Add<&'r CwiseSlice<R>> for &'l CwiseSlice<L>
where {&'l L: Add<&'r R>,}
{ type Output = CwiseVec<<&'l L as Add<&'r R>>::Output>; add(|(a,b)| a+b) });
bin_op_alloc_impl!(impl{'r, 'l, L, R} Sub<&'r CwiseSlice<R>> for &'l CwiseSlice<L>
where {&'l L: Sub<&'r R>,}
{ type Output = CwiseVec<<&'l L as Sub<&'r R>>::Output>; sub(|(a,b)| a-b) });

/* CwiseVec ops */

bin_op_self_mut_impl!(impl{L, R} Mul<CwiseVec<R>> for CwiseVec<L>
where {L: Mul<R, Output = L> + Copy, R: Copy}
{ type Output = CwiseVec<L>; mul(|(a, &b)| *a = *a * b) });
bin_op_self_mut_impl!(impl{L, R} Div<CwiseVec<R>> for CwiseVec<L>
where {L: Div<R, Output = L> + Copy, R: Copy}
{ type Output = CwiseVec<L>; div(|(a, &b)| *a = *a / b) });
bin_op_self_mut_impl!(impl{L, R} Add<CwiseVec<R>> for CwiseVec<L>
where {L: Add<R, Output = L> + Copy, R: Copy}
{ type Output = CwiseVec<L>; add(|(a, &b)| *a = *a + b) } );
bin_op_self_mut_impl!(impl{L, R} Sub<CwiseVec<R>> for CwiseVec<L>
where {L: Sub<R, Output = L> + Copy, R: Copy}
{ type Output = CwiseVec<L>; sub(|(a, &b)| *a = *a - b) } );

// Scalar ops

//bin_op_scalar_self_mut_impl!(impl{L, R} Sub<R> for CwiseVec<L>
//where {L: Sub<R, Output = L> + Copy, R: Copy}
//{ type Output = CwiseVec<L>; sub(|(a, b)| *a = *a - b) } );

/* Mixed ops */

/* CwiseVec op CwiseSlice */
bin_op_self_mut_impl!(impl{'l, L, R} Mul<&'l CwiseSlice<R>> for CwiseVec<L>
where { for<'l2, 'r> &'l2 L: Mul<&'r R, Output = L>, }
{ type Output = CwiseVec<L>; mul(|(a, b)| *a = &*a * b) });
bin_op_self_mut_impl!(impl{'l, L, R} Div<&'l CwiseSlice<R>> for CwiseVec<L>
where { for<'l2, 'r> &'l2 L: Div<&'r R, Output = L>, }
{ type Output = CwiseVec<L>; div(|(a, b)| *a = &*a / b) });
bin_op_self_mut_impl!(impl{'l, L, R} Add<&'l CwiseSlice<R>> for CwiseVec<L>
where { for<'l2, 'r> &'l2 L: Add<&'r R, Output = L>, }
{ type Output = CwiseVec<L>; add(|(a, b)| *a = &*a + b) });
bin_op_self_mut_impl!(impl{'l, L, R} Sub<&'l CwiseSlice<R>> for CwiseVec<L>
where { for<'l2, 'r> &'l2 L: Sub<&'r R, Output = L>, }
{ type Output = CwiseVec<L>; sub(|(a, b)| *a = &*a - b) });

/* CwiseSlice op CwiseVec */
bin_op_rhs_mut_impl!(impl{'l, L, R} Mul<CwiseVec<R>> for &'l CwiseSlice<L>
where {for<'l2, 'r> &'l2 L: Mul<&'r R, Output = R>,}
{ type Output = CwiseVec<R>; mul(|(a, b)| *b = a * &*b) });
bin_op_rhs_mut_impl!(impl{'l, L, R} Div<CwiseVec<R>> for &'l CwiseSlice<L>
where {for<'l2, 'r> &'l2 L: Div<&'r R, Output = R>,}
{ type Output = CwiseVec<R>; div(|(a, b)| *b = a / &*b) });
bin_op_rhs_mut_impl!(impl{'l, L, R} Add<CwiseVec<R>> for &'l CwiseSlice<L>
where {for<'l2, 'r> &'l2 L: Add<&'r R, Output = R>,}
{ type Output = CwiseVec<R>; add(|(a, b)| *b = a + &*b) });
bin_op_rhs_mut_impl!(impl{'l, L, R} Sub<CwiseVec<R>> for &'l CwiseSlice<L>
where {for<'l2, 'r> &'l2 L: Sub<&'r R, Output = R>,}
{ type Output = CwiseVec<R>; sub(|(a, b)| *b = a - &*b) });

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F;

    #[test]
    fn ops() {
        let a = CwiseVec(vec![1.0, 2.0, 3.0]);
        let b = CwiseVec(vec![9.0, 1.0, 5.0]);
        let mul_exp = CwiseVec(vec![9.0, 2.0, 15.0]);
        assert_eq!(a.as_slice() * b.as_slice(), mul_exp);
        assert_eq!(a.clone() * b.as_slice(), mul_exp);
        assert_eq!(a.as_slice() * b.clone(), mul_exp);
        assert_eq!(a.clone() * b.clone(), mul_exp);

        let div_exp = CwiseVec(vec![1.0 / 9.0, 2.0, 3.0 / 5.0]);
        assert_eq!(a.as_slice() / b.as_slice(), div_exp);
        assert_eq!(a.clone() / b.as_slice(), div_exp);
        assert_eq!(a.as_slice() / b.clone(), div_exp);
        assert_eq!(a.clone() / b.clone(), div_exp);

        let add_exp = CwiseVec(vec![10.0, 3.0, 8.0]);
        assert_eq!(a.as_slice() + b.as_slice(), add_exp);
        assert_eq!(a.clone() + b.as_slice(), add_exp);
        assert_eq!(a.as_slice() + b.clone(), add_exp);
        assert_eq!(a.clone() + b.clone(), add_exp);

        let sub_exp = CwiseVec(vec![-8.0, 1.0, -2.0]);
        assert_eq!(a.as_slice() - b.as_slice(), sub_exp);
        assert_eq!(a.clone() - b.as_slice(), sub_exp);
        assert_eq!(a.as_slice() - b.clone(), sub_exp);
        assert_eq!(a.clone() - b.clone(), sub_exp);
    }

    #[test]
    fn deriv_for_cwise() {
        let a = F::new(1.0, CwiseVec(vec![1.0, 0.0, 0.0]));
        let b = F::cst(2.0);

        let c = a * b;
        eprintln!("{:?}", c.deriv());
    }
}
