mod matrix;
mod vector;

use std::{fmt::Debug, iter::Sum, ops::*};

pub trait Abs {
    fn abs(self) -> Self;
}

pub trait Sqrt {
    fn sqrt(self) -> Self;
}

pub trait Field:
    Add<Output = Self>
    + AddAssign
    + Add
    + Sub<Output = Self>
    + SubAssign
    + Sub
    + Mul<Output = Self>
    + MulAssign
    + Mul
    + Div<Output = Self>
    + DivAssign
    + Div
    + Neg<Output = Self>
    + Sum<Self>
    + Abs
    + Sqrt
    + Into<f64>
    + Default
    + Debug
    + PartialEq
    + PartialOrd
    + Copy
{
}

// Implement it on any type that already can
impl<
    T: Add<Output = T>
        + AddAssign
        + Sub<Output = T>
        + SubAssign
        + Mul<Output = T>
        + MulAssign
        + Div<Output = T>
        + DivAssign
        + Neg<Output = T>
        + Sum<T>
        + Abs
        + Sqrt
        + Into<f64>
        + Default
        + Debug
        + PartialEq
        + PartialOrd
        + Copy,
> Field for T
{
}

// TODO Macro of macro
macro_rules! impl_signed {
    ( $($t:ty),+ ) => {
        $(
            impl Abs for $t {
                fn abs(self) -> Self {
                    <$t>::abs(self)
                }
            }

            impl Sqrt for $t {
                fn sqrt(self) -> Self {
                    ((self as f64).sqrt()) as $t
                }
            }
        )+
    };
}

macro_rules! impl_unsigned {
    ( $($t:ty),+ ) => {
        $(
            impl Abs for $t {
                fn abs(self) -> Self {
                    self
                }
            }

            impl Sqrt for $t {
                fn sqrt(self) -> Self {
                    ((self as f64).sqrt()) as $t
                }
            }
        )+
    };
}

macro_rules! impl_float {
    ( $($t:ty),+ ) => {
        $(
            impl Abs for $t {
                fn abs(self) -> Self {
                    <$t>::abs(self)
                }
            }

            impl Sqrt for $t {
                fn sqrt(self) -> Self {
                    <$t>::sqrt(self)
                }
            }
        )+
    };
}

impl_signed!(i8, i16, i32, i64, i128, isize);
impl_unsigned!(u8, u16, u32, u64, u128, usize);
impl_float!(f32, f64);

// TODO Test mains
// TODO commutative operations (at least Mul)

fn lerp<V: Mul<f32, Output = V> + Add<Output = V>>(u: V, v: V, t: f32) -> V {
    u * (1. - t) + v * t
}

#[cfg(test)]
mod tests {
    use crate::{matrix::Matrix, vector::Vector};

    use super::*;

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0., 1., 0.), 0.);

        assert_eq!(lerp(0., 1., 1.), 1.);

        assert_eq!(lerp(0., 1., 0.5), 0.5);

        assert_eq!(lerp(21., 42., 0.3), 27.3);

        assert_eq!(
            lerp(Vector::from([2., 1.]), Vector::from([4., 2.]), 0.3),
            Vector::from([2.6, 1.3])
        );

        assert_eq!(
            lerp(
                Matrix::from([[2., 1.], [3., 4.]]),
                Matrix::from([[20., 10.], [30., 40.]]),
                0.5
            ),
            Matrix::from([[11., 5.5], [16.5, 22.]])
        );
    }
}
