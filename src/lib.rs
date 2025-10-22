mod matrix;
mod vector;

use std::{fmt::Debug, ops::*};

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
    + std::iter::Sum
    + Debug
    + PartialEq
    + Copy
{
}

// Implement it on any type that already can
impl<
    T: Add<Output = Self>
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
        + std::iter::Sum
        + Debug
        + PartialEq
        + Copy,
> Field for T
{
}

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
        println!("{}", lerp(0., 1., 0.));
        // 0.0
        println!("{}", lerp(0., 1., 1.));
        // 1.0
        println!("{}", lerp(0., 1., 0.5));
        // 0.5
        println!("{}", lerp(21., 42., 0.3));
        // 27.3
        println!(
            "{}",
            lerp(Vector::from([2., 1.]), Vector::from([4., 2.]), 0.3)
        );
        // [2.6]
        // [1.3]
        println!(
            "{}",
            lerp(
                Matrix::from([[2., 1.], [3., 4.]]),
                Matrix::from([[20., 10.], [30., 40.]]),
                0.5
            )
        );
        // [[11., 5.5]
        // [16.5, 22.]]
    }
}
