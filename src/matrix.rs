use std::array;
use std::fmt::Display;
use std::ops::*;

use crate::Field;
use crate::vector::Vector;

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Matrix<K: Field, const N: usize, const M: usize> {
    data: [[K; M]; N],
}

impl<K: Field, const N: usize, const M: usize> Add for Matrix<K, N, M> {
    type Output = Matrix<K, N, M>;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| self.data[i][j] + rhs.data[i][j])),
        }
    }
}

impl<K: Field, const N: usize, const M: usize> Sub for Matrix<K, N, M> {
    type Output = Matrix<K, N, M>;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| self.data[i][j] - rhs.data[i][j])),
        }
    }
}

impl<K: Field, const N: usize, const M: usize> Mul<K> for Matrix<K, N, M> {
    type Output = Matrix<K, N, M>;

    fn mul(self, rhs: K) -> Self::Output {
        Self {
            data: array::from_fn(|i| array::from_fn(|j| self.data[i][j] * rhs)),
        }
    }
}

impl<K: Field, const N: usize, const M: usize> Mul<Vector<K, M>> for Matrix<K, N, M> {
    type Output = Vector<K, N>;

    fn mul(self, rhs: Vector<K, M>) -> Self::Output {
        array::from_fn(|i| (0..M).map(|j| self.data[i][j] * rhs.data[j]).sum()).into()
    }
}

impl<K: Field, const N: usize, const M: usize, const P: usize> Mul<Matrix<K, M, P>>
    for Matrix<K, N, M>
{
    type Output = Matrix<K, N, P>;

    fn mul(self, rhs: Matrix<K, M, P>) -> Self::Output {
        array::from_fn(|i| {
            array::from_fn(|j| (0..M).map(|k| self.data[i][k] * rhs.data[k][j]).sum())
        })
        .into()
    }
}

impl<K: Field, const N: usize, const M: usize> From<[[K; M]; N]> for Matrix<K, N, M> {
    fn from(value: [[K; M]; N]) -> Self {
        Self { data: value }
    }
}

impl<K: Field, const N: usize, const M: usize> Display for Matrix<K, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Matrix {\n")?;

        for i in 0..N {
            f.write_fmt(format_args!("   {:?}\n", self.data[i]))?;
        }
        f.write_str("}\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix() {
        let mat1 = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let mat2 = Matrix::from([[9, 8, 7], [6, 5, 4], [3, 2, 1]]);

        assert_eq!(
            mat1.clone() + mat2.clone(),
            [[10, 10, 10], [10, 10, 10], [10, 10, 10]].into()
        );
        assert_eq!(
            mat1.clone() - mat2.clone(),
            [[-8, -6, -4], [-2, 0, 2], [4, 6, 8]].into()
        );
        assert_eq!(mat1 * 5, [[5, 10, 15], [20, 25, 30], [35, 40, 45]].into());
    }

    #[test]
    fn test_matrix_mul_vector() {
        let u = Matrix::from([[1., 0.], [0., 1.]]);
        let v = Vector::from([4., 2.]);
        assert_eq!(u * v, [4., 2.].into());

        let u = Matrix::from([[2., 0.], [0., 2.]]);
        let v = Vector::from([4., 2.]);
        assert_eq!(u * v, [8., 4.].into());

        let u = Matrix::from([[2., -2.], [-2., 2.]]);
        let v = Vector::from([4., 2.]);
        assert_eq!(u * v, [4., -4.].into());

        let u = Matrix::from([[1., 2.], [3., 4.], [5., 6.]]);
        let v = Vector::from([7., 8.]);
        assert_eq!(u * v, [23., 53., 83.].into());
    }

    #[test]
    fn test_matrix_mul_matrix() {
        let u = Matrix::from([[1., 0.], [0., 1.]]);
        let v = Matrix::from([[1., 0.], [0., 1.]]);
        assert_eq!(u * v, [[1., 0.], [0., 1.]].into());

        let u = Matrix::from([[1., 0.], [0., 1.]]);
        let v = Matrix::from([[2., 1.], [4., 2.]]);
        assert_eq!(u * v, [[2., 1.], [4., 2.]].into());

        let u = Matrix::from([[3., -5.], [6., 8.]]);
        let v = Matrix::from([[2., 1.], [4., 2.]]);
        assert_eq!(u * v, [[-14., -7.], [44., 22.]].into());

        let u = Matrix::from([[1., 2., 3.], [4., 5., 6.]]);
        let v = Matrix::from([[10., 11.], [20., 21.], [30., 31.]]);
        assert_eq!(u * v, [[140., 146.], [320., 335.]].into());
    }
}
