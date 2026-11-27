use std::array;
use std::fmt::Display;
use std::ops::*;

use crate::Field;
use crate::vector::Vector;

#[allow(dead_code)]
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Matrix<K: Field, const H: usize, const W: usize> {
    data: [[K; W]; H],
}

#[allow(dead_code)]
impl<K: Field, const N: usize> Matrix<K, N, N> {
    fn identity() -> Self {
        array::from_fn(|row| array::from_fn(|col| if row == col { K::one() } else { K::zero() }))
            .into()
    }

    fn trace(&self) -> K {
        (0..N).map(|i| self.data[i][i]).sum()
    }

    fn determinant(&self) -> K {
        let mut c = [0; N];
        let mut perm: [usize; N] = array::from_fn(|i| i);
        let mut parity = K::one();
        let mut sum = (0..N)
            .map(|i| self.data[perm[i]][i])
            .fold(K::one(), |product, value| product * value);

        let mut i = 0;
        while i < N {
            if c[i] < i {
                if i & 1 == 0 {
                    perm.swap(0, i);
                } else {
                    perm.swap(c[i], i);
                }

                parity = -parity;

                sum += parity
                    * (0..N)
                        .map(|i| self.data[perm[i]][i])
                        .fold(K::one(), |product, value| product * value);

                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        sum
    }

    fn inverse(&self) -> Result<Self, &'static str> {
        if self.determinant() == K::zero() {
            return Err("Singular matrix has no inverse");
        }

        // TODO need 2 * N columns
        // let augmented: Matrix<K, N, 2 * N> = array::from_fn(|i| {
        //     array::from_fn(|j| {
        //         if j < N {
        //             self.data[i][j]
        //         } else {
        //             if i == j - N { K::one() } else { K::zero() }
        //         }
        //     })
        // })
        // .into();

        // augmented.row_echelon();

        todo!()
    }
}

#[allow(dead_code)]
impl<K: Field, const H: usize, const W: usize> Matrix<K, H, W> {
    fn transpose(&self) -> Matrix<K, W, H> {
        array::from_fn(|i| array::from_fn(|j| self.data[j][i])).into()
    }

    fn row_echelon(&self) -> Self {
        let mut echeloned = self.clone();
        let mut pivot_col = 0;

        'outer: for row in 0..H {
            if pivot_col >= W {
                break;
            }

            let mut pivot_row = row;
            while echeloned.data[pivot_row][pivot_col] == K::zero() {
                pivot_row += 1;
                if pivot_row == H {
                    pivot_row = row;
                    pivot_col += 1;
                    if W == pivot_col {
                        break 'outer;
                    }
                }
            }

            if pivot_row != row {
                echeloned.data.swap(row, pivot_row);
            }

            let divisor = echeloned.data[row][pivot_col];
            echeloned.data[row].iter_mut().for_each(|x| *x /= divisor);

            for j in 0..H {
                if j != row {
                    let hold = echeloned.data[j][pivot_col];
                    for k in 0..W {
                        echeloned.data[j][k] -= hold * echeloned.data[row][k];
                    }
                }
            }

            pivot_col += 1;
        }

        echeloned
    }
}

impl<K: Field, const H: usize, const W: usize> Add for Matrix<K, H, W> {
    type Output = Matrix<K, H, W>;

    fn add(self, rhs: Self) -> Self::Output {
        array::from_fn(|i| array::from_fn(|j| self.data[i][j] + rhs.data[i][j])).into()
    }
}

impl<K: Field, const H: usize, const W: usize> Sub for Matrix<K, H, W> {
    type Output = Matrix<K, H, W>;

    fn sub(self, rhs: Self) -> Self::Output {
        array::from_fn(|i| array::from_fn(|j| self.data[i][j] - rhs.data[i][j])).into()
    }
}

impl<K: Field, const H: usize, const W: usize> Mul<K> for Matrix<K, H, W> {
    type Output = Matrix<K, H, W>;

    fn mul(self, rhs: K) -> Self::Output {
        array::from_fn(|i| array::from_fn(|j| self.data[i][j] * rhs)).into()
    }
}

impl<K: Field, const H: usize, const W: usize> Mul<Vector<K, W>> for Matrix<K, H, W> {
    type Output = Vector<K, H>;

    fn mul(self, rhs: Vector<K, W>) -> Self::Output {
        array::from_fn(|i| (0..W).map(|j| self.data[i][j] * rhs.data[j]).sum()).into()
    }
}

impl<K: Field, const H: usize, const N: usize, const W: usize> Mul<Matrix<K, N, W>>
    for Matrix<K, H, N>
{
    type Output = Matrix<K, H, W>;

    fn mul(self, rhs: Matrix<K, N, W>) -> Self::Output {
        array::from_fn(|i| {
            array::from_fn(|j| (0..N).map(|k| self.data[i][k] * rhs.data[k][j]).sum())
        })
        .into()
    }
}

impl<K: Field, const H: usize, const W: usize> From<[[K; W]; H]> for Matrix<K, H, W> {
    fn from(value: [[K; W]; H]) -> Self {
        Self { data: value }
    }
}

impl<K: Field, const H: usize, const W: usize> Display for Matrix<K, H, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Matrix {\n")?;

        for row in self.data {
            f.write_fmt(format_args!("   {:?}\n", row))?;
        }
        f.write_str("}\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    #[test]
    fn test_matrix() {
        let mat1 = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let mat2 = Matrix::from([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]);

        assert_eq!(
            mat1.clone() + mat2.clone(),
            [[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]].into()
        );
        assert_eq!(
            mat1.clone() - mat2.clone(),
            [[-8., -6., -4.], [-2., 0., 2.], [4., 6., 8.]].into()
        );
        assert_eq!(
            mat1 * 5.,
            [[5., 10., 15.], [20., 25., 30.], [35., 40., 45.]].into()
        );
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

    #[test]
    fn test_matrix_trace() {
        let u = Matrix::from([[1., 0.], [0., 1.]]);
        assert_eq!(u.trace(), 2.);

        let u = Matrix::from([[2., -5., 0.], [4., 3., 7.], [-2., 3., 4.]]);
        assert_eq!(u.trace(), 9.);

        let u = Matrix::from([[-2., -8., 4.], [1., -23., 4.], [0., 6., 4.]]);
        assert_eq!(u.trace(), -21.);
    }

    #[test]
    fn test_matrix_transpose() {
        let u = Matrix::from([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(
            u.transpose(),
            [[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]].into()
        );
        assert_eq!(u.transpose().transpose(), u);

        let u = Matrix::from([[1., 2.], [3., 4.], [5., 6.]]);
        assert_eq!(u.transpose(), [[1., 3., 5.], [2., 4., 6.]].into());
        assert_eq!(u.transpose().transpose(), u);
    }

    #[test]
    fn test_matrix_row_echelon() {
        let u = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        assert_eq!(
            u.row_echelon(),
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]].into()
        );

        let u = Matrix::from([[1., 2.], [3., 4.]]);
        assert_eq!(u.row_echelon(), [[1., 0.], [0., 1.]].into());

        let u = Matrix::from([[1., 2.], [2., 4.]]);
        assert_eq!(u.row_echelon(), [[1., 2.], [0., 0.]].into());

        let u = Matrix::from([
            [8., 5., -2., 4., 28.],
            [4., 2.5, 20., 4., -4.],
            [8., 5., 1., 4., 17.],
        ]);
        assert_eq!(
            u.row_echelon(),
            [
                [1., 0.625, 0., 0., -12.166666666666668],
                [0., 0., 1., 0., -3.666666666666667],
                [0., 0., 0., 1., 29.500000000000004]
            ]
            .into()
        );

        let u = Matrix::from([[2., 1., -1., 8.], [-3., -1., 2., -11.], [-2., 1., 2., -3.]]);
        assert_eq!(
            u.row_echelon(),
            [[1., 0., 0., 2.], [0., 1., 0., 3.], [0., 0., 1., -1.]].into()
        );
    }

    #[test]
    fn test_matrix_determinant() {
        let u = Matrix::from([[1., -1.], [-1., 1.]]);
        assert_eq!(u.determinant(), 0.);

        let u = Matrix::from([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]);
        assert_eq!(u.determinant(), 8.);

        let u = Matrix::from([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]);
        assert_eq!(u.determinant(), -174.);

        let u = Matrix::from([
            [8., 5., -2., 4.],
            [4., 2.5, 20., 4.],
            [8., 5., 1., 4.],
            [28., -4., 17., 1.],
        ]);
        assert_eq!(u.determinant(), 1032.);

        let u = Matrix::from([
            [2., -1., 3., 0., -2., 1.],
            [4., -1., 5., 2., -3., 4.],
            [-2., 3., -4., 1., 4., -1.],
            [6., -2., 8., 3., -5., 7.],
            [-4., 4., -6., 2., 7., -2.],
            [2., 1., 2., 4., -1., 8.],
        ]);
        assert_eq!(u.determinant(), -6.);
    }

    #[test]
    fn test_matrix_inverse() {
        let u = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        assert_eq!(
            u.inverse(),
            Ok([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]].into())
        );
        assert_eq!(u.inverse().unwrap() * u, Matrix::identity()); // TODO may need an epsilon

        let u = Matrix::from([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]);
        assert_eq!(
            u.inverse(),
            Ok([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]].into())
        );
        assert_eq!(u.inverse().unwrap() * u, Matrix::identity());

        let u = Matrix::from([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]);
        assert_eq!(
            u.inverse(),
            Ok([
                [0.649425287, 0.097701149, -0.655172414],
                [-0.781609195, -0.126436782, 0.965517241],
                [0.143678161, 0.074712644, -0.206896552]
            ]
            .into())
        );
        assert_eq!(u.inverse().unwrap() * u, Matrix::identity());

        let u = Matrix::from([[2., 3.], [4., 6.]]);
        assert_matches!(u.inverse(), Err(_));
    }
}
