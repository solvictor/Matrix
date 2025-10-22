use std::array;
use std::fmt::Display;
use std::{fmt::Debug, ops::*};

use crate::Field;

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Vector<K: Field, const N: usize> {
    data: [K; N],
}

impl<K: Field, const N: usize> Add for Vector<K, N> {
    type Output = Vector<K, N>;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: array::from_fn(|i| self.data[i] + rhs.data[i]),
        }
    }
}

impl<K: Field, const N: usize> Sub for Vector<K, N> {
    type Output = Vector<K, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: array::from_fn(|i| self.data[i] - rhs.data[i]),
        }
    }
}

impl<K: Field, const N: usize> Mul<K> for Vector<K, N> {
    type Output = Vector<K, N>;

    fn mul(self, rhs: K) -> Self::Output {
        Self {
            data: array::from_fn(|i| self.data[i] * rhs),
        }
    }
}

impl<K: Field, const N: usize> Vector<K, N> {
    fn dot(&self, v: Self) -> K {
        self.data.iter().zip(&v.data).map(|(&a, &b)| a * b).sum()
    }

    fn norm_1(&self) -> K {
        self.data.iter().map(|x| x.abs()).sum()
    }

    fn norm(&self) -> K {
        self.data.iter().map(|&x| x * x).sum::<K>().sqrt()
    }

    fn norm_inf(&self) -> K {
        self.data
            .iter()
            .map(|x| x.abs())
            .max_by(|&a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(K::default())
    }
}

impl<K: Field, const N: usize> From<[K; N]> for Vector<K, N> {
    fn from(value: [K; N]) -> Self {
        Self { data: value }
    }
}

impl<K: Field, const N: usize> Display for Vector<K, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Vector {{ {:?} }}\n", self.data))
    }
}

fn linear_combination<K: Field, const N: usize>(u: &[Vector<K, N>], coefs: &[K]) -> Vector<K, N> {
    assert_eq!(
        u.len(),
        coefs.len(),
        "Number of vectors and coefficients must match"
    );

    Vector::from(array::from_fn(|i| {
        u.iter().zip(coefs).map(|(v, &c)| v.data[i] * c).sum()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector() {
        let vec1 = Vector::from([1, 2, 3]);
        let vec2 = Vector::from([3, 2, 1]);

        assert_eq!(vec1.clone() + vec2.clone(), Vector::from([4, 4, 4]));
        assert_eq!(vec1.clone() - vec2.clone(), Vector::from([-2, 0, 2]));
        assert_eq!(vec1 * 5, Vector::from([5, 10, 15]));

        let vec1 = Vector::from([1., 2., 3.]);
        let vec2 = Vector::from([3., 2., 1.]);

        assert_eq!(vec1.clone() + vec2.clone(), Vector::from([4., 4., 4.]));
        assert_eq!(vec1.clone() - vec2.clone(), Vector::from([-2., 0., 2.]));
        assert_eq!(vec1 * 5., Vector::from([5., 10., 15.]));
    }

    #[test]
    fn test_comb() {
        let e1 = Vector::from([1., 0., 0.]);
        let e2 = Vector::from([0., 1., 0.]);
        let e3 = Vector::from([0., 0., 1.]);

        let v1 = Vector::from([1., 2., 3.]);
        let v2 = Vector::from([0., 10., -100.]);

        assert_eq!(
            linear_combination(&[e1, e2, e3], &[10., -2., 0.5]),
            Vector::from([10., -2., 0.5])
        );

        assert_eq!(
            linear_combination(&[v1, v2], &[10., -2.]),
            Vector::from([10., 0., 230.])
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_comb() {
        let e1 = Vector::from([1., 0., 0.]);
        let e2 = Vector::from([0., 1., 0.]);
        let e3 = Vector::from([0., 0., 1.]);

        let _ = linear_combination(&[e1.clone(), e2.clone(), e3.clone()], &[10., -2.]);
        let _ = linear_combination(&[e1.clone(), e2.clone()], &[10., -2., 0.5]);
    }

    #[test]
    fn test_dot() {
        let u = Vector::from([0., 0.]);
        let v = Vector::from([1., 1.]);

        assert_eq!(u.dot(v), 0.);

        let u = Vector::from([1., 1.]);
        let v = Vector::from([1., 1.]);

        assert_eq!(u.dot(v), 2.);

        let u = Vector::from([-1., 6.]);
        let v = Vector::from([3., 2.]);

        assert_eq!(u.dot(v), 9.);
    }

    #[test]
    fn test_norm() {
        let u = Vector::from([0., 0., 0.]);
        assert_eq!([u.norm_1(), u.norm(), u.norm_inf()], [0., 0., 0.]);

        let u = Vector::from([1., 2., 3.]);
        assert_eq!(
            [u.norm_1(), u.norm(), u.norm_inf()],
            [6., 3.7416573867739413, 3.]
        );

        let u = Vector::from([-1., -2.]);
        assert_eq!(
            [u.norm_1(), u.norm(), u.norm_inf()],
            [3., 2.23606797749979, 2.]
        );
    }
}
