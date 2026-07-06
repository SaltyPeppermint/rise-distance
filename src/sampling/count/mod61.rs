//! Fingerprint counter for cheap feasibility probes.
//!
//! [`Mod61`] is a drop-in [`Counter`](crate::sampling::Counter) that does all
//! arithmetic in the prime field `Z/(2^61 - 1)` instead of with exact big
//! integers. Running a counting analysis with it computes the image of the
//! exact count modulo the Mersenne prime `2^61 - 1`: a nonzero fingerprint
//! *proves* the exact count is nonzero, while a zero fingerprint is a real
//! zero unless the exact count happens to be a nonzero multiple of the prime
//! (probability on the order of `2^-61` for these combinatorial counts).
//!
//! Overflow safety: residues are kept `< 2^61 - 1` at all times. Sums of two
//! residues fit in 62 bits, subtraction adds at most one extra `P`, and
//! products are computed in `u128` (at most 122 bits) before being reduced
//! with the Mersenne identity `2^61 ≡ 1 (mod P)`. No operation can wrap.

use std::fmt::Display;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
};

use num::traits::{Num, One, Zero};
use rand::Rng;
use rand::distributions::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler};
use serde::{Deserialize, Serialize};

/// The Mersenne prime `2^61 - 1`.
const P: u64 = (1 << 61) - 1;

/// An element of `Z/(2^61 - 1)`. Invariant: the residue is always `< P`.
///
/// `Ord` compares residues, which is meaningless as an ordering on the
/// underlying exact counts; it exists only to satisfy the
/// [`Counter`](crate::sampling::Counter) bounds. The same goes for
/// division (exact field division) and uniform sampling.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(from = "u64", into = "u64")]
pub struct Mod61(u64);

impl Mod61 {
    pub const ZERO: Self = Self(0);

    /// Reduce a value `<= 2 * P` into the canonical range.
    const fn reduce_once(v: u64) -> u64 {
        if v >= P { v - P } else { v }
    }

    const fn add_inner(self, rhs: Self) -> Self {
        // Both < P, so the sum is < 2^62 and cannot wrap.
        Self(Self::reduce_once(self.0 + rhs.0))
    }

    const fn sub_inner(self, rhs: Self) -> Self {
        // self.0 + P < 2^62, and the sum is >= P - rhs.0 > 0.
        Self(Self::reduce_once(self.0 + P - rhs.0))
    }

    #[expect(clippy::cast_possible_truncation)]
    const fn mul_inner(self, rhs: Self) -> Self {
        // Both operands are < 2^61, so the product is < 2^122 and splits as
        // x = hi * 2^61 + lo with hi, lo <= P; by 2^61 ≡ 1, x ≡ hi + lo.
        let x = self.0 as u128 * rhs.0 as u128;
        let lo = (x & P as u128) as u64;
        let hi = (x >> 61) as u64;
        // lo + hi <= 2 * P < 2^63; reduce twice to cover the == 2 * P edge.
        Self(Self::reduce_once(Self::reduce_once(lo + hi)))
    }

    const fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut acc = Self(1);
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.mul_inner(base);
            }
            base = base.mul_inner(base);
            exp >>= 1;
        }
        acc
    }

    /// Multiplicative inverse; total on nonzero elements since `P` is prime.
    const fn inv(self) -> Self {
        assert!(self.0 != 0, "division by zero mod 2^61 - 1");
        self.pow(P - 2)
    }

    const fn div_inner(self, rhs: Self) -> Self {
        self.mul_inner(rhs.inv())
    }

    /// Field division is exact, so the remainder is always zero (still
    /// panics on a zero divisor, matching integer semantics).
    #[expect(clippy::unused_self)]
    const fn rem_inner(self, rhs: Self) -> Self {
        assert!(rhs.0 != 0, "division by zero mod 2^61 - 1");
        Self(0)
    }
}

impl From<u64> for Mod61 {
    fn from(v: u64) -> Self {
        Self(v % P)
    }
}

impl From<Mod61> for u64 {
    fn from(v: Mod61) -> Self {
        v.0
    }
}

impl TryFrom<usize> for Mod61 {
    type Error = std::num::TryFromIntError;

    fn try_from(v: usize) -> Result<Self, Self::Error> {
        u64::try_from(v).map(Self::from)
    }
}

impl Display for Mod61 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Zero for Mod61 {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for Mod61 {
    fn one() -> Self {
        Self(1)
    }
}

impl Num for Mod61 {
    type FromStrRadixErr = std::num::ParseIntError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        u64::from_str_radix(str, radix).map(Self::from)
    }
}

macro_rules! impl_binop {
    ($OpTrait:ident, $op:ident, $AssignTrait:ident, $assign:ident, $inner:ident) => {
        impl $OpTrait for Mod61 {
            type Output = Mod61;

            fn $op(self, rhs: Mod61) -> Mod61 {
                self.$inner(rhs)
            }
        }

        impl $OpTrait<&Mod61> for Mod61 {
            type Output = Mod61;

            fn $op(self, rhs: &Mod61) -> Mod61 {
                self.$inner(*rhs)
            }
        }

        impl $AssignTrait for Mod61 {
            fn $assign(&mut self, rhs: Mod61) {
                *self = self.$inner(rhs);
            }
        }

        impl $AssignTrait<&Mod61> for Mod61 {
            fn $assign(&mut self, rhs: &Mod61) {
                *self = self.$inner(*rhs);
            }
        }
    };
}

impl_binop!(Add, add, AddAssign, add_assign, add_inner);
impl_binop!(Sub, sub, SubAssign, sub_assign, sub_inner);
impl_binop!(Mul, mul, MulAssign, mul_assign, mul_inner);
impl_binop!(Div, div, DivAssign, div_assign, div_inner);
impl_binop!(Rem, rem, RemAssign, rem_assign, rem_inner);

impl Sum for Mod61 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(0), Self::add_inner)
    }
}

impl<'a> Sum<&'a Mod61> for Mod61 {
    fn sum<I: Iterator<Item = &'a Mod61>>(iter: I) -> Self {
        iter.fold(Self(0), |acc, x| acc.add_inner(*x))
    }
}

impl Product for Mod61 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(1), Self::mul_inner)
    }
}

impl<'a> Product<&'a Mod61> for Mod61 {
    fn product<I: Iterator<Item = &'a Mod61>>(iter: I) -> Self {
        iter.fold(Self(1), |acc, x| acc.mul_inner(*x))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UniformMod61(UniformInt<u64>);

impl UniformSampler for UniformMod61 {
    type X = Mod61;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformInt::new(low.borrow().0, high.borrow().0))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformInt::new_inclusive(low.borrow().0, high.borrow().0))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        // Bounds come from canonical residues, so the sample is < P.
        Mod61(self.0.sample(rng))
    }
}

impl SampleUniform for Mod61 {
    type Sampler = UniformMod61;
}

#[cfg(test)]
mod tests {
    use super::*;

    const P128: u128 = (1 << 61) - 1;

    /// Edge-case residues plus a few arbitrary ones.
    fn samples() -> Vec<u64> {
        vec![0, 1, 2, 3, 7, 61, P - 2, P - 1, 0x1234_5678_9abc_def0 % P]
    }

    #[test]
    fn from_reduces() {
        assert_eq!(u64::from(Mod61::from(P)), 0);
        assert_eq!(u64::from(Mod61::from(P + 5)), 5);
        // 2^64 - 1 = 8 * 2^61 - 1 ≡ 8 - 1 = 7 (mod 2^61 - 1)
        assert_eq!(u64::from(Mod61::from(u64::MAX)), 7);
    }

    #[test]
    fn ops_match_u128_reference() {
        for &a in &samples() {
            for &b in &samples() {
                let (ma, mb) = (Mod61::from(a), Mod61::from(b));
                let (a, b) = (u128::from(a), u128::from(b));
                assert_eq!(u128::from(u64::from(ma + mb)), (a + b) % P128, "add");
                assert_eq!(
                    u128::from(u64::from(ma - mb)),
                    (P128 + a - b) % P128,
                    "sub"
                );
                assert_eq!(u128::from(u64::from(ma * mb)), (a * b) % P128, "mul");
            }
        }
    }

    #[test]
    fn div_inverts_mul() {
        for &a in &samples() {
            for &b in &samples() {
                if b == 0 {
                    continue;
                }
                let (ma, mb) = (Mod61::from(a), Mod61::from(b));
                assert_eq!((ma / mb) * mb, ma, "a={a} b={b}");
                assert_eq!(ma % mb, Mod61::ZERO);
            }
        }
    }

    #[test]
    fn fermat_little_theorem() {
        for &a in &samples() {
            if a == 0 {
                continue;
            }
            assert_eq!(Mod61::from(a).pow(P - 1), Mod61::from(1u64));
        }
    }

    #[test]
    fn sum_and_product_match_reference() {
        let values = samples();
        let sum: Mod61 = values.iter().map(|&v| Mod61::from(v)).sum();
        let expected_sum = values.iter().map(|&v| u128::from(v)).sum::<u128>() % P128;
        assert_eq!(u128::from(u64::from(sum)), expected_sum);

        let product: Mod61 = values.iter().map(|&v| Mod61::from(v)).product();
        let expected_product = values
            .iter()
            .fold(1u128, |acc, &v| (acc * u128::from(v)) % P128);
        assert_eq!(u128::from(u64::from(product)), expected_product);
    }

    #[test]
    fn assign_ops_match_binops() {
        let a = Mod61::from(P - 2);
        let b = Mod61::from(12_345u64);
        let mut x = a;
        x += &b;
        assert_eq!(x, a + b);
        x -= &b;
        assert_eq!(x, a);
        x *= &b;
        assert_eq!(x, a * b);
    }
}
