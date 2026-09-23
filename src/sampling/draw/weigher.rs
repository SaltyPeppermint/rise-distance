use num::BigUint;

pub trait Weigher {
    fn node_weight(&self, count: &BigUint) -> BigUint;
    fn child_weight(&self, child_count: &BigUint, rest_count: &BigUint) -> BigUint;
}

pub struct Uniform;

impl Weigher for Uniform {
    fn node_weight(&self, _count: &BigUint) -> BigUint {
        BigUint::ONE
    }

    fn child_weight(&self, _child_count: &BigUint, _rest_count: &BigUint) -> BigUint {
        BigUint::ONE
    }
}
pub struct Count;

impl Weigher for Count {
    fn node_weight(&self, count: &BigUint) -> BigUint {
        count.clone()
    }

    fn child_weight(&self, child_count: &BigUint, rest_count: &BigUint) -> BigUint {
        child_count.to_owned() * rest_count
    }
}
