use crate::sampling::Counter;

pub trait Weigher<C: Counter>: Sync + Send {
    fn node_weight(&self, count: &C) -> C;
    fn child_weight(&self, child_count: &C, rest_count: &C) -> C;
}

pub struct NaiveWeigher;

impl<C: Counter> Weigher<C> for NaiveWeigher {
    fn node_weight(&self, _count: &C) -> C {
        C::one()
    }

    fn child_weight(&self, _child_count: &C, _rest_count: &C) -> C {
        C::one()
    }
}
pub struct CountWeigher;

impl<C: Counter> Weigher<C> for CountWeigher {
    fn node_weight(&self, count: &C) -> C {
        count.clone()
    }

    fn child_weight(&self, child_count: &C, rest_count: &C) -> C {
        child_count.to_owned() * rest_count
    }
}
