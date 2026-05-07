use crate::count::Counter;

use super::Weigher;

pub struct NaiveWeigher;

impl<C: Counter> Weigher<C> for NaiveWeigher {
    fn node_weight(&self, _count: &C) -> C {
        C::one()
    }

    fn child_weight(&self, _child_count: &C, _rest_count: &C) -> C {
        C::one()
    }
}
