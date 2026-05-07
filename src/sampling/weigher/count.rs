use crate::count::Counter;

use super::Weigher;

pub struct CountWeigher;

impl<C: Counter> Weigher<C> for CountWeigher {
    fn node_weight(&self, count: &C) -> C {
        count.clone()
    }

    fn child_weight(&self, child_count: &C, rest_count: &C) -> C {
        child_count.to_owned() * rest_count
    }
}
