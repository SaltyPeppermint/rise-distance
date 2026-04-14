use std::collections::VecDeque;
use std::hash::Hash;

use hashbrown::HashSet;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

/// A data structure to maintain a queue of unique elements.
///
/// Notably, insert/pop operations have O(1) expected amortized runtime complexity.
///
/// Thanks @Bastacyclop for the implementation!
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(crate) struct UniqueQueue<T: Eq + Hash + Clone> {
    set: HashSet<T>, // hashbrown::
    queue: VecDeque<T>,
}

impl<U: Eq + Hash + Clone + Default> FromIterator<U> for UniqueQueue<U> {
    fn from_iter<T: IntoIterator<Item = U>>(iter: T) -> Self {
        let mut queue = Self::default();
        for t in iter {
            queue.insert(t);
        }
        queue
    }
}

impl<T: Eq + Hash + Clone> UniqueQueue<T> {
    pub fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter {
            self.insert(t);
        }
    }

    // pub fn pop(&mut self) -> Option<T> {
    //     let res = self.queue.pop_front();
    //     if let Some(t) = &res {
    //         self.set.remove(t);
    //     }
    //     res
    // }

    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }

    /// Drain all elements from the queue, returning them in order.
    pub fn drain(&mut self) -> std::collections::vec_deque::Drain<'_, T> {
        self.set.clear();
        self.queue.drain(..)
    }
}

pub fn combined_rng<const N: usize>(values: [u64; N]) -> ChaCha12Rng {
    const { assert!(N >= 1 && N <= 4, "must provide 1 to 4 u64 values") };

    let mut seed = [0u8; 32];
    for (i, v) in values.iter().enumerate() {
        seed[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
    }
    ChaCha12Rng::from_seed(seed)
}
