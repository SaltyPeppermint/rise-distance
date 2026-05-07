use std::iter::{Product, Sum};

use num_traits::{NumAssignRef, NumRef};
use rand::distributions::uniform::SampleUniform;

mod novel;
mod plain;

pub use novel::{NodeMatch, NovelTermCount};
pub use plain::PlainTermCount;

pub trait Counter:
    Clone
    + Send
    + Sync
    + NumRef
    + NumAssignRef
    + Default
    + std::fmt::Debug
    + SampleUniform
    + PartialEq
    + Ord
    + for<'a> Sum<&'a Self>
    + TryInto<u64, Error: std::fmt::Debug>
    + TryFrom<u64, Error: std::fmt::Debug>
    + TryFrom<usize, Error: std::fmt::Debug>
    + Product // + Weight
{
}

impl<
    T: Clone
        + Send
        + Sync
        + NumRef
        + NumAssignRef
        + Default
        + std::fmt::Debug
        + SampleUniform
        + PartialEq
        + Ord
        + for<'a> Sum<&'a Self>
        + TryInto<u64, Error: std::fmt::Debug>
        + TryFrom<u64, Error: std::fmt::Debug>
        + TryFrom<usize, Error: std::fmt::Debug>
        + Product, // + Weight,
> Counter for T
{
}
