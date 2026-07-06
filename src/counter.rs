use std::fmt::Display;
use std::iter::Sum;

use num::traits::{NumAssignRef, NumRef};
use rand::distributions::uniform::SampleUniform;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub trait Counter:
    Clone
    + Send
    + Sync
    + NumRef
    + NumAssignRef
    + Default
    + std::fmt::Debug
    + Display
    + SampleUniform
    + PartialOrd
    + for<'a> Sum<&'a Self>
    + TryInto<u64, Error: std::fmt::Debug>
    + TryFrom<u64, Error: std::fmt::Debug>
    + TryFrom<usize, Error: std::fmt::Debug>
    + Serialize
    + DeserializeOwned
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
        + Display
        + SampleUniform
        + PartialOrd
        + for<'a> Sum<&'a Self>
        + TryInto<u64, Error: std::fmt::Debug>
        + TryFrom<u64, Error: std::fmt::Debug>
        + TryFrom<usize, Error: std::fmt::Debug>
        + Serialize
        + DeserializeOwned,
> Counter for T
{
}
