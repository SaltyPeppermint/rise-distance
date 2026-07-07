use std::fmt::Display;
use std::iter::Sum;

use num::traits::{FromPrimitive, NumAssignRef, NumRef, ToPrimitive};
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
    + ToPrimitive
    + FromPrimitive
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
        + ToPrimitive
        + FromPrimitive
        + Serialize
        + DeserializeOwned,
> Counter for T
{
}
