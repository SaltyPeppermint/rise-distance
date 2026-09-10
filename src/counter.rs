use std::fmt::Display;
use std::iter::Sum;

use num::traits::{ConstOne, FromPrimitive, NumAssignRef, NumRef, ToPrimitive};
use rand::distributions::uniform::SampleUniform;
use serde::Serialize;
use serde::de::DeserializeOwned;

// TODO DELETE ME
pub trait Counter:
    Clone
    + NumRef
    + NumAssignRef
    + Default
    + std::fmt::Debug
    + Display
    + SampleUniform
    + PartialOrd
    + ConstOne
    + for<'a> Sum<&'a Self>
    + ToPrimitive
    + FromPrimitive
    + Serialize
    + DeserializeOwned
{
}

impl<
    T: Clone
        + NumRef
        + NumAssignRef
        + Default
        + std::fmt::Debug
        + Display
        + SampleUniform
        + PartialOrd
        + ConstOne
        + for<'a> Sum<&'a Self>
        + ToPrimitive
        + FromPrimitive
        + Serialize
        + DeserializeOwned,
> Counter for T
{
}
