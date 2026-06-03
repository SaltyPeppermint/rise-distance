pub mod dios_egraphs;
pub mod lambda;
pub mod math;
pub mod mini_rise;
pub mod prop;

use std::fmt::Display;

use egg::{Analysis, FromOp, Language};
use serde::{Deserialize, Serialize};

/// Trait for node labels in e-graphs and exprs.
pub trait MyLanguage:
    Serialize
    + for<'de> Deserialize<'de>
    + Send
    + Sync
    + Display
    + Language<Discriminant: Send + Sync>
    + FromOp<Error: Display>
    + 'static
{
}

impl<
    L: Serialize
        + for<'de> Deserialize<'de>
        + Send
        + Sync
        + Display
        + Language<Discriminant: Send + Sync>
        + FromOp<Error: Display>
        + 'static,
> MyLanguage for L
{
}

/// Trait for node labels in e-graphs and exprs.
pub trait MyAnalysis<L: MyLanguage>:
    Serialize
    + for<'de> Deserialize<'de>
    + Send
    + Sync
    + std::fmt::Debug
    + Analysis<L, Data: Send + Sync + Clone + Eq + Default>
    + Default
    + Clone
    + 'static
{
}

impl<
    N: Serialize
        + for<'de> Deserialize<'de>
        + Send
        + Sync
        + std::fmt::Debug
        + Analysis<L, Data: Send + Sync + Clone + Eq + Default>
        + Default
        + Clone
        + 'static,
    L: MyLanguage,
> MyAnalysis<L> for N
{
}

/// Which language the terms in a run folder are drawn from.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum AvailableLanguages {
    Dios,
    Math,
    Prop,
}
