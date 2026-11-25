pub mod lambda;
pub mod math;
pub mod prop;

use serde::{Deserialize, Serialize};

/// Which language the terms in a run folder are drawn from.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    Math,
    Prop,
}
