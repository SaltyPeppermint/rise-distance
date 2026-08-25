mod layered;
mod novel;

use layered::LayeredDp;

pub(crate) use layered::{
    CountData, RootBudgets, count_terms_rooted, plain_dp_rooted, root_budgets,
};

pub use novel::NodeMatches;
pub use novel::{NodeMatch, NovelTermCount};
pub(crate) use novel::{enumerate_matches_rooted, find_novel_root_sizes_rooted, prune_matches};
