use clap::{Parser, Subcommand};
use rise_distance::langs::mini_rise::{self, TilingSearch};
use rise_distance::search::{BruteArgs, CutArgs, SearchMode};

#[derive(Parser)]
#[command(
    about = "Sketch-based tiling/reorder searches via the guide pipeline",
    after_help = "\
Examples:
  # Cut at iteration 8 and sample 200 terms:
  rise tile3d tile cut --cut-iters 8 --sample-count 200
  # Brute-force (no cut), up to 100 iterations:
  rise tile3d tile brute --max-iters 100
"
)]
struct Args {
    nest: Experiment,
    search: TilingSearch,

    #[command(subcommand)]
    mode: Mode,
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum Experiment {
    Tile1d,
    Tile2d,
    Tile3d,
    Tile4d,
    // Reorder3d,
}

#[derive(Subcommand)]
enum Mode {
    /// Cut at an iteration, sample the novel frontier, continue + verify.
    Cut(CutArgs),
    /// Grow one continuous egraph and check the sketches directly.
    Brute(BruteArgs),
}

fn main() {
    let args = Args::parse();
    let mode = match args.mode {
        Mode::Cut(a) => SearchMode::Cut(a),
        Mode::Brute(a) => SearchMode::Brute(a),
    };
    println!("--- {:?} {:?} {:?}", args.nest, args.search, mode);
    let result = match args.nest {
        Experiment::Tile1d => mini_rise::tile_1d(args.search, mode),
        Experiment::Tile2d => mini_rise::tile_2d(args.search, mode),
        Experiment::Tile3d => mini_rise::tile_3d(args.search, mode),
        Experiment::Tile4d => mini_rise::tile_4d(args.search, mode),
        // Experiment::Reorder3d => mini_rise::reorder_3d(mode),
    };
    match result.reached {
        Some(goal) => println!(
            "=== reached ({} sampled guide terms)\n{}",
            result.sampled.len(),
            goal,
        ),
        None => println!(
            "=== NOT reached ({} sampled guide terms)",
            result.sampled.len(),
        ),
    }
}
