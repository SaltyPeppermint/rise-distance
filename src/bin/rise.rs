use clap::Parser;
use rise_distance::egg::mini_rise::{self, TilingSearch};

#[derive(Parser)]
struct Args {
    command: Command,
    search: TilingSearch,
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum Command {
    Tile1d,
    Tile2d,
    Tile3d,
    Tile4d,
    Reorder3d,
}

fn main() {
    let args = Args::parse();
    println!("--- {:?} {:?}", args.command, args.search);
    match args.command {
        Command::Tile1d => mini_rise::tile_1d(args.search),
        Command::Tile2d => mini_rise::tile_2d(args.search),
        Command::Tile3d => mini_rise::tile_3d(args.search),
        Command::Tile4d => mini_rise::tile_4d(args.search),
        Command::Reorder3d => mini_rise::reorder_3d(),
    };
}
