use std::fs::File;
use std::path::Path;

use polars::prelude::*;

use super::{GoalSummary, GuideError, GuideEval};
use crate::{Label, OriginTree, tee_println};

/// Dump eval results to a new Parquet file inside `run_folder/out/`.
///
/// Each call creates the next numbered file (`0.parquet`, `1.parquet`, ...).
///
/// # Panics
///
/// Panics if it cannot create/open the file or write the data.
pub fn dump_full_eval_parquet<L: Label>(
    run_folder: &Path,
    seed: &str,
    goal: &OriginTree<L>,
    results: &[GuideEval<L>],
) {
    let out_dir = run_folder.join("out");
    std::fs::create_dir_all(&out_dir).expect("create out/ directory");

    let next_id = out_dir
        .read_dir()
        .expect("read out/ directory")
        .filter_map(|e| e.ok()?.path().file_stem()?.to_str()?.parse::<usize>().ok())
        .max()
        .map_or(0, |m| m + 1);

    let parquet_path = out_dir.join(format!("{next_id}.parquet"));
    let goal_str = goal.to_string();
    let n = results.len();

    // List columns need a dedicated builder; df! doesn't handle them.
    let mut nodes_builder = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
        "nodes_to_reach".into(),
        n,
        n * 10,
        DataType::UInt64,
    );
    let mut classes_builder = ListPrimitiveChunkedBuilder::<UInt64Type>::new(
        "classes_to_reach".into(),
        n,
        n * 10,
        DataType::UInt64,
    );
    for r in results {
        if let Some(iters) = &r.iterations {
            nodes_builder.append_slice(
                &iters
                    .iter()
                    .map(|i| i.egraph_nodes as u64)
                    .collect::<Vec<_>>(),
            );
            classes_builder.append_slice(
                &iters
                    .iter()
                    .map(|i| i.egraph_classes as u64)
                    .collect::<Vec<_>>(),
            );
        } else {
            nodes_builder.append_null();
            classes_builder.append_null();
        }
    }

    let mut df = df! {
        "seed"                => vec![seed; n],
        "goal"                => vec![goal_str.as_str(); n],
        "guide"               => results.iter().map(|r| r.guide.to_string()).collect::<Vec<_>>(),
        "zs_distance"         => results.iter().map(|r| r.measurements.zs_distance as u64).collect::<Vec<_>>(),
        "structural_overlap"  => results.iter().map(|r| r.measurements.structural_distance.overlap() as u64).collect::<Vec<_>>(),
        "structural_zs_sum"   => results.iter().map(|r| r.measurements.structural_distance.zs_sum() as u64).collect::<Vec<_>>(),
        "iterations_to_reach" => results.iter().map(|r| r.iterations.as_ref().map(|i| i.len() as u64)).collect::<Vec<Option<u64>>>(),
        "ms_to_reach"         => results.iter().map(|r| r.iterations.as_ref().map(|i| i.iter().map(|x| x.total_time).sum::<f64>())).collect::<Vec<Option<f64>>>(),
    }
    .expect("build DataFrame");

    df.with_column(nodes_builder.finish().into_column())
        .expect("add nodes_to_reach column");
    df.with_column(classes_builder.finish().into_column())
        .expect("add classes_to_reach column");

    let file = File::create(&parquet_path).expect("create parquet file");
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut df)
        .expect("write parquet");

    tee_println!("Wrote goal to {}", parquet_path.display());
}

/// Write a flat `top_k_summary.parquet` from pre-computed summaries.
///
/// # Panics
///
/// Panics on I/O or Arrow errors.
pub fn dump_summary_parquet(path: &Path, summaries: &[GoalSummary]) {
    let n: usize = summaries
        .iter()
        .map(|s| {
            s.entries_per_k
                .values()
                .map(|trials| trials.len())
                .sum::<usize>()
        })
        .sum();

    let mut seeds: Vec<&str> = Vec::with_capacity(n);
    let mut goals: Vec<&str> = Vec::with_capacity(n);
    let mut ks: Vec<u64> = Vec::with_capacity(n);
    let mut iters: Vec<Option<u64>> = Vec::with_capacity(n);
    let mut nodes: Vec<Option<u64>> = Vec::with_capacity(n);
    let mut classes: Vec<Option<u64>> = Vec::with_capacity(n);
    let mut total_applied: Vec<Option<u64>> = Vec::with_capacity(n);
    let mut total_time: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut not_enough_samples: Vec<bool> = Vec::with_capacity(n);
    let mut unreached: Vec<bool> = Vec::with_capacity(n);

    for summary in summaries {
        for (k, trials) in &summary.entries_per_k {
            for trial in trials {
                seeds.push(&summary.seed);
                goals.push(&summary.goal);
                ks.push(*k as u64);
                match trial {
                    Ok(t) => {
                        iters.push(Some(t.iters as u64));
                        nodes.push(Some(t.nodes as u64));
                        classes.push(Some(t.classes as u64));
                        total_applied.push(Some(t.total_applied as u64));
                        total_time.push(Some(t.total_time));
                        not_enough_samples.push(false);
                        unreached.push(false);
                    }
                    Err(e) => {
                        iters.push(None);
                        nodes.push(None);
                        classes.push(None);
                        total_applied.push(None);
                        total_time.push(None);
                        not_enough_samples.push(matches!(e, GuideError::InsufficientSamples));
                        unreached.push(matches!(e, GuideError::Unreached));
                    }
                }
            }
        }
    }

    let mut df = df! {
        "seed"                => seeds,
        "goal"                => goals,
        "k"                   => ks,
        "iters"               => iters,
        "nodes"               => nodes,
        "classes"             => classes,
        "total_applied"       => total_applied,
        "total_time"          => total_time,
        "not_enough_samples"  => not_enough_samples,
        "unreached"           => unreached,
    }
    .expect("build summary DataFrame");

    let file = File::create(path).expect("create summary parquet file");
    ParquetWriter::new(file)
        .with_compression(ParquetCompression::Zstd(None)) // ZstdLevel::try_new(3).expect("valid zstd level")
        .finish(&mut df)
        .expect("write parquet");
}
