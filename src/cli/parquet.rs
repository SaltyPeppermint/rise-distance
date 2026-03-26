use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, ListBuilder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use super::{GoalSummary, GuideEval};
use crate::{Label, OriginTree, tee_println};

/// Dump eval results to a new Parquet file inside `run_folder/out/`.
///
/// Each call creates the next numbered file (`0.parquet`, `1.parquet`, …).
///
/// # Panics
///
/// Panics if it cannot create/open the file or write the data.
pub fn dump_to_parquet<L: Label>(
    run_folder: &Path,
    goal: &OriginTree<L>,
    results: &[GuideEval<'_, L>],
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

    let schema = parquet_schema();

    // Build column arrays
    let n = results.len();
    let mut goals = StringBuilder::with_capacity(n, goal_str.len() * n);
    let mut guides = StringBuilder::new();
    let mut zs_distances = UInt64Builder::with_capacity(n);
    let mut structural_overlaps = UInt64Builder::with_capacity(n);
    let mut structural_zs_sums = UInt64Builder::with_capacity(n);
    let mut iterations_to_reach = UInt64Builder::with_capacity(n);
    let mut ms_to_reach = Float64Builder::with_capacity(n);
    let mut nodes_to_reach = ListBuilder::new(UInt64Builder::new()).with_field(Field::new(
        "item",
        DataType::UInt64,
        false,
    ));
    let mut classes_to_reach = ListBuilder::new(UInt64Builder::new()).with_field(Field::new(
        "item",
        DataType::UInt64,
        false,
    ));

    for r in results {
        goals.append_value(&goal_str);
        guides.append_value(r.guide.guide.to_string());
        zs_distances.append_value(r.guide.zs_distance.try_into().unwrap());
        structural_overlaps.append_value(r.guide.structural_distance.overlap().try_into().unwrap());
        structural_zs_sums.append_value(r.guide.structural_distance.zs_sum().try_into().unwrap());

        if let Some(iters) = &r.iterations {
            iterations_to_reach.append_value(iters.len().try_into().unwrap());
            let t = iters.iter().map(|i| i.total_time).sum::<f64>();
            ms_to_reach.append_value(t);
            let node_vals = iters
                .iter()
                .map(|i| Some(i.egraph_nodes.try_into().unwrap()));
            nodes_to_reach.append_value(node_vals);
            let class_vals = iters
                .iter()
                .map(|i| Some(i.egraph_classes.try_into().unwrap()));
            classes_to_reach.append_value(class_vals);
        } else {
            iterations_to_reach.append_null();
            ms_to_reach.append_null();
            nodes_to_reach.append_null();
            classes_to_reach.append_null();
        }
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(goals.finish()) as ArrayRef,
            Arc::new(guides.finish()) as ArrayRef,
            Arc::new(zs_distances.finish()) as ArrayRef,
            Arc::new(structural_overlaps.finish()) as ArrayRef,
            Arc::new(structural_zs_sums.finish()) as ArrayRef,
            Arc::new(iterations_to_reach.finish()) as ArrayRef,
            Arc::new(ms_to_reach.finish()) as ArrayRef,
            Arc::new(nodes_to_reach.finish()) as ArrayRef,
            Arc::new(classes_to_reach.finish()) as ArrayRef,
        ],
    )
    .expect("build record batch");

    let file = File::create(&parquet_path).expect("create parquet file");
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(3).expect("valid zstd level"),
        ))
        .build();
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(props)).expect("create parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");

    tee_println!("Wrote goal to {}", parquet_path.display());
}

fn parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("goal", DataType::Utf8, false),
        Field::new("guide", DataType::Utf8, false),
        Field::new("zs_distance", DataType::UInt64, false),
        Field::new("structural_overlap", DataType::UInt64, false),
        Field::new("structural_zs_sum", DataType::UInt64, false),
        Field::new("iterations_to_reach", DataType::UInt64, true),
        Field::new("ms_to_reach", DataType::Float64, true),
        Field::new(
            "nodes_to_reach",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
            true,
        ),
        Field::new(
            "classes_to_reach",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
            true,
        ),
    ]))
}

/// Write a flat `top_k_summary.parquet` from pre-computed summaries.
///
/// Columns: `goal` (Utf8), `k` (`UInt64`), `iters`/`nodes`/`classes`/`total_applied`
/// (nullable `UInt64`), `total_time` (nullable `Float64`).
///
/// # Panics
///
/// Panics on I/O or Arrow errors.
pub fn dump_goal_summary_parquet(path: &Path, summaries: &[GoalSummary]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("goal", DataType::Utf8, false),
        Field::new("k", DataType::UInt64, false),
        Field::new("iters", DataType::UInt64, true),
        Field::new("nodes", DataType::UInt64, true),
        Field::new("classes", DataType::UInt64, true),
        Field::new("total_applied", DataType::UInt64, true),
        Field::new("total_time", DataType::Float64, true),
    ]));

    let n: usize = summaries
        .iter()
        .map(|s| s.entries.iter().map(|e| e.trials.len()).sum::<usize>())
        .sum();

    let mut goals = StringBuilder::with_capacity(n, 64 * n);
    let mut ks = UInt64Builder::with_capacity(n);
    let mut iters = UInt64Builder::with_capacity(n);
    let mut nodes = UInt64Builder::with_capacity(n);
    let mut classes = UInt64Builder::with_capacity(n);
    let mut total_applied = UInt64Builder::with_capacity(n);
    let mut total_time = Float64Builder::with_capacity(n);

    for summary in summaries {
        for entry in &summary.entries {
            for trial in &entry.trials {
                goals.append_value(&summary.goal);
                ks.append_value(entry.k as u64);
                if let Some(t) = trial {
                    iters.append_value(t.iters as u64);
                    nodes.append_value(t.nodes as u64);
                    classes.append_value(t.classes as u64);
                    total_applied.append_value(t.total_applied as u64);
                    total_time.append_value(t.total_time);
                } else {
                    iters.append_null();
                    nodes.append_null();
                    classes.append_null();
                    total_applied.append_null();
                    total_time.append_null();
                }
            }
        }
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(goals.finish()) as ArrayRef,
            Arc::new(ks.finish()) as ArrayRef,
            Arc::new(iters.finish()) as ArrayRef,
            Arc::new(nodes.finish()) as ArrayRef,
            Arc::new(classes.finish()) as ArrayRef,
            Arc::new(total_applied.finish()) as ArrayRef,
            Arc::new(total_time.finish()) as ArrayRef,
        ],
    )
    .expect("build summary record batch");

    let file = File::create(path).expect("create summary parquet file");
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(3).expect("valid zstd level"),
        ))
        .build();
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(props)).expect("create parquet writer");
    writer.write(&batch).expect("write parquet batch");
    writer.close().expect("close parquet writer");
}
