# 1. One-time prep: enrich terms.json with goals (no profiling — fast and only needs to happen once)
cargo run --release --bin goal -- --goals 5 --goal-sample-strategy naive --take-first 5 data/seed_terms/dusky-cramp

# 2. Run the guide experiments under samply
samply record cargo run --release --bin guide -- --repetitions 50 --full-union --take-first 5 data/seed_terms/clear-gel