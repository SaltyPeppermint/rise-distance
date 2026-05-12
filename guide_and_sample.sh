# 1. One-time prep: enrich terms.json with goals (no profiling — fast and only needs to happen once)
cargo run --release --bin goal -- \
    --goals 5 --goal-sample-strategy naive --take-first 5 \
    data/seed_terms/dusky-cramp

# 2. Run the guide experiments under samply
samply record cargo run --release --bin guide -- \
    -g 100 --trials 50 --full-union \
    data/seed_terms/dusky-cramp