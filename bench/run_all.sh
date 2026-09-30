#!/usr/bin/env bash
# Head-to-head: original Python runtime vs this Rust port, same rows, same protocol.
# Usage: bench/run_all.sh [output.jsonl]   (run on an otherwise idle machine)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-bench/results/final.jsonl}
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
[ -f bench/data/typed.jsonl ] || python3 bench/py_bench.py export
cargo build --release --features cuda --bin julia1
RUST=./target/release/julia1
for args in "--device cuda --single 500" "--device cpu --threads 4" "--device cpu --threads 16"; do
  python3 bench/py_bench.py bench $args --output "$OUT"
  $RUST bench $args --output "$OUT"
done
python3 bench/summarize.py "$OUT"
