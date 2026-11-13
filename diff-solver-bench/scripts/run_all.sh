#!/usr/bin/env bash
set -euo pipefail
for task in sudoku nqueens graph_coloring tsp scheduling portfolio circuit packing; do
  python -m src.benchmark --task "$task" --solver gradient
done
