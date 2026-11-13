# DiffSolverBench

Evaluation harness for **DiffSolverBench: A Benchmark for Differentiable
Constraint Solving** (NeurIPS 2024, Datasets and Benchmarks Track).

This project directory ships the eight task generators, the reference
solver implementations, the metrics used in the paper and the script that
reproduces the leaderboard.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

## Leaderboard

`results/leaderboard.csv` contains the numbers reported in the paper.

## Citation

```bibtex
@inproceedings{bennett2024diffsolver,
  title={DiffSolverBench: A Benchmark for Differentiable Constraint Solving},
  author={Bennett, A. and Okonkwo, R. and Lindqvist, P.},
  booktitle={NeurIPS 2024 Datasets and Benchmarks},
  year={2024}
}
```
