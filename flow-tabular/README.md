# Flow Matching for Tabular Data

Official implementation of **Flow Matching for High-Dimensional Tabular
Data Generation** (AISTATS 2025).

This project directory contains the flow-matching model, the tabular
pre-processing pipeline, training and sampling scripts and the
sample-quality results reported in the paper.

## Usage

```bash
pip install -r requirements.txt
python train.py --data data/adult.csv --epochs 200
python sample.py --checkpoint checkpoints/flow.pt --n 10000
```

## Results

`results/sample_quality.csv` reports the column-wise similarity scores for
the released checkpoints.

## Citation

```bibtex
@inproceedings{bennett2025flow,
  title={Flow Matching for High-Dimensional Tabular Data Generation},
  author={Bennett, A. and Sorensen, N.},
  booktitle={AISTATS 2025},
  year={2025}
}
```
