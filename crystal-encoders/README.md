# Crystal Encoders

Reference implementation of **Contrastive Crystal Encoders for Materials
Property Prediction**.

This project directory contains the complete training and evaluation
pipeline used for the paper, including data preparation, model
definitions, training scripts and the evaluation code that produces the
main results table.

## Contents

```
requirements.txt
src/data.py         crystal structure loading and featurisation
src/model.py        contrastive crystal encoder
src/train.py        training entry point
results/            saved metrics from the released checkpoints
```

## Setup

```bash
pip install -r requirements.txt
python src/train.py
```

## Results

`results/property_mae.csv` lists the mean absolute error of the released
checkpoints on the MP-20 and OQMD splits.

## Citation

```bibtex
@article{bennett2024crystal,
  title={Contrastive Crystal Encoders for Materials Property Prediction},
  author={Bennett, A. and Duarte, L. and Iyer, S.},
  journal={Under review at NeurIPS},
  year={2024}
}
```
