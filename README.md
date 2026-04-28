# PSNA Anonymous Code

This repository contains the anonymized implementation of PSNA for multivariate time series forecasting.

## Directory Structure

- `run.py`: main entry point for training and evaluation.
- `model/`: PSNA model definition and normalization modules.
- `layers/`: spectral, graph, and encoder layers.
- `experiments/`: experiment runners.
- `data_provider/`: dataset loading utilities.
- `scripts/`: shell scripts for benchmark experiments.
- `utils/`: metrics, time features, and training utilities.
- `dataset/`: placeholder for datasets. Large datasets are not included in this anonymous repository.

## Environment

A typical environment is:

```bash
conda create -n psna python=3.10
conda activate psna
pip install -r requirements.txt
```

Install the PyTorch and PyTorch Geometric builds that match your CUDA version if the default installation is not suitable.

## Data Preparation

Place benchmark datasets under `dataset/` or update `--root_path` and `--data_path` in the scripts. The expected format follows common long-term forecasting benchmarks, where each CSV file contains a timestamp column and multiple variable columns.

Example layout:

```text
dataset/
  ETTh1.csv
  ETTh2.csv
  ETTm1.csv
  ETTm2.csv
  electricity.csv
  exchange_rate.csv
  weather.csv
```

## Running Experiments

Example:

```bash
bash scripts/ETTh1.sh
```

Or run directly:

```bash
python run.py --is_training 1 --model_id ETTh1_96_96 --model PSNA --data ETTh1 --root_path ./dataset/ --data_path ETTh1.csv --features M --seq_len 96 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7
```

Please adjust dataset paths, variable dimensions, and prediction horizons according to the target dataset.

## Notes for Anonymous Review

This repository has been anonymized for review. IDE files, Python cache files, checkpoints, logs, and generated results are excluded.