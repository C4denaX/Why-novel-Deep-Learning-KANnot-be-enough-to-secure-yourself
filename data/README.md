# Data

The datasets are not distributed with this repository. Obtain them from their authors (see the references in the main [README](../README.md#datasets)) and place one CSV file per dataset in this folder, with the following names:

| Key (`--dataset`) | Dataset | Expected file |
|---|---|---|
| `cicmaldroid` | CICMaldroid2020 | `data/cicmaldroid.csv` |
| `drebin` | Drebin-215 | `data/debrin.csv` |
| `malgenome` | Malgenome-215 | `data/malgenome.csv` |
| `tuandromd` | Tuandromd | `data/TUANDROMD.csv` |
| `wustl_iiot_2021` | WUSTL-IIOT-2021 | `data/wustl_iiot_2021.csv` |
| `x_iiotid` | X-IIoTID | `data/X-IIoTID dataset.csv` |
| `edge_iiotset` | Edge-IIoTset | `data/DNN-EdgeIIoT-dataset.csv` |

The file names are defined in `src/config.json` (`datasets`) and in `revision/run_seed_experiment.py` (`DATASETS`). The Drebin-215 file is named `debrin.csv` in both places. Another folder can be used with `--data-dir`.

## File format

Each file is a CSV with a header row and one row per sample. By default, the class label is the last column.

- `src/` (`run_ml.py`, `run_dl.py`) reads the label from the last column, or from `preprocessing.label_column` in `src/config.json` if it is set. Labels can be integers or strings: they are mapped to 0..C-1. Non-numeric feature columns are one-hot encoded and boolean columns are converted to 0/1. Columns listed in `preprocessing.drop_columns` are removed before anything else, which is the place to exclude columns that should not be used as features, such as identifiers.
- `revision/run_seed_experiment.py` expects every column to be numeric or boolean, with the label in the last column as an integer. It applies no further preprocessing.

This repository does not include the steps used to build these CSV files from the original releases of the datasets.

Everything in this folder except this README is ignored by git.
