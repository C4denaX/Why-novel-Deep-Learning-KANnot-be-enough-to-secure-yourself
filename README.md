# Why novel Deep Learning KANnot be enough to secure yourself? A cybersecurity study

This repository contains the code and the result files of the paper. The study compares Kolmogorov-Arnold Network (KAN) models with deep learning (DL) models and tree-based machine learning (ML) models on four Android malware datasets and three Industrial Internet of Things (IIoT) intrusion detection datasets, in terms of F1-Score, runtime and number of parameters.

- `src/` implements the experimental workflow described in the Methodology section of the paper: preprocessing, feature selection with Extremely Randomized Trees, a 70/30 split, grid search with five-fold cross-validation, and evaluation on the held-out test set. It was rewritten for this release following Section 3 of the paper.
- `revision/` contains the scripts of two additional analyses: five training runs of each DL and KAN model (seeds 40-44) to estimate the run-to-run variation, and the sensitivity of the KAN configurations to the grid size and the spline order on CICMaldroid2020 (three runs per point).
- `results/` contains the result files (CSV).

## Repository structure

```
.
├── src/                    experimental pipeline
│   ├── config.json         datasets, search spaces and fixed hyperparameters
│   ├── preprocessing.py    data transformation, data optimisation, ERT feature selection
│   ├── data_split.py       70/30 stratified split and stratified k-fold cross-validation
│   ├── metrics.py          evaluation metrics
│   ├── dl_models.py        construction of the DL and KAN models
│   ├── run_ml.py           ML models: grid search, retraining, evaluation
│   ├── run_dl.py           DL and KAN models: grid search, retraining, evaluation
│   └── utils.py
├── architectures/          DL and KAN architectures (PyTorch)
├── kan_convolutional/      KAN layers from Convolutional-KANs (MIT, see THIRD_PARTY_NOTICES.md)
├── revision/               five-run and KAN sensitivity analyses
├── results/
│   ├── five_runs/          <dataset>/<model>__seed<N>.csv and seed_windows_per_cell.csv
│   └── kan_sensitivity/    cicmaldroid/<model>__g<grid>[__k<order>]__seed<N>.csv
├── data/                   place the dataset files here (see data/README.md)
├── requirements.txt
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## Models

| Group | Name in the paper | Name in the code and result files | Definition |
|---|---|---|---|
| DL | CNN (Small) | `CNN (Small)` | `SimpleCNN` in `architectures/SimpleModels.py` |
| DL | CNN (Medium) | `CNN (Medium)` | `SimpleCNN_2` in `architectures/SimpleModels.py` |
| DL | CNN (Big) | `CNN (Big)` | `ConvNet1D` in `architectures/ConvNet.py` |
| DL | RNN | `RNN` | `RNNMalware_Model1` in `architectures/RnnModels.py` |
| DL | LSTM | `LSTM` | `LSTMMalware_Model1` in `architectures/RnnModels.py` |
| DL | GRU | `GRU` | `GRUMalware_Model1` in `architectures/RnnModels.py` |
| KAN | ConKAN (Small) | `ConvKAN (Small)` | `Simple_ConvKAN` in `architectures/kan1d.py` |
| KAN | ConKAN (Medium) | `ConvKAN (Medium)` | `Simple_ConvKAN_2` in `architectures/kan1d.py` |
| KAN | ConKAN (Big) | `ConvKAN (Big)` | `ConvKAN` in `architectures/kan1d.py` |
| KAN | Conv & KAN | `Convs_and_KAN` | `NormalConvsKAN` in `architectures/conv_and_kan.py` |
| KAN | KAN | `KAN` | `FastKAN` from the `fastkan` package, layers `[features, 128, classes]` |
| ML | RF | `Random Forest` | scikit-learn `RandomForestClassifier` |
| ML | XGBoost | `Xgboost` | `xgboost.XGBClassifier` |
| ML | LightGBM | `LightGBM` | `lightgbm.LGBMClassifier` |
| ML | AdaBoost | `Adaboost` | scikit-learn `AdaBoostClassifier` |
| ML | CatBoost | `Catboost` | `catboost.CatBoostClassifier` |
| ML | GradientBoost | `GradientBoost` | scikit-learn `GradientBoostingClassifier` |

The ConKAN layers are `KANConv1DLayer` (B-splines, `kan_convolutional/kan_conv.py`), the KAN layer of Conv & KAN is `KANLinear` (B-splines, `kan_convolutional/KANLinear.py`), and the KAN model uses FastKAN (Gaussian radial basis functions; Z. Li, *Kolmogorov-Arnold Networks are Radial Basis Function Networks*, arXiv:2405.06721, 2024).

## Datasets

| Dataset | Task | Key | Reference |
|---|---|---|---|
| CICMaldroid2020 | Android malware category classification | `cicmaldroid` | S. Mahdavifar, A. F. Abdul Kadir, R. Fatemi, D. Alhadidi, A. A. Ghorbani. Dynamic Android Malware Category Classification using Semi-Supervised Deep Learning. 2020 IEEE Intl Conf on Dependable, Autonomic and Secure Computing, Intl Conf on Pervasive Intelligence and Computing, Intl Conf on Cloud and Big Data Computing, Intl Conf on Cyber Science and Technology Congress (DASC/PiCom/CBDCom/CyberSciTech), pp. 515-522, 2020. [doi:10.1109/DASC-PICom-CBDCom-CyberSciTech49142.2020.00094](https://doi.org/10.1109/DASC-PICom-CBDCom-CyberSciTech49142.2020.00094) |
| Drebin-215 | Android malware detection | `drebin` | S. Y. Yerima, S. Sezer. DroidFusion: A Novel Multilevel Classifier Fusion Approach for Android Malware Detection. IEEE Transactions on Cybernetics 49(2):453-466, 2019. [doi:10.1109/TCYB.2017.2777960](https://doi.org/10.1109/TCYB.2017.2777960) |
| Malgenome-215 | Android malware detection | `malgenome` | Same reference as Drebin-215. |
| Tuandromd | Android malware detection | `tuandromd` | P. Borah, D. K. Bhattacharyya, J. K. Kalita. Malware Dataset Generation and Evaluation. 2020 IEEE 4th Conference on Information & Communication Technology (CICT), pp. 1-6, 2020. |
| WUSTL-IIOT-2021 | IIoT intrusion detection | `wustl_iiot_2021` | M. Zolanvari, M. A. Teixeira, L. Gupta, K. M. Khan, R. Jain. Machine Learning-Based Network Vulnerability Analysis of Industrial Internet of Things. IEEE Internet of Things Journal 6(4):6822-6834, 2019. [doi:10.1109/JIOT.2019.2912022](https://doi.org/10.1109/JIOT.2019.2912022) |
| X-IIoTID | IIoT intrusion detection | `x_iiotid` | M. Al-Hawawreh, E. Sitnikova, N. Aboutorab. X-IIoTID: A Connectivity-Agnostic and Device-Agnostic Intrusion Data Set for Industrial Internet of Things. IEEE Internet of Things Journal 9(5):3962-3977, 2022. [doi:10.1109/JIOT.2021.3102056](https://doi.org/10.1109/JIOT.2021.3102056) |
| Edge-IIoTset | IIoT intrusion detection | `edge_iiotset` | M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, H. Janicke. Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning. IEEE Access 10:40281-40306, 2022. [doi:10.1109/ACCESS.2022.3165809](https://doi.org/10.1109/ACCESS.2022.3165809) |

The data files are not included. [`data/README.md`](data/README.md) lists the file names that the code expects and the required format.

## Installation

Python 3.9 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs FastKAN from https://github.com/ZiyaoLi/fast-kan at a fixed commit, so `git` must be available. To use a GPU, install the PyTorch build that matches your CUDA version before the other requirements.

All commands below are run from the repository root.

## Experimental pipeline (`src/`)

### Steps

| Methodology step (paper, Section 3) | Implementation |
|---|---|
| Data transformation: categorical features to binary representations, one numerical value per category | `preprocessing.transform_features` one-hot encodes non-numeric feature columns and converts boolean columns to 0/1; `preprocessing.encode_labels` maps the class labels to 0..C-1 |
| Data optimisation: review of data types to reduce memory | `preprocessing.optimise_dtypes` downcasts integer columns and stores floating-point columns as float32 |
| Data split: 70% training, 30% held-out test | `data_split.train_test`: stratified, shuffled split with `random_state=42` |
| Feature selection with Extremely Randomized Trees (ERT) | `preprocessing.ERTFeatureSelector`: scikit-learn `ExtraTreesClassifier` with `SelectFromModel`, fitted on the training split only; it keeps the features whose importance is at least the mean importance |
| Five-fold cross-validation and Grid Search (Table 3) | `data_split.cv_folds` (stratified, shuffled); `run_ml.py` uses `GridSearchCV` with the macro F1-Score; `run_dl.py` runs the grid over epochs, learning_rate, weight_decay and gamma and selects the combination with the highest mean F1-Score over the folds |
| DL training: AdamW, ExponentialLR, CrossEntropyLoss | `run_dl.fit_and_evaluate` |
| KAN-specific hyperparameters fixed (Table 3) | `dl.kan` in `src/config.json`: ConKAN grid size 5 and spline order 3 on [-1, 1] with GELU; Conv & KAN grid size 10 and spline order 3 on [0, 1] with SiLU; KAN with 8 radial basis functions on [-2, 2] with SiLU |
| Output layer adapted to the number of classes | `dl_models.build_model` sets the output size of every model to the number of classes of the dataset |
| Evaluation on the held-out set: accuracy, precision, recall, F1-Score, runtime, number of parameters | After the search, each model is retrained on the whole training split with the best values and evaluated on the test split. `metrics.py` holds the metric functions |

The search spaces are in `src/config.json`. They are discrete values inside the ranges of Table 3 of the paper; the `_notes` entry of the file explains the choices where Table 3 lists a value that current libraries no longer accept or an alias of another parameter. The paper does not give the number of trees or the threshold of the ERT selection; the values in the configuration are those of this implementation.

For the DL and KAN models, a run of E epochs is the first E epochs of a run with more epochs and the same seed, because the learning rate schedule does not depend on the total number of epochs. `run_dl.py` therefore trains each fold once for the largest number of epochs in the grid and evaluates it after each number of epochs listed in the grid.

Precision, recall and F1-Score are macro averages computed with scikit-learn. For the DL and KAN models, `metrics.dl_metrics` averages over the classes that appear in the predictions (`labels=np.unique(predictions)`); for the ML models, `metrics.ml_metrics` uses `average="macro"`. Runtime is the wall-clock time, in seconds, of the final training on the training split plus the evaluation on the test split; the time of the grid search is stored separately.

### Running

```bash
python -m src.run_ml --dataset cicmaldroid
python -m src.run_dl --dataset cicmaldroid
```

Options of both runners:

| Option | Meaning |
|---|---|
| `--dataset KEY` | dataset key (required): `cicmaldroid`, `drebin`, `malgenome`, `tuandromd`, `wustl_iiot_2021`, `x_iiotid`, `edge_iiotset` |
| `--models NAME [NAME ...]` | subset of models, with the names of the tables above (default: all models of the result tables) |
| `--config PATH` | alternative configuration file (default: `src/config.json`) |
| `--data-dir DIR` | folder with the dataset files (default: `data`) |
| `--out-dir DIR` | output folder (default: `outputs`) |
| `--seed N` | random seed (default: `seed` in the configuration, 42) |
| `--folds K` | number of cross-validation folds (default: 5) |
| `--device {auto,cpu,cuda}` | `auto` uses CUDA when available. In `run_ml.py` it applies to XGBoost only |
| `--subsample F` | use a stratified fraction F of the rows, for quick checks of the pipeline |
| `--n-jobs N` | `run_ml.py` only: parallel jobs of `GridSearchCV` (default: 1) |

Example of a quick check with a reduced configuration: `python -m src.run_dl --dataset malgenome --config my_small_config.json --folds 2 --subsample 0.2 --models "CNN (Small)" KAN`.

Seeds are set for Python, NumPy and PyTorch before every training run, and the ML models receive the seed through their `random_state` (or `random_seed`) parameter, except XGBoost, whose seed is part of its search space.

With the configuration as distributed, the grid search trains each DL and KAN model 27 times per fold for 200 epochs, and the convolutional models use a batch size of 4. A GPU is recommended for the DL and KAN models, and smaller search spaces can be set in a copy of the configuration.

### Outputs

For each dataset, in `outputs/`:

- `ml/<dataset>.csv`: `Model`, `Test Accuracy`, `Test Precision`, `Test Recall`, `Test F1 Score`, `Runtime (s)`.
- `dl_kan/<dataset>.csv`: `Model`, `Test Accuracy`, `Test Precision`, `Test Recall`, `Test F1 Score`, `Test Loss`, `Number of Parameters`, `Runtime (s)`.
- `ml/<dataset>_search.json` and `dl_kan/<dataset>_search.json`: description of the data used (rows, selected features, classes), and for every model the best hyperparameters, the cross-validation score of every combination and the time of the grid search.

Models of one dataset can be run in separate calls: rows of models that are already in the CSV file are replaced.

Results obtained with this code may differ from the values reported in the paper because of library versions, hardware and random seeds.

## Additional analyses (`revision/`)

`run_seed_experiment.py` trains one DL or KAN model on one dataset with one seed and writes one CSV file. The launchers run many of these jobs in parallel and skip the jobs whose output file already exists. `run_seed_experiment.py` exits if CUDA is not available.

### Five training runs per DL and KAN model (seeds 40-44)

```bash
# one run
python revision/run_seed_experiment.py --dataset cicmaldroid --model "ConvKAN (Small)" --seed 40

# all datasets and models, seeds 40-44
python revision/launch_seed_runs.py --seeds 40 41 42 43 44

# mean, standard deviation and variance of the F1-Score per dataset and model
python revision/aggregate_seed_windows.py --results-dir results_seeds
```

- `run_seed_experiment.py`: `--dataset` (required), `--model` (required, one of the eleven DL and KAN names), `--seed` (required), `--data-dir` (default `data`), `--out-dir` (default `results_seeds`), `--threads` (CPU threads of PyTorch, default 2), `--grid-size` and `--spline-order` (override the KAN hyperparameters; used by the sensitivity analysis). Output: `<out-dir>/<dataset>/<model>[__g<grid>][__k<order>]__seed<N>.csv`, plus a `_history.json` file with the per-epoch values.
- `launch_seed_runs.py`: `--seeds` (default `42 123 2024`), `--datasets`, `--models` (default: all), `--workers` (parallel jobs, default 16), `--threads` (default 2), `--data-dir`, `--out-dir` (default `results_seeds`), `--log-dir` (default `logs_seeds`), `--dry-run` (list the pending jobs only).
- `aggregate_seed_windows.py`: `--results-dir` (default `results_seeds`). It needs seeds 40 to 44 for the 77 dataset and model pairs. It computes the mean, standard deviation and variance of the F1-Score, in points, for the five seeds and for the three-seed subsets 40-42, 41-43 and 42-44, and writes `seed_windows_per_cell.csv`, `seed_windows_summary.csv` and `seed_windows_tables.md` (in Spanish) to the results folder.

To aggregate the runs distributed in `results/five_runs`, pass `--results-dir results/five_runs` (the aggregation scripts write their output files to that folder).

### Sensitivity of the KAN configurations to grid size and spline order

```bash
python revision/launch_grid_sensitivity.py --dataset cicmaldroid
python revision/aggregate_sensitivity.py --results-dir results/kan_sensitivity --out-tex tables_sensitivity.tex
```

- `launch_grid_sensitivity.py`: `--dataset` (default `cicmaldroid`), `--models` (default: the five KAN configurations), `--seeds` (default `42 123 2024`), `--sweep {grid,order,both}` (default `both`), `--workers` (default 4), `--threads` (default 2), `--data-dir`, `--out-dir` (default `results_sensitivity`), `--log-dir` (default `logs_sensitivity`), `--dry-run`. The grid sweep uses grid sizes 3, 5, 10 and 20 for ConKAN and Conv & KAN, and 4, 8, 16 and 32 radial basis functions for the KAN model, with the default spline order. The order sweep uses spline orders 1, 2 and 5 with the default grid size (order 3 is already covered by the grid sweep); it does not apply to the KAN model.
- `aggregate_sensitivity.py`: `--results-dir` (default `results_sensitivity`), `--dataset` (default `cicmaldroid`), `--out-tex` (default `results_sensitivity/tables_sensitivity.tex`). It prints the mean and standard deviation of the F1-Score, the number of parameters and the mean runtime of each configuration, and writes two LaTeX tables.

## Results (`results/`)

All metric values are fractions between 0 and 1, except in `seed_windows_per_cell.csv`.

### `results/five_runs/`

- `<dataset>/<model>__seed<N>.csv`, N = 40 to 44: one file per run (7 datasets, 11 models, 5 seeds; 385 files) with the columns `Dataset`, `Seed`, `Model`, `Test Accuracy`, `Test Precision`, `Test Recall`, `Test F1 Score`, `Test Loss`, `Number of Parameters`, `Best Epoch` (epoch whose values are reported), `Runtime (s)` and `Device` (GPU name). Precision, recall and F1-Score are computed as in `metrics.dl_metrics`. The runtime depends on how many jobs shared the machine.
- `seed_windows_per_cell.csv`: one row per dataset and model for seeds 40-44 (`Seeds` = `40-44`), with `f1_mean`, `f1_std` (sample standard deviation) and `f1_var` of the F1-Score in percentage points, and `any_collapsed` (1 if the Test Recall of any of the runs is above 0.9999).

### `results/kan_sensitivity/cicmaldroid/`

96 files, one per run, for the five KAN configurations on CICMaldroid2020 with seeds 42, 123 and 2024. Same columns as the five-run files plus `Grid Size` and `Spline Order` (empty when the default order of 3 was used).

The current version of `run_seed_experiment.py` also writes columns that the distributed files do not have: `Grid Size` and `Spline Order` (absent from the five-run files), the precision, recall and F1-Score averaged over all the classes of the test set (`... (all classes)`), and the number of predicted classes (`Predicted Classes`).

## Citation

Sergio Ruiz-Villafranca, Luis Miguel García-Sáez, José Roldán-Gómez, Juan Manuel Castelo Gómez, Javier Carrillo-Mondéjar. *Why novel Deep Learning KANnot be enough to secure yourself? A cybersecurity study.*

## License

The code in this repository is released under the MIT License ([`LICENSE`](LICENSE)). It includes code from [Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs) (MIT License) in `kan_convolutional/`, and some modules in `architectures/` are adapted from it. FastKAN ([fast-kan](https://github.com/ZiyaoLi/fast-kan), Apache License 2.0) is a dependency and is not included. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
