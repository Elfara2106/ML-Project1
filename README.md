# ML-Project1: predicting MICHD (CS-433, EPFL)

Binary classification of Myocardial Infarct / Coronary Heart Disease (MICHD)
from the BRFSS 2015 survey (328'135 training individuals, 321 features, ~8.8%
positive labels), implemented with NumPy only.

**Best submission (AIcrowd):** F1 = 0.451, accuracy = 0.881, regularized
logistic regression. Local 5-fold cross-validation: F1 = 0.438 ± 0.005.

## Setup

- Python 3 with NumPy (`matplotlib` is only needed for `plots.py`).
- Download the data from the
  [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
  and put the three files in a `data/` folder at the root of the repository
  (it is not versioned):

```
data/x_train.csv
data/y_train.csv
data/x_test.csv
```

## Reproducing our best submission

```bash
python run.py
```

This trains the final model on the full training set and writes
`submission.csv` (~2 min). Other methods and hyper-parameters can be chosen
from the command line, e.g. `python run.py --method ridge --lambda_ 1e-4`
(see `python run.py --help`).

## Reproducing the experiments and figures of the report

```bash
python experiments.py --experiment ablation   # preprocessing ablation study (~15 min)
python experiments.py --experiment methods    # comparison of the 6 methods (~15 min)
python experiments.py --experiment tuning     # hyper-parameter grid search (~45 min)
python plots.py --threshold                   # figures/ from results/ (~1 min)
```

Every number is a 5-fold cross-validated F1 score. The experiments save their
results in `results/*.tsv`, from which `plots.py` draws the figures in
`figures/`. The first run caches the training data as `.npy` files in `data/`
(parsing the csv takes ~40 s).

## Files

| File | Content |
| --- | --- |
| `implementations.py` | The 6 required methods: GD, SGD, least squares, ridge, logistic and regularized logistic regression |
| `preprocessing.py` | Feature cleaning (`build_features`), F1 / accuracy, decision threshold selection, train/validation and k-fold splits |
| `run.py` | Trains the final model and writes the submission |
| `experiments.py` | Cross-validation experiments: ablation study, method comparison, hyper-parameter grid search |
| `plots.py` | Figures of the report |
| `helpers.py` | Data loading and submission writing (provided by the course) |
| `results/` | Results of the experiments (tsv) |
| `figures/` | Figures of the report (pdf) |

## Pipeline

1. **Special codes.** In the BRFSS codebook, 7/77/777 mean "don't know" and
   9/99/999 "refused" (replaced by NaN), 88/888 mean "none" (replaced by 0).
2. **One-hot encoding** of the categorical features (at most 10 distinct
   values); a missing value is its own category and the most frequent category
   is dropped.
3. Numerical features with more than 50% of NaN are dropped, the others are
   imputed with the median. All statistics are computed on the training set only.
4. **Standardization** and a bias column (774 features in total).
5. **Model:** regularized logistic regression, λ = 1e-4, γ = 0.5, 1000 GD steps.
6. **Decision threshold:** chosen to maximize the F1 score on the training set,
   since only ~9% of the labels are positive (probability threshold ≈ 0.2
   instead of 0.5).

## Results (5-fold cross-validation)

| Preprocessing (logistic regression) | F1 |
| --- | --- |
| Baseline (median imputation) | 0.412 ± 0.004 |
| + special codes | 0.424 ± 0.006 |
| + one-hot encoding | **0.439 ± 0.005** |
| + NaN indicators | 0.439 ± 0.005 |
| + polynomial features (degree 3) | 0.439 ± 0.005 |

| Method (special codes + one-hot) | Hyper-parameters | F1 |
| --- | --- | --- |
| Least squares | | 0.429 ± 0.005 |
| Ridge regression | λ = 1e-3 | 0.429 ± 0.005 |
| GD on MSE | γ = 0.05, 1000 steps | 0.429 ± 0.005 |
| SGD on MSE | γ = 1e-4, 300'000 steps | 0.420 ± 0.006 |
| Logistic regression | γ = 0.5, 1000 steps | **0.439 ± 0.005** |
| Regularized logistic regression | λ = 1e-4, γ = 0.5, 1000 steps | 0.438 ± 0.005 |
