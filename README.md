# ML-Project1: predicting MICHD (CS-433, EPFL)

Binary classification of Myocardial Infarct / Coronary Heart Disease (MICHD)
from the BRFSS survey data (328'135 training individuals, 321 features), using
only NumPy.

## Setup

Python 3 with NumPy only. Download the data from the
[AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and put the three files in a `data/` folder at the root of the repository
(it is ignored by git):

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
`submission.csv`. Other methods and hyper-parameters can be chosen from the
command line, e.g. `python run.py --method ridge --lambda_ 1e-3`
(see `python run.py --help`).

## Reproducing the experiments of the report

```bash
python experiments.py --experiment ablation   # preprocessing ablation study
python experiments.py --experiment methods    # comparison of the 6 methods
```

Every number is a 5-fold cross-validated F1 score. The first run caches the
training data as `.npy` files in `data/` (parsing the csv takes ~40 s).

## Files

| File | Content |
| --- | --- |
| `implementations.py` | The 6 required methods (GD, SGD, least squares, ridge, logistic, regularized logistic) |
| `preprocessing.py` | Feature cleaning (`build_features`), metrics (F1, accuracy), threshold selection, train/validation and k-fold splits |
| `run.py` | Trains the final model and writes the submission |
| `experiments.py` | Cross-validation experiments (ablation study, method comparison) |
| `helpers.py` | Data loading and submission writing (provided by the course) |

## Pipeline

1. **Special codes**: in the BRFSS codebook 7/77/777 mean "don't know",
   9/99/999 "refused" (both replaced by NaN) and 88/888 "none" (replaced by 0).
2. **One-hot encoding** of the categorical features (at most 10 distinct
   values); a missing value is its own category.
3. Numerical features with more than 50% of NaN are dropped, the others are
   imputed with the median of the training set.
4. **Standardization** (statistics of the training set only) and a bias column.
5. Model training, then the decision threshold is chosen to maximize the F1
   score on the training set, since only ~9% of the labels are positive.
