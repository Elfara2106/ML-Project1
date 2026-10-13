"""Cross-validation experiments used in the report.

Usage:
    python experiments.py --experiment ablation   # preprocessing ablation study
    python experiments.py --experiment methods    # comparison of the 6 methods

Every number is a k-fold cross-validated F1 score (mean +- std over the folds).
In each fold, the features are built and the decision threshold is chosen
using the training folds only; the held-out fold is used only for the score.
"""

import argparse
import os
import time

import numpy as np

import helpers
import preprocessing
from run import train

# Preprocessing variants of the ablation study, each one adds one step
ABLATION = [
    ("baseline (median imputation)", dict()),
    ("+ special codes", dict(fix_codes=True)),
    ("+ one-hot categorical", dict(fix_codes=True, one_hot=True)),
    ("+ NaN indicators", dict(fix_codes=True, one_hot=True, nan_indicators=True)),
    ("+ polynomial (deg 3)", dict(fix_codes=True, one_hot=True, poly_degree=3)),
]

# Hyper-parameters of each method (chosen on a validation split)
METHOD_PARAMS = {
    "least_squares": dict(),
    "ridge": dict(lambda_=1e-3),
    "mean_square_gd": dict(max_iters=1000, gamma=0.05),
    "mean_square_sgd": dict(max_iters=300000, gamma=1e-4),
    "log": dict(max_iters=1000, gamma=0.5),
    "reg_log": dict(lambda_=1e-4, max_iters=1000, gamma=0.5),
}


def load_data(data_path):
    """Load the training data, using a .npy cache since the csv is slow to parse."""
    cache_x = os.path.join(data_path, "x_train.npy")
    cache_y = os.path.join(data_path, "y_train.npy")
    if not os.path.exists(cache_x):
        x_train, _, y_train, _, _ = helpers.load_csv_data(data_path)
        np.save(cache_x, x_train)
        np.save(cache_y, y_train)
    return np.load(cache_x), np.load(cache_y)


def cross_validate(x, y, method, feature_kwargs, params, k_fold=5, seed=1):
    """K-fold cross-validated F1 and accuracy of a (preprocessing, method) pair.

    Returns:
        f1s, accs: numpy arrays of shape (k_fold,)
    """
    folds = preprocessing.k_fold_indices(len(y), k_fold, seed)
    args = argparse.Namespace(lambda_=0.0, max_iters=1000, gamma=0.1)
    vars(args).update(params)
    f1s, accs = [], []
    for k in range(k_fold):
        te_idx = folds[k]
        tr_idx = np.concatenate([folds[i] for i in range(k_fold) if i != k])
        tx_tr, tx_te = preprocessing.build_features(
            x[tr_idx], x[te_idx], **feature_kwargs
        )
        np.random.seed(seed)
        w, _ = train(method, y[tr_idx], tx_tr, args)
        threshold, _ = preprocessing.best_threshold(tx_tr, w, y[tr_idx])
        y_pred = preprocessing.predict_labels(tx_te, w, threshold)
        f1s.append(preprocessing.f1_score(y[te_idx], y_pred))
        accs.append(preprocessing.accuracy(y[te_idx], y_pred))
    return np.array(f1s), np.array(accs)


def report(name, f1s, accs, start):
    """Print one line of results."""
    print(
        f"{name:32s} F1 = {f1s.mean():.4f} +- {f1s.std():.4f}   "
        f"acc = {accs.mean():.4f}   ({time.time() - start:.0f}s)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Cross-validation experiments")
    parser.add_argument("--experiment", choices=["ablation", "methods"], required=True)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--k_fold", type=int, default=5)
    args = parser.parse_args()

    x, y = load_data(args.data_path)
    print(f"{args.k_fold}-fold cross-validation on {x.shape[0]} samples\n")

    if args.experiment == "ablation":
        print("Preprocessing ablation (logistic regression)")
        for name, feature_kwargs in ABLATION:
            start = time.time()
            f1s, accs = cross_validate(
                x, y, "log", feature_kwargs, METHOD_PARAMS["log"], args.k_fold
            )
            report(name, f1s, accs, start)
    else:
        print("Comparison of the methods (special codes + one-hot)")
        for method, params in METHOD_PARAMS.items():
            # SGD diverges when the rare one-hot categories are standardized
            feature_kwargs = dict(
                fix_codes=True, one_hot=True, scale_binary=method != "mean_square_sgd"
            )
            start = time.time()
            f1s, accs = cross_validate(
                x, y, method, feature_kwargs, params, args.k_fold
            )
            report(method, f1s, accs, start)


if __name__ == "__main__":
    main()
