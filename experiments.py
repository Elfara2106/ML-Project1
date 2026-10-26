"""Cross-validation experiments used in the report.

Usage:
    python experiments.py --experiment ablation   # preprocessing ablation study
    python experiments.py --experiment methods    # comparison of the 6 methods
    python experiments.py --experiment tuning     # hyper-parameter grid search

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

# Hyper-parameters of each method, confirmed by --experiment tuning: every
# value is the best of its grid or within 0.001 F1 of it (the std over the
# folds is ~0.005), e.g. lambda_ = 1e-4 and 1e-3 are equivalent for reg_log
METHOD_PARAMS = {
    "least_squares": dict(),
    "ridge": dict(lambda_=1e-3),
    "mean_square_gd": dict(max_iters=1000, gamma=0.05),
    "mean_square_sgd": dict(max_iters=300000, gamma=1e-4),
    "log": dict(max_iters=1000, gamma=0.5),
    "reg_log": dict(lambda_=1e-4, max_iters=1000, gamma=0.5),
}


# Grid of the hyper-parameter search: (method, name of the varied parameter,
# tested values, fixed parameters). The step sizes of GD are bounded by 2 / L,
# with L ~ 31.7 the largest eigenvalue of X^T X / N, above which GD diverges.
TUNING_GRID = [
    ("ridge", "lambda_", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], dict()),
    ("mean_square_gd", "gamma", [0.01, 0.02, 0.05], dict(max_iters=1000)),
    ("mean_square_sgd", "gamma", [3e-5, 1e-4, 3e-4], dict(max_iters=300000)),
    ("log", "gamma", [0.1, 0.25, 0.5, 1.0], dict(max_iters=1000)),
    ("reg_log", "lambda_", [1e-5, 1e-4, 1e-3, 1e-2], dict(max_iters=1000, gamma=0.5)),
]


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


def evaluate(method, params, y_tr, tx_tr, y_te, tx_te, seed=1):
    """Train on (y_tr, tx_tr) and choose the threshold on it.

    Returns:
        f1_train, f1_test: F1 scores on the training and on the held-out part.
        Comparing both shows whether the model over- or under-fits.
    """
    args = argparse.Namespace(lambda_=0.0, max_iters=1000, gamma=0.1)
    vars(args).update(params)
    np.random.seed(seed)
    w, _ = train(method, y_tr, tx_tr, args)
    threshold, f1_train = preprocessing.best_threshold(tx_tr, w, y_tr)
    y_pred = preprocessing.predict_labels(tx_te, w, threshold)
    return f1_train, preprocessing.f1_score(y_te, y_pred)


def tuning(x, y, k_fold=5, seed=1, output="results/tuning.tsv"):
    """K-fold cross-validated grid search over TUNING_GRID.

    The features of each fold are built once and shared by all the tested
    values, which is much faster than calling cross_validate for each value.
    The results are printed and saved in `output` (one line per value).
    """
    folds = preprocessing.k_fold_indices(len(y), k_fold, seed)
    configs = [
        (method, name, value, fixed)
        for method, name, values, fixed in TUNING_GRID
        for value in values
    ]
    f1s = np.zeros((len(configs), k_fold))
    f1s_train = np.zeros((len(configs), k_fold))
    for k in range(k_fold):
        start = time.time()
        te_idx = folds[k]
        tr_idx = np.concatenate([folds[i] for i in range(k_fold) if i != k])
        features = {}
        for c, (method, name, value, fixed) in enumerate(configs):
            # SGD diverges when the rare one-hot categories are standardized
            scale_binary = method != "mean_square_sgd"
            if scale_binary not in features:
                features[scale_binary] = preprocessing.build_features(
                    x[tr_idx],
                    x[te_idx],
                    fix_codes=True,
                    one_hot=True,
                    scale_binary=scale_binary,
                )
            tx_tr, tx_te = features[scale_binary]
            params = dict(fixed, **{name: value})
            f1s_train[c, k], f1s[c, k] = evaluate(
                method, params, y[tr_idx], tx_tr, y[te_idx], tx_te, seed
            )
        print(f"fold {k + 1}/{k_fold} done ({time.time() - start:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write("method\tparameter\tvalue\tf1_mean\tf1_std\tf1_train_mean\n")
        for (method, name, value, _), scores, scores_train in zip(
            configs, f1s, f1s_train
        ):
            f.write(
                f"{method}\t{name}\t{value:g}\t{scores.mean():.5f}\t"
                f"{scores.std():.5f}\t{scores_train.mean():.5f}\n"
            )

    print()
    for method, name, values, _ in TUNING_GRID:
        rows = [c for c, conf in enumerate(configs) if conf[0] == method]
        best = rows[int(np.argmax(f1s[rows].mean(axis=1)))]
        for c in rows:
            mark = "  <- best" if c == best else ""
            print(
                f"{method:16s} {name:8s}= {configs[c][2]:<8g} "
                f"F1 = {f1s[c].mean():.4f} +- {f1s[c].std():.4f}{mark}"
            )
        print()
    print(f"Results saved in {output}")


def report(name, f1s, accs, start, output):
    """Print one line of results and append it to the `output` tsv file."""
    print(
        f"{name:32s} F1 = {f1s.mean():.4f} +- {f1s.std():.4f}   "
        f"acc = {accs.mean():.4f}   ({time.time() - start:.0f}s)",
        flush=True,
    )
    with open(output, "a") as f:
        f.write(f"{name}\t{f1s.mean():.4f}\t{f1s.std():.4f}\t{accs.mean():.4f}\n")


def new_results_file(output, first_column):
    """Create (or overwrite) a tsv results file with its header."""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(f"{first_column}\tf1_mean\tf1_std\tacc_mean\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-validation experiments")
    parser.add_argument(
        "--experiment", choices=["ablation", "methods", "tuning"], required=True
    )
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--k_fold", type=int, default=5)
    args = parser.parse_args()

    x, y = load_data(args.data_path)
    print(f"{args.k_fold}-fold cross-validation on {x.shape[0]} samples\n")

    if args.experiment == "ablation":
        print("Preprocessing ablation (logistic regression)")
        new_results_file("results/ablation.tsv", "step")
        for name, feature_kwargs in ABLATION:
            start = time.time()
            f1s, accs = cross_validate(
                x, y, "log", feature_kwargs, METHOD_PARAMS["log"], args.k_fold
            )
            report(name, f1s, accs, start, "results/ablation.tsv")
    elif args.experiment == "tuning":
        print("Hyper-parameter grid search (special codes + one-hot)")
        tuning(x, y, args.k_fold)
    else:
        print("Comparison of the methods (special codes + one-hot)")
        new_results_file("results/methods.tsv", "method")
        for method, params in METHOD_PARAMS.items():
            # SGD diverges when the rare one-hot categories are standardized
            feature_kwargs = dict(
                fix_codes=True, one_hot=True, scale_binary=method != "mean_square_sgd"
            )
            start = time.time()
            f1s, accs = cross_validate(
                x, y, method, feature_kwargs, params, args.k_fold
            )
            report(method, f1s, accs, start, "results/methods.tsv")


if __name__ == "__main__":
    main()
