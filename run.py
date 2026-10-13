"""Train a model on the MICHD dataset and write the AIcrowd submission.

Example:
    python run.py                                  # our best submission
    python run.py --method ridge --lambda_ 1e-3    # another model

The method and its hyper-parameters are given as command-line arguments so
that they can be changed without editing the code. The default values are the
ones of our best submission.
"""

import argparse

import numpy as np

import helpers
import implementations
import preprocessing

METHODS = [
    "least_squares",
    "ridge",
    "mean_square_gd",
    "mean_square_sgd",
    "log",
    "reg_log",
]


def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch machine learning")
    parser.add_argument(
        "--method", type=str, default="reg_log", choices=METHODS, help="Method to use"
    )
    parser.add_argument(
        "--lambda_", type=float, default=1e-4, help="Regularization parameter"
    )
    parser.add_argument(
        "--max_iters", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Step size (learning rate)"
    )
    parser.add_argument(
        "--nan_threshold",
        type=float,
        default=0.5,
        help="Drop the features with a larger fraction of NaN",
    )
    parser.add_argument(
        "--fix_codes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace the BRFSS special codes (don't know / refused / none)",
    )
    parser.add_argument(
        "--one_hot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="One-hot encode the categorical features",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Folder containing x_train.csv, y_train.csv, x_test.csv",
    )
    parser.add_argument(
        "--sub_sample",
        action="store_true",
        help="Use 1/50 of the training data (for debugging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Name of the submission file",
    )
    return parser.parse_args()


def train(method, y, tx, args):
    """Train the chosen method.

    Args:
        method: one of METHODS
        y: labels in {-1, 1}
        tx: design matrix of shape (N, D)
        args: parsed command-line arguments (hyper-parameters)

    Returns:
        w: numpy array of shape (D,), learned weights
        loss: training loss
    """
    initial_w = np.zeros(tx.shape[1])
    # logistic regression expects labels in {0, 1}
    y01 = preprocessing.to_01(y)

    if method == "least_squares":
        return implementations.least_squares(y, tx)
    if method == "ridge":
        return implementations.ridge_regression(y, tx, args.lambda_)
    if method == "mean_square_gd":
        return implementations.mean_squared_error_gd(
            y, tx, initial_w, args.max_iters, args.gamma
        )
    if method == "mean_square_sgd":
        return implementations.mean_squared_error_sgd(
            y, tx, initial_w, args.max_iters, args.gamma
        )
    if method == "log":
        return implementations.logistic_regression(
            y01, tx, initial_w, args.max_iters, args.gamma
        )
    if method == "reg_log":
        return implementations.reg_logistic_regression(
            y01, tx, args.lambda_, initial_w, args.max_iters, args.gamma
        )
    raise ValueError(f"Unknown method {method}")


def main():
    args = parse_args()
    np.random.seed(1)  # SGD uses np.random: fixed seed for reproducibility

    x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(
        args.data_path, args.sub_sample
    )
    print(f"Data loaded: x_train {x_train.shape}, x_test {x_test.shape}")
    print(f"Fraction of positive labels: {np.mean(y_train == 1):.3f}")

    tx_train, tx_test = preprocessing.build_features(
        x_train,
        x_test,
        nan_threshold=args.nan_threshold,
        fix_codes=args.fix_codes,
        one_hot=args.one_hot,
        # SGD diverges when the rare one-hot categories are standardized
        scale_binary=args.method != "mean_square_sgd",
    )
    print(f"Number of features after preprocessing: {tx_train.shape[1]}")

    # Train on the full training set. The decision threshold is the one that
    # maximizes the F1 score on the training set (the classes are unbalanced,
    # so the default threshold is far from optimal). This is exactly the
    # procedure evaluated by cross-validation in experiments.py.
    w, loss = train(args.method, y_train, tx_train, args)
    threshold, train_f1 = preprocessing.best_threshold(tx_train, w, y_train)
    print(f"[{args.method}] train loss = {loss:.4f}, train F1 = {train_f1:.4f}")

    y_pred = preprocessing.predict_labels(tx_test, w, threshold)
    print(f"Fraction of positive predictions on test: {np.mean(y_pred == 1):.3f}")

    helpers.create_csv_submission(test_ids, y_pred, args.output)
    print(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()
