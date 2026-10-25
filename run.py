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
import copy
import helpers
import implementations
import preprocessing
import matplotlib.pyplot as plt

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

    parser.add_argument(
        "--tune",
        action="store_true",
        help="if activated, search the best hyperparameter for mean square gd, sgd and logistic regression",
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

def tune_hyperparameters(x, y, args, k_fold=5, seed=1):
    """Finds the best hyperparameter and plots the validation curve."""
    if args.method == "least_squares":
        print("Least_squares has no hyperparameter to tune.")
        return args

    print(f"\n---Tuning hyperparameters for {args.method} ---")
    
    # 1. Configuration based on the method
    if args.method in ["ridge", "reg_log"]:
        param_name = "lambda_"
        param_grid = np.logspace(-5, 0, 8) # 12 values between 10^-5 and 1
    else: 
        param_name = "gamma"
        param_grid = np.logspace(-4, 0, 10) # 10 values between 10^-4 and 1

    best_param_val = getattr(args, param_name)
    best_f1 = -1

    feature_kwargs = dict(
        nan_threshold=args.nan_threshold,
        fix_codes=args.fix_codes,
        one_hot=args.one_hot,
        scale_binary=args.method != "mean_square_sgd",
    )

    folds = preprocessing.k_fold_indices(len(y), k_fold, seed)

    # --- NEW: Lists for the plot ---
    mean_train_scores = []
    mean_val_scores = []
    tested_params = []

    for val in param_grid:
        current_args = copy.deepcopy(args)
        if param_name == "lambda_":
            current_args.lambda_ = val
        else:
            current_args.gamma = val
            
        f1_tr_scores = []
        f1_te_scores = []
        
        for k in range(k_fold):
            te_idx = folds[k]
            tr_idx = np.concatenate([folds[i] for i in range(k_fold) if i != k])
            
            tx_tr, tx_te = preprocessing.build_features(x[tr_idx], x[te_idx], **feature_kwargs)
            
            # Training
            w, _ = train(current_args.method, y[tr_idx], tx_tr, current_args)
            threshold, _ = preprocessing.best_threshold(tx_tr, w, y[tr_idx])
            
            # Validation (Test) Score
            y_pred_te = preprocessing.predict_labels(tx_te, w, threshold)
            f1_te_scores.append(preprocessing.f1_score(y[te_idx], y_pred_te))
            
            # Training Score - Needed for a proper validation curve
            y_pred_tr = preprocessing.predict_labels(tx_tr, w, threshold)
            f1_tr_scores.append(preprocessing.f1_score(y[tr_idx], y_pred_tr))
            
        mean_f1_te = np.mean(f1_te_scores)
        mean_f1_tr = np.mean(f1_tr_scores)
        
        # Add to lists for plotting
        mean_val_scores.append(mean_f1_te)
        mean_train_scores.append(mean_f1_tr)
        tested_params.append(val)
        
        print(f"Test {param_name} = {val:.1e}  -> Train F1 = {mean_f1_tr:.4f} | Val F1 = {mean_f1_te:.4f}")
        
        if mean_f1_te > best_f1:
            best_f1 = mean_f1_te
            best_param_val = val

    print(f"Best {param_name} found: {best_param_val:.1e} (F1 = {best_f1:.4f})\n")
    
    # --- NEW: Validation Curve Plot Generation ---
    plt.figure(figsize=(9, 6))
    
    # semilogx so the X-axis is logarithmic (essential for lambda and gamma)
    plt.semilogx(tested_params, mean_train_scores, marker='o', color='blue', linewidth=2, label='Train F1 Score')
    plt.semilogx(tested_params, mean_val_scores, marker='s', color='red', linewidth=2, label='Validation F1 Score')
    
    # Aesthetics
    plt.title(f"Validation curve for method: {args.method}", fontsize=14, pad=15)
    plt.xlabel(f"Hyperparameter {param_name} (log scale)", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    
    # Dashed line to indicate the optimal point
    plt.axvline(x=best_param_val, color='black', linestyle='--', alpha=0.7, 
                label=f'Best {param_name} ({best_param_val:.1e})')
    
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Remove top and right borders for a cleaner look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Smart save (uses the method name)
    filename = f"validation_curve_{args.method}.pdf"
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Plot saved as '{filename}'")
    plt.show()

    # Update arguments with the best found value
    if param_name == "lambda_":
        args.lambda_ = best_param_val
    else:
        args.gamma = best_param_val
        
    return args

def main():
    args = parse_args()
    np.random.seed(1)  # SGD uses np.random: fixed seed for reproducibility

    x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data(
        args.data_path, args.sub_sample
    )
    print(f"Data loaded: x_train {x_train.shape}, x_test {x_test.shape}")
    print(f"Fraction of positive labels: {np.mean(y_train == 1):.3f}")

    if args.tune:
        args = tune_hyperparameters(x_train, y_train, args)

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
