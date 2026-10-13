"""Data cleaning, evaluation metrics and validation utilities (NumPy only)."""

import numpy as np

# ---------------------------------------------------------------------------
# Feature cleaning
# ---------------------------------------------------------------------------


def replace_special_codes(x):
    """Replace the BRFSS special answer codes by their meaning.

    In the BRFSS codebook, the largest codes of a question are not real values:
    7 / 77 / 777 / 7777 mean "don't know", 9 / 99 / 999 / 9999 mean "refused"
    and 88 / 888 / 8888 mean "none" (e.g. PHYSHLTH = 88 means 0 bad days).
    A column is treated as coded when its maximum is 9, 99, 999 or 9999; a
    code is then only replaced when it is isolated (the value just below it
    is absent), so that e.g. EMPLOY1 = 7 ("retired") is kept.

    Args:
        x: numpy array of shape (N, D), raw features

    Returns:
        a copy of x where "don't know" / "refused" are NaN and "none" is 0
    """
    x = x.copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        values = np.unique(col[~np.isnan(col)])
        if values.size == 0:
            continue
        vmax = values.max()
        n_digits = len(str(int(vmax)))
        if vmax != 10**n_digits - 1:
            continue
        repdigit = (10**n_digits - 1) // 9  # 1, 11, 111, 1111
        dont_know, none, refused = 7 * repdigit, 8 * repdigit, 9 * repdigit

        col[col == refused] = np.nan
        if dont_know - 1 not in values:
            col[col == dont_know] = np.nan
        if n_digits > 1 and none - 1 not in values:
            col[col == none] = 0
    return x


def build_features(
    x_train,
    x_test,
    nan_threshold=0.5,
    fix_codes=False,
    one_hot=False,
    nan_indicators=False,
    max_categories=10,
    poly_degree=1,
    scale_binary=True,
):
    """Clean the raw features and build the design matrices.

    Every statistic is computed on the training set only and then applied to
    both sets, so that no information leaks from the test set. With all the
    options set to False this is our baseline preprocessing.

    Steps:
        0. (fix_codes) replace the BRFSS special codes, see replace_special_codes
        1. (one_hot) one-hot encode the categorical columns, i.e. the ones with
           at most `max_categories` distinct values. A missing value is its own
           category, so these columns are kept whatever their NaN fraction.
           The most frequent category is dropped to avoid collinearity.
        2. drop the numerical columns with a NaN fraction above `nan_threshold`
        3. (nan_indicators) add a 0/1 column "value is missing" for every kept
           numerical column that has missing values
        4. replace the remaining NaN by the median of the column, and add the
           powers 2..poly_degree of the numerical columns
        5. drop the constant columns
        6. standardize each column (zero mean, unit variance); if not
           scale_binary, the binary columns created in 1. and 3. are kept 0/1
        7. add a bias column of ones

    Args:
        x_train: numpy array of shape (N, D), raw training features
        x_test: numpy array of shape (M, D), raw test features
        nan_threshold: maximal fraction of NaN allowed in a numerical column
        fix_codes, one_hot, nan_indicators: enable the optional steps
        max_categories: maximal number of values of a categorical column
        poly_degree: also add the powers 2..poly_degree of the numerical columns
        scale_binary: also standardize the binary columns. This conditions the
            problem better for GD, but a rare category then takes huge values
            (about 1/sqrt(frequency)) on a few samples, which makes SGD diverge

    Returns:
        tx_train: numpy array of shape (N, D' + 1)
        tx_test: numpy array of shape (M, D' + 1)
    """
    if fix_codes:
        x_train, x_test = replace_special_codes(x_train), replace_special_codes(x_test)

    feats_tr, feats_te = [], []
    nan_frac = np.mean(np.isnan(x_train), axis=0)
    numerical = []
    for j in range(x_train.shape[1]):
        col_tr, col_te = x_train[:, j], x_test[:, j]
        values, counts = np.unique(col_tr[~np.isnan(col_tr)], return_counts=True)
        if one_hot and 0 < values.size <= max_categories:
            # missing values are encoded as an extra category
            if np.isnan(col_tr).any():
                values = np.append(values, np.nan)
                counts = np.append(counts, np.isnan(col_tr).sum())
            values = np.delete(values, np.argmax(counts))
            for v in values:
                if np.isnan(v):
                    feats_tr.append(np.isnan(col_tr))
                    feats_te.append(np.isnan(col_te))
                else:
                    feats_tr.append(col_tr == v)
                    feats_te.append(col_te == v)
        elif nan_frac[j] <= nan_threshold:
            numerical.append(j)

    x_tr = x_train[:, numerical]
    x_te = x_test[:, numerical]

    if nan_indicators:
        has_nan = np.isnan(x_tr).any(axis=0)
        feats_tr += list(np.isnan(x_tr[:, has_nan]).T)
        feats_te += list(np.isnan(x_te[:, has_nan]).T)

    # median imputation
    medians = np.nanmedian(x_tr, axis=0)
    x_tr = np.where(np.isnan(x_tr), medians, x_tr)
    x_te = np.where(np.isnan(x_te), medians, x_te)

    # polynomial expansion of the numerical columns (centered by the median
    # first, otherwise x and x**2 are almost collinear for positive features)
    x_tr, x_te = x_tr - medians, x_te - medians
    x_tr = np.hstack([x_tr**d for d in range(1, poly_degree + 1)])
    x_te = np.hstack([x_te**d for d in range(1, poly_degree + 1)])

    n_numerical = x_tr.shape[1]
    if feats_tr:
        x_tr = np.c_[x_tr, np.array(feats_tr, dtype=float).T]
        x_te = np.c_[x_te, np.array(feats_te, dtype=float).T]

    # remove constant columns (they carry no information and break the std)
    std = np.std(x_tr, axis=0)
    non_constant = std > 0
    x_tr, x_te, std = x_tr[:, non_constant], x_te[:, non_constant], std[non_constant]

    # standardization (binary columns get mean 0 and std 1 if not scaled)
    mean = np.mean(x_tr, axis=0)
    if not scale_binary:
        is_binary = np.arange(len(non_constant))[non_constant] >= n_numerical
        mean[is_binary], std[is_binary] = 0, 1
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    return add_bias(x_tr), add_bias(x_te)


def add_bias(x):
    """Prepend a column of ones to x."""
    return np.c_[np.ones(x.shape[0]), x]


# ---------------------------------------------------------------------------
# Labels, predictions and metrics
# ---------------------------------------------------------------------------


def to_01(y):
    """Map labels from {-1, 1} to {0, 1}."""
    return (y + 1) / 2


def predict_labels(tx, w, threshold=0.0):
    """Predict labels in {-1, 1} by thresholding the linear score tx w.

    For logistic regression a threshold of 0 on the score corresponds to a
    probability of 0.5.
    """
    return np.where(tx.dot(w) > threshold, 1, -1)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred):
    """F1 score of the positive class (labels in {-1, 1})."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == -1))
    fn = np.sum((y_pred == -1) & (y_true == 1))
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def best_threshold(tx, w, y, n_candidates=200):
    """Find the threshold on the score tx w that maximizes the F1 score.

    The classes are very unbalanced, so the default threshold is far from
    optimal for F1.

    Returns:
        threshold, f1: the best threshold and the F1 it reaches on (tx, y)
    """
    scores = tx.dot(w)
    candidates = np.quantile(scores, np.linspace(0.5, 0.995, n_candidates))
    f1s = [f1_score(y, np.where(scores > t, 1, -1)) for t in candidates]
    i = int(np.argmax(f1s))
    return candidates[i], f1s[i]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def train_val_split(n, val_ratio=0.2, seed=1):
    """Return shuffled index arrays (train_idx, val_idx) for n samples."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    return idx[n_val:], idx[:n_val]


def k_fold_indices(n, k_fold, seed=1):
    """Split n shuffled indices into k_fold folds.

    Returns:
        numpy array of shape (k_fold, n // k_fold)
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold_size = n // k_fold
    return np.array([idx[i * fold_size : (i + 1) * fold_size] for i in range(k_fold)])
