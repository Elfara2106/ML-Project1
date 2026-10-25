"""Implementations of the 6 basic ML methods required for project 1.

All functions return (w, loss) where w is the last weight vector and loss the
corresponding cost, computed on the full dataset and without penalty term.
"""
        
import numpy as np


# ---------------------------------------------------------------------------
# Linear regression helpers
# ---------------------------------------------------------------------------


def calculate_mse(e):
    """Calculate the mse (with the 1/2 factor) for the error vector e."""
    return 1 / 2 * np.mean(e**2)


def compute_error(y, tx, w):
    """Compute the error vector e = y - tx w."""
    return y - tx.dot(w)


def compute_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        grad: numpy array of shape (D,), the gradient of the loss at w.
        err: numpy array of shape (N,), the error vector y - tx w.
    """
    err = compute_error(y, tx, w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.

    Takes as input two iterables (the output desired values 'y' and the input
    data 'tx') and yields `num_batches` mini-batches of `batch_size` matching
    elements from `y` and `tx`. Data is randomly sampled to avoid the ordering
    of the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)
    batch_size = min(data_size, batch_size)
    # Maximum amount of non-overlapping batches that can be extracted.
    max_batches = int(data_size / batch_size)
    # Points that would be excluded if no overlap is allowed.
    remainder = data_size - max_batches * batch_size

    if shuffle:
        # Random start index of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Random offset so that the remainder points can also be sampled
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # Without shuffling the batches are taken in a circular order
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start_index in idxs:
        end_index = start_index + batch_size
        yield y[start_index:end_index], tx[start_index:end_index]


# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent (GD).

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), initial weights
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: numpy array of shape (D,), last weight vector
        loss: scalar, MSE loss at w
    """
    w = initial_w
    for _ in range(max_iters):
        grad, _ = compute_gradient(y, tx, w)
        w = w - gamma * grad

    loss = calculate_mse(compute_error(y, tx, w))
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (SGD), batch size 1.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), initial weights
        max_iters: number of SGD steps (one sample per step)
        gamma: step size

    Returns:
        w: numpy array of shape (D,), last weight vector
        loss: scalar, MSE loss at w on the full dataset
    """
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # gradient on a single sample, evaluated at the current w
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad

    loss = calculate_mse(compute_error(y, tx, w))
    return w, loss


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)

    Returns:
        w: numpy array of shape (D,), optimal weights
        loss: scalar, MSE loss at w
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        # tx is rank deficient (collinear features): the solution is not
        # unique, take the one of minimal norm given by the pseudo-inverse
        w = np.linalg.pinv(a).dot(b)
    loss = calculate_mse(compute_error(y, tx, w))
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Minimizes 1/(2N) ||y - tx w||^2 + lambda_ ||w||^2.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)
        lambda_: scalar, regularization parameter

    Returns:
        w: numpy array of shape (D,), optimal weights
        loss: scalar, MSE loss at w (without the penalty term)
    """
    a_i = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + a_i
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = calculate_mse(compute_error(y, tx, w))
    return w, loss


# ---------------------------------------------------------------------------
# Logistic regression (labels y in {0, 1})
# ---------------------------------------------------------------------------


def sigmoid(t):
    """Apply the sigmoid function on t (numerically stable form).

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    # 1 / (1 + exp(-t)) == 0.5 * (1 + tanh(t / 2)), which never overflows
    return 0.5 * (1 + np.tanh(0.5 * t))


def calculate_sigmoid_gradient(y, tx, w):
    """Compute the gradient of the logistic loss at w.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        numpy array of shape (D,)
    """
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y) / len(y)


def compute_log_loss(y, tx, w):
    """Compute the mean negative log-likelihood of the logistic model.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D,)

    Returns:
        scalar loss
    """
    z = tx.dot(w)
    # log(1 + exp(z)) computed without overflow
    return np.mean(np.logaddexp(0, z) - y * z)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: numpy array of shape (N,), labels in {0, 1}
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,), initial weights
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: numpy array of shape (D,), last weight vector
        loss: scalar, logistic loss at w
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * calculate_sigmoid_gradient(y, tx, w)

    loss = compute_log_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Minimizes the logistic loss + lambda_ ||w||^2.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}
        tx: numpy array of shape (N, D)
        lambda_: scalar, regularization parameter
        initial_w: numpy array of shape (D,), initial weights
        max_iters: number of GD steps
        gamma: step size

    Returns:
        w: numpy array of shape (D,), last weight vector
        loss: scalar, logistic loss at w (without the penalty term)
    """
    w = initial_w
    for _ in range(max_iters):
        grad = calculate_sigmoid_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

    loss = compute_log_loss(y, tx, w)
    return w, loss
