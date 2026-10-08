import numpy as np




def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e**2)

def compute_error(y, tx, w):
    """Compute error"""
    return y - tx.dot(w)





def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = compute_error(y,tx,w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def mean_squared_error_gd(y, tx, initial_w,max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: optimal loss
        w: optimal parameter
    """

    
    w = initial_w

    grad, err = compute_gradient(y, tx, w)
    loss = calculate_mse(err)
    
    for i in range(max_iters):

        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)

        w = w - gamma * grad

        

    return w, loss




def mean_squared_error_sgd(y, tx, initial_w, batch_size , max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
 
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: optimal loss
        w: optimal parameter
    """

    # Define parameters to store w and loss
    
    w = initial_w
    grad, err = compute_gradient(y, tx, w)
    loss = calculate_mse(err)

    for n_iter in range(max_iters):
        ### SOLUTION
        for y_batch, tx_batch in n_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            # compute a stochastic gradient and loss
            grad, e = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            
            loss = calculate_mse(e)
            # store w and loss
            

        
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
        returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    e = compute_error(y,tx,w)
    mse =  calculate_mse(e)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return (1 + np.exp(-t))**(-1)



def calculate_hessian(y, tx, w):
    """return the Hessian of the sigmoid.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)
    """
    z = tx.dot(w)
    pred = sigmoid(z)
    S = pred * (1 - pred)
    H = tx.T.dot(S * tx)/len(y)
    return H 


def calculate_sigmoid_gradient(y, tx, w):
    """compute the gradient of loss according to the sigmoid function.
    
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)
    
        Returns:
            a vector of shape (D, 1)
    """
    z = tx.dot(w)
    pred = sigmoid(z)
    e = pred - y
    grad = tx.T.dot(e) / len(y)  
    return grad 



def logistic_regression(y, tx, initial_w,max_iters, gamma):
    """Implementation of logistic regression
    
        Args:
            y: numpy array of shape=(N, )
            tx: numpy array of shape=(N,2)
            initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
            max_iters: a scalar denoting the total number of iterations of SGD
            gamma: a scalar denoting the stepsize
    
        Returns:
            loss: optimal loss value
            w: optimal w

    """
    h = calculate_hessian(y, tx, initial_w)
    gr = calculate_sigmoid_gradient(y,tx,w)
    w = initial_w
    for i in range(max_iters):
        h = calculate_hessian(y, tx, w)
        w = w - gamma * (np.linalg.solve(h, gr))
        e = compute_error(y, tx, w)
        loss = calculate_mse(e)

    return loss, w 












