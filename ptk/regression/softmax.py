import numpy as np

from scipy.misc import logsumexp


def predict(X, W):
    X = np.atleast_2d(X)
    n, p = X.shape
    k = W.shape[0] + 1
    Y = np.zeros((n, k))
    Z = np.c_[ np.zeros(n), np.dot(X, W.T) ]
    P, _ = _softmax(Z)
    return P.squeeze()


def _softmax(z):
    Z = np.atleast_2d(z)
    S = logsumexp(z, axis=2)
    f = np.exp(Z - S[:, None])

    n, k = f.shape
    g = np.zeros((n, k, k))
    for i in range(n):
        p = f[i]
        g[i] = np.diag(p) - p[:, None] * p[None, :]

    return f.squeeze(), g.squeeze()
