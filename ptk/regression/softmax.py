import numpy as np

from scipy.optimize import minimize
from scipy.misc import logsumexp


def fit(X, Y, W0=None):
    ny, k = Y.shape
    nx, d = X.shape
    assert nx == ny

    if W0 is None:
        W0 = np.random.normal(size(k - 1, d))

    def loglik(w):
        W = w.reshape(W0.shape)
        P = predict(X, W)
        return np.sum(Y * np.log(P))

    def loglik_grad(w):
        W = w.reshape(W0.shape)
        Z = _linear_scores(X, W)
        P, dP_dZ = _softmax(Z)

        G = np.zeros((W.shape[0] + 1, W.shape[1]))
        for i in range(1, k):
            for j in range(k):
                w_ij = Y[:, j] / P[:, j] * dP_dZ[:, i, j]
                G[i] += np.sum(w_ij[:, None] * X, axis=0)

        return G[1:, :].ravel()

    def f(w):
        return -loglik(w) / nx

    def g(w):
        return -loglik_grad(w) / nx

    solution = minimize(f, W0.ravel(), jac=g, method='BFGS')
    W = solution['x'].reshape(W0.shape)
    return W


def predict(X, W):
    Z = _linear_scores(X, W)
    P, _ = _softmax(Z)
    return P.squeeze()


def _linear_scores(X, W):
    X = np.atleast_2d(X)
    n, p = X.shape
    k = W.shape[0] + 1
    Z = np.c_[ np.zeros(n), np.dot(X, W.T) ]
    return Z


def _softmax(z):
    Z = np.atleast_2d(z)
    S = logsumexp(z, axis=1)
    f = np.exp(Z - S[:, None])

    n, k = f.shape
    g = np.zeros((n, k, k))
    for i in range(n):
        p = f[i]
        g[i] = np.diag(p) - p[:, None] * p[None, :]

    return f.squeeze(), g.squeeze()
