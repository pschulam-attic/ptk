import numpy as np

from scipy import linalg


def sufficient_stats(X, y, C=None):
    n, _ = X.shape
    
    if C is None:
        C = np.eye(n)

    ss1 = np.dot(X.T, linalg.solve(C, X))
    ss2 = np.dot(X.T, linalg.solve(C, y))

    return ss1, ss2


def orthogonal_sufficient_stats(X, X2, y, C=None):
    X_orth = residuals(X2, X, C)
    y_orth = residuals(X2, y, C)
    return sufficient_stats(X_orth, y_orth, C)


def residuals(X, y, C=None):
    ss1, ss2 = sufficient_stats(X, y, C)
    coef = linalg.solve(ss1, ss2)
    return y - np.dot(X, coef)
