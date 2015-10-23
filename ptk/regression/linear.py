import numpy as np

from scipy import linalg


def penalized_mle(X, y, P):
    ss1, ss2 = sufficient_stats(X, y)
    ss1 += P
    return linalg.solve(ss1, ss2)


def sufficient_stats(X, y):
    return np.dot(X.T, X), np.dot(X.T, y)

