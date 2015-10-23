import numpy as np

from scipy import linalg


def sufficient_stats(X, y):
    return np.dot(X.T, X), np.dot(X.T, y)
