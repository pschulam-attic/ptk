import numpy as np

from scipy import linalg


def least_squares(X, y, C=None, P=0.0):
    ss1, ss2 = sufficient_stats(X, y, C)
    ss1 += P

    coef = linalg.solve(ss1, ss2)
    residuals = y - np.dot(X, coef)
    
    return coef, residuals


def block_least_squares(Xs, ys, Cs=None, P=0.0):
    num_blocks = len(Xs)

    ss1 = 0.0
    ss2 = 0.0

    for i in range(num_blocks):
        X = Xs[i]
        y = ys[i]

        if Cs is None:
            C = None
        else:
            C = Cs[i]

        ss1i, ss2i = sufficient_stats(X, y, C)

        ss1 += ss1i
        ss2 += ss2i

    coef = linalg.solve(ss1 + P, ss2)

    residuals = [y - np.dot(X, coef) for X, y in zip(Xs, ys)]

    return coef, residuals


def sufficient_stats(X, y, C=None):
    if C is None:
        ss1 = np.dot(X.T, X)
        ss2 = np.dot(X.T, y)
    else:
        ss1 = np.dot(X.T, linalg.solve(C, X))
        ss2 = np.dot(X.T, linalg.solve(C, y))
        
    return ss1, ss2

