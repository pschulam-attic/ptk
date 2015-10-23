import numpy as np


def constant(x1, x2):
    n = len(x1)
    m = len(x2)
    K = np.ones((n, m))
    return K


def gaussian(x1, x2, lengthscale):
    r = _abs_distance(x1, x2)
    return np.exp(-0.5 * r**2 / lengthscale**2)


def ornstein_uhlenbeck(x1, x2, lengthscale):
    r = _abs_distance(x1, x2)
    return np.exp(- r / lengthscale)


_root3 = np.sqrt(3)

def matern32(x1, x2, lengthscale):
    r = _abs_distance(x1, x2)
    return ((1 + _root3 * r / lengthscale) *
            (np.exp(- _root3 * r / lengthscale)))


_root5 = np.sqrt(5)

def matern52(x1, x2, lengthscale):
    r = _abs_distance(x1, x2)
    return ((1 + _root5 * r / lengthscale + (5/3) * r**2 / lengthscale**2) *
            (np.exp(- _root5 * r / lengthscale)))


def _abs_distance(x1, x2):
    x1 = x1[:, None]
    x2 = x2[None, :]
    return np.abs(x1 - x2)
