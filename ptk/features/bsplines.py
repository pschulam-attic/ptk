import numpy as np

from scipy.interpolate import splev


def evaluate(x, knots, degree):
    '''Evaluate all B-spline bases.

    '''
    c = np.eye(num_bases(knots, degree))
    return np.array(splev(x, (knots, c, degree)))


def num_bases(knots, degree):
    '''The dimension of the implied B-spline space.

    '''
    return len(knots) - degree - 1


def uniform_knots(low, high, num_bases, degree):
    '''Create the standard uniform B-spline knots.

    Parameters
    ----------
    low : The lower bound of the domain.
    high : The upper bound of the domain.
    num_bases : The desired number of bases.
    degree : The degree of the polynomial pieces.

    Returns
    -------
    An array of knots.

    '''
    num_knots = num_bases + degree + 1
    num_interior = num_knots - 2 * (degree + 1)
    
    knots = np.linspace(low, high, num_interior + 2).tolist()
    knots = degree * [low] + knots + degree * [high]
    return np.array(knots)


def pspline_penalty(knots, degree, order=1):
    '''Construct a P-spline penalty matrix for regression.

    '''
    D = np.eye(num_bases(knots, degree))
    D = np.diff(D, order)
    return np.dot(D, D.T)
