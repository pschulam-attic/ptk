import numpy as np

from collections import defaultdict


def group_outcomes(identifiers, times, values):
    '''Create a collection of time series from "long format" data.

    Parameters
    ----------
    identifiers : Values associating units with measurements.
    times : Times of each measurement.
    values : The measurements.

    Returns
    -------
    A dictionary mapping unique identifiers to 2-tuples of arrays
    containing the times and values of all measurements belonging to
    that unit.

    '''
    grouped = defaultdict(list)
    for i, x, y in zip(identifiers, times, values):
        grouped[i].append((x, y))

    ordered = {}
    for i, outcomes in grouped.items():
        outcomes = sorted(outcomes, key=lambda xy: xy[0])
        x, y = list(zip(*outcomes))
        ordered[i] = (np.array(x), np.array(y))

    return ordered
