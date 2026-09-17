import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size

    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)

    d = np.full((n + 1, n + 1), np.inf)

    d[0, 0] = 0

    window = int(r * n)

    for i in range(1, n + 1):

        j_start = max(1, i - window)
        j_end = min(n, i + window)

        for j in range(j_start, j_end + 1):

            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            d[i, j] = cost + min(
                d[i - 1, j],
                d[i, j - 1],
                d[i - 1, j - 1]
            )

    dtw_dist = d[n, n]

    return dtw_dist