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
    
    ed_dist = 0

    # INSERT YOUR CODE
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
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
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

    dtw_dist = 0

    # INSERT YOUR CODE
    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)

    if ts1.ndim != 1 or ts2.ndim != 1:
        raise ValueError("Временные ряды должны быть одномерными")

    if len(ts1) != len(ts2) or len(ts1) == 0:
        raise ValueError("Ряды должны быть непустыми и одинаковой длины")

    if not 0 <= r <= 1:
        raise ValueError("Параметр r должен находиться от 0 до 1")

    n = len(ts1)
    radius = int(r * n)

    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(n, i + radius)

        for j in range(j_start, j_end + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            D[i, j] = cost + min(
                D[i - 1, j],
                D[i, j - 1],
                D[i - 1, j - 1]
            )

    dtw_dist = float(D[n, n])

    return dtw_dist
