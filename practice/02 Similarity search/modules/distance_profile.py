import numpy as np

from modules.utils import z_normalize
from modules.metrics import ED_distance, norm_ED_distance



def brute_force(ts: np.ndarray, query: np.ndarray, is_normalize: bool = True) -> np.ndarray:
    """
    Calculate the distance profile using the brute force algorithm

    Parameters
    ----------
    ts: time series
    query: query, shorter than time series
    is_normalize: normalize or not time series and query

    Returns
    -------
    dist_profile: distance profile between query and time series
    """

    n = len(ts)
    m = len(query)
    N = n-m+1

    dist_profile = np.zeros(shape=(N,))

    # INSERT YOUR CODE

    if is_normalize:
        query = z_normalize(query)

    for i in range(N):
        subsequence = ts[i:i + m]

        if is_normalize:
            if np.std(subsequence) == 0:
                dist_profile[i] = np.inf
                continue

            subsequence = z_normalize(subsequence)

        dist_profile[i] = ED_distance(query, subsequence)

    return dist_profile