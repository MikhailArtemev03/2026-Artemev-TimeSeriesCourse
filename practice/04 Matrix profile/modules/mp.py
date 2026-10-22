import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config


def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile

    Parameters
    ----------
    ts1: the first time series
    m: the subsequence length
    exclusion_zone: exclusion zone
    ts2: the second time series

    Returns
    -------
    output: the matrix profile structure
            (matrix profile, matrix profile index, subsequence length, exclusion zone, the first and second time series)
    """

    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / 4))

    old_denom = config.STUMPY_EXCL_ZONE_DENOM

    if ts2 is None:
        config.STUMPY_EXCL_ZONE_DENOM = m / exclusion_zone

    try:
        mp = stumpy.stump(
            np.asarray(ts1, dtype=float),
            m=m,
            T_B=None if ts2 is None else np.asarray(ts2, dtype=float),
            ignore_trivial=(ts2 is None)
        )
    finally:
        config.STUMPY_EXCL_ZONE_DENOM = old_denom

    return {'mp': mp[:, 0],
            'mpi': mp[:, 1],
            'm' : m,
            'excl_zone': exclusion_zone,
            'data': {'ts1' : ts1, 'ts2' : ts2}
            }