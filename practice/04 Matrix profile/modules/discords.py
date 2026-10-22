import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # INSERT YOUR CODE
    distances = np.asarray(matrix_profile['mp'], dtype=float).copy()
    indices = np.asarray(matrix_profile['mpi'], dtype=int)
    excl_zone = matrix_profile['excl_zone']

    distances[~np.isfinite(distances)] = -np.inf
    distances[indices < 0] = -np.inf

    for i in range(top_k):
        idx = int(np.argmax(distances))

        if not np.isfinite(distances[idx]):
            break

        discords_idx.append(idx)
        discords_dist.append(float(distances[idx]))
        discords_nn_idx.append(int(indices[idx]))

        apply_exclusion_zone(distances, idx, excl_zone, -np.inf)
        
    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
