import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # INSERT YOUR CODE
    distances = np.asarray(matrix_profile['mp'], dtype=float).copy()
    indices = np.asarray(matrix_profile['mpi'], dtype=int)
    excl_zone = matrix_profile['excl_zone']

    distances[~np.isfinite(distances)] = np.inf
    distances[indices < 0] = np.inf

    for i in range(top_k):
        idx = int(np.argmin(distances))

        if not np.isfinite(distances[idx]):
            break

        nn_idx = int(indices[idx])

        motifs_idx.append((min(idx, nn_idx), max(idx, nn_idx)))
        motifs_dist.append(float(distances[idx]))

        apply_exclusion_zone(distances, idx, excl_zone, np.inf)
        apply_exclusion_zone(distances, nn_idx, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
