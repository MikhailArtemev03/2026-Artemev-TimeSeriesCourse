import numpy as np

from modules.metrics import ED_distance, norm_ED_distance, DTW_distance
from modules.utils import z_normalize


class PairwiseDistance:
    """
    Distance matrix between time series

    Parameters
    ----------
    metric: distance metric between two time series
            Options: {euclidean, dtw}
    is_normalize: normalize or not time series
    """

    def __init__(self, metric: str = 'euclidean',
                 is_normalize: bool = False) -> None:

        self.metric: str = metric
        self.is_normalize: bool = is_normalize


    @property
    def distance_metric(self) -> str:
        """Return the distance metric"""

        norm_str = ""

        if self.is_normalize:
            norm_str = "normalized "
        else:
            norm_str = "non-normalized "

        return norm_str + self.metric + " distance"



    def _choose_distance(self):
        """
        Choose distance function for calculation of matrix

        Returns
        -------
        dist_func: function reference
        """

        dist_func = None

        if self.metric == 'euclidean':
            if self.is_normalize:
                dist_func = norm_ED_distance
            else:
                dist_func = ED_distance

        elif self.metric == 'dtw':
            dist_func = DTW_distance

        else:
            raise ValueError("Unknown distance metric")

        return dist_func


    def calculate(self, input_data: np.ndarray) -> np.ndarray:
        """
        Calculate distance matrix

        Parameters
        ----------
        input_data: time series set

        Returns
        -------
        matrix_values: distance matrix
        """

        matrix_shape = (
            input_data.shape[0],
            input_data.shape[0]
        )

        matrix_values = np.zeros(shape=matrix_shape)


        dist_func = self._choose_distance()

        if self.is_normalize and self.metric != 'euclidean':
            data = np.array([
                z_normalize(ts)
                for ts in input_data
            ])
        else:
            data = input_data


        k = data.shape[0]

        for i in range(k):
            for j in range(i + 1, k):

                distance = dist_func(
                    data[i],
                    data[j]
                )

                matrix_values[i, j] = distance
                matrix_values[j, i] = distance

        return matrix_values