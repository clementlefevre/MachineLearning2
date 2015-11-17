__author__ = 'JW'

import numpy as np


def euclidian_distance(V1, V2):
    distance = 0.0
    for i in range(V1.shape[0]):
        distance += (V1[i] - V2[i]) ** 2

    return np.sqrt([distance])
