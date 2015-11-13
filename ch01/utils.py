# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")

COLORS = ['g', 'k', 'b', 'm', 'r']
LINESTYLES = ['-', '-', '--', ':', '-']

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)


class Result():
    def __init__(self, serieName=None, threshold_accuracy=[]):
        self.serieName = serieName
        self.threshold_accuracy = threshold_accuracy


class Threshold_accuracy():
    def __init__(self, threshold=None, accuracy=None):
        self.threshold = threshold
        self.accuracy = accuracy

    def convertToValues(self, accuracy_result):
        array = np.array([[0, 0]])
        for i in range(len(accuracy_result.threshold_accuracy)):
            value = np.array([[accuracy_result.threshold_accuracy[i].threshold,
                               accuracy_result.threshold_accuracy[i].accuracy]])
            array = np.append(array, value
                              , axis=0)
        array = np.delete(array, 0, 0)
        return array[array[:, 0].argsort()]
