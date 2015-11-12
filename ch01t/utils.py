# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os

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
    def __init__(self, threshold=None, accuracy  = None):
        self.threshold = threshold
        self.accuracy = accuracy