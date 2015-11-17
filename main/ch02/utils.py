# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

__package__ = __name__

import os
import numpy as np
import logging

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")

COLORS = ['g', 'k', 'b', 'm', 'r']
LINESTYLES = ['-', '-', '--', ':', '-']

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)


def getModel(features, labels, feature_names):
    accuracy_results = []
    best_acc = -1.0

    for fi in range(features.shape[1]):
        features_fi = features[:, fi]
        result = Result(feature_names[fi])
        threshold_accuracy = []
        threshold_list = features_fi.copy()
        threshold_list.sort()

        for t in threshold_list:
            pred = features_fi > t
            acc = (pred == labels).mean()
            rev_acc = (pred == ~labels).mean()

            if acc < rev_acc:
                acc = rev_acc
                reverse = True

            else:
                reverse = False

            threshold_accuracy.append(Threshold_accuracy(t, acc))

            if acc > best_acc:
                best_acc = acc
                best_t = t
                best_fi = fi
                best_reverse = reverse

        result.threshold_accuracy = threshold_accuracy
        accuracy_results.append(result)

    # logging.info("best accuracy : %f   for threshold = %f on feature Index : %i, using reverse :%s", best_acc, best_t,
    #              best_fi, best_reverse)

    return accuracy_results, best_fi, best_t, best_reverse


def predict(model, features):
    accuracy_results, best_fi, best_t, best_reverse = model

    if best_reverse == False:
        return features[:, best_fi] > best_t
    else:
        return features[:, best_fi] < best_t


def accuracy(model, features, labels):
    prediction = predict(model, features)

    return np.mean(prediction == labels)


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
