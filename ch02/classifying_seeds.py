__author__ = 'JW'

import utils
import load
import logging
import numpy as np
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import plotChart

logging.basicConfig(format=' %(message)s', level=logging.INFO)
global features, labels


class SeedsClassification():
    def __init__(self):
        global features, labels
        features, labels = load.load_dataset("seeds")
        load.test_load(features, labels)
        logging.info("data loaded successfully")

    def main(self):
        global features, labels
        self.knn_with_kFold()
        self.knn_manual()
        plotChart.drawFigure(features, labels, 1, [0, 2], "fig_no_normalization_1_neighbor")
        features = (features - features.mean(0)) / features.std(0)
        plotChart.drawFigure(features, labels, 1, [0, 2], "fig_manual_normalization_1_neighbor")

    # Create a model on knn with One neighbours and and calculate the accuracy on testing data with cross_validation on 5 folds
    def knn_with_kFold(self):
        classifier = KNeighborsClassifier(n_neighbors=1)
        kFold = cross_validation.KFold(len(features), n_folds=5, shuffle=True)
        accuracy = 0.0
        globalaccuracy = 0.0

        for training, testing in kFold:
            model = classifier.fit(features[training], labels[training])
            prediction = classifier.predict(features[testing])
            accuracy = np.mean(prediction == labels[testing])
            globalaccuracy += accuracy
            print("accuracy {0:.1%}").format(accuracy)
        print("global accuracy {0:.1%}").format(globalaccuracy / 5)

    def knn_manual(self):
        classifier = KNeighborsClassifier(n_neighbors=1)
        training = np.ones(len(features), bool)
        training[-10:] = False
        testing = ~training

        split_idx = 0.90 * len(features)
        shuffled = sp.random.permutation(list(range(len(features))))
        testing = sorted(shuffled[:split_idx])

        training = sorted(shuffled[split_idx:])
        print "training : {0}".format(training)
        print "testing : {0}".format(testing)
        model = classifier.fit(features[training], labels[training])
        prediction = classifier.predict(features[testing])
        accuracy = np.mean(prediction == labels[testing])
        print("accuracy manual : {0:.1%}").format(accuracy)

    def drawKnnWithSklearn(self):
        pass


if __name__ == '__main__':
    SeedsClassification().main()
