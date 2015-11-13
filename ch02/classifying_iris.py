from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import logging
from  ch01.utils import DATA_DIR, CHART_DIR, COLORS, LINESTYLES, Result, Threshold_accuracy
from matplotlib.font_manager import FontProperties

data = load_iris()

logging.basicConfig(format=' %(message)s', level=logging.INFO)


class IrisClassification():
    global features, feature_names, target, target_names

    def __init__(self):
        self.getData()

    def main(self):
        self.getData()
        self.drawFig1()
        self.buildFirstClassificationModel()
        self.drawFig2()

    def getData(self):
        global features, feature_names, target, target_names
        features = data.data
        feature_names = data.feature_names
        target = data.target
        target_names = data.target_names
        logging.info("Features : %s", feature_names)

    def drawFig1(self):
        global features, feature_names, target, target_names
        fig, axes = plt.subplots(2, 3)
        plt.title("Iris repartition per parameters")

        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        # Set up 3 different pairs of (color, marker)
        color_markers = [
            ('r', '>'),
            ('g', 'o'),
            ('b', 'x'),
        ]
        for i, (p0, p1) in enumerate(pairs):
            ax = axes.flat[i]

            for t in range(3):
                # Use a different color/marker for each class `t`
                c, marker = color_markers[t]
                # limit the features list on indexes that are True on the target List for the value == t
                # and take the px-th parameter.
                ax.scatter(features[target == t, p0], features[
                    target == t, p1], marker=marker, c=c)
            ax.set_xlabel(feature_names[p0])
            ax.set_ylabel(feature_names[p1])
            ax.set_xticks([0, max(features[target == t, p0])])
            ax.set_yticks([0, max(features[target == t, p1])])
            # plt.legend(["%s" % target_name for target_name in target_names], loc="upper left")

        fig.tight_layout()
        fig.savefig('figure1.png')

    def buildFirstClassificationModel(self):
        global features, feature_names, target, target_names
        labels = target_names[target]
        is_setosa = labels == "setosa"

        # get then max petal length of setosa and min petal length of non-setosa
        plength_setosa = features[is_setosa, 2]
        plength_non_setosa = features[~is_setosa, 2]
        maxSetosa = plength_setosa.max()
        min_nonSetosa = plength_non_setosa.min()
        logging.info("max plength for setosa :  %f", maxSetosa)
        logging.info("min plength for non-setosa :  %f", min_nonSetosa)


        # select Features & Labels for non Setosa:
        features_nonSetosa = features[~(labels == "setosa")]
        labels_nonSetosa = labels[labels != 'setosa']
        is_virginica = (labels_nonSetosa == "virginica")



        # loop on all  4 Features for non Setosa  and define which one define at best the
        best_acc = -1.0

        # store the results in a list of 6 differents series
        accuracy_results = []

        for fi in range(features_nonSetosa.shape[1]):
            feature_i = features_nonSetosa[:, fi]

            result = Result(feature_names[fi])
            threshold_accuracy = []

            for t in feature_i:
                pred = feature_i > t
                acc = (pred == is_virginica).mean()
                rev_acc = (pred == ~is_virginica).mean()

                if acc > rev_acc:
                    reverse = False
                else:
                    reverse = True
                    acc = rev_acc

                threshold_accuracy.append(Threshold_accuracy(t, acc))

                if acc > best_acc:
                    best_acc = acc
                    best_fi = fi
                    best_t = t
                    best_reverse = reverse
            result.threshold_accuracy = threshold_accuracy
            accuracy_results.append(result)
        logging.info("Best feature : %s - Best threshold : %f - Is Reverse : %s - Best Accuracy : %f",
                     feature_names[fi], best_t,
                     best_reverse, best_acc)

        self.drawAccuracyComparisionChart(accuracy_results)

        # test the threshold on a sample
        example = np.array([1.6, 2.5, 4.3, 2.6])
        self.is_virginica_test(best_fi, best_t, best_reverse, example)

    def is_virginica_test(self, fi, t, reverse, example):
        'Apply threshold model to a new example'
        example_fi = example[fi]
        test = example_fi > t
        if reverse:
            test = not test
        acc = test.mean()
        return acc

    def drawAccuracyComparisionChart(self, accuracy_results):

        plt.figure(num=None, figsize=(8, 6))
        plt.clf()
        fontP = FontProperties()
        fontP.set_size('small')

        for accuracy_result, color, linestyle in zip(accuracy_results, COLORS, LINESTYLES):
            array = Threshold_accuracy().convertToValues(accuracy_result)
            plt.plot(array[:, 0], array[:, 1],
                     linestyle=linestyle, linewidth=0.5, c=color)

        plt.legend(["iris type =%s" % accuracy_result.serieName for accuracy_result in accuracy_results],
                   prop=fontP)
        plt.savefig("FirstClassificationChart")

    def drawFig2(self):
        self.getData()
        global features, feature_names, target, target_names

        logging.info("Target Names  = %s", target_names)


        # define two thresholds
        t1 = 1.65
        t2 = 2

        # define the two axis : petal_width (f0) and petal_length(f1)
        f0, f1 = 3, 2

        area1c = (1., 1, 1)
        area2c = (.7, .7, .7)


        # remove the setosa from features
        labels = target_names[target]
        is_setosa = labels == "setosa"
        is_versicolor = labels == "versicolor"
        is_virginica = labels == "virginica"

        features_wo_setosa = features[~is_setosa]
        features_virginica = features[is_virginica]
        features_versicolor = features[is_versicolor]

        # define the axis with 10% margin
        x0 = features_wo_setosa[:, f0].min() * 0.9
        x1 = features_wo_setosa[:, f0].max() * 1.1

        y0 = features_wo_setosa[:, f1].min() * 0.9
        y1 = features_wo_setosa[:, f1].max() * 1.1

        fig, ax = plt.subplots()

        # draw the two thresholds lines :
        ax.fill_between([t1, x1], [y0, y0], [y1, y1], color=area1c)
        ax.fill_between([x0, t1], [y0, y0], [y1, y1], color=area2c)

        # draw the two t1 and t2 lines
        ax.plot([t1, t1], [y0, y1], 'k--', lw=2)
        ax.plot([t2, t2], [y0, y1], 'k:', lw=2)

        # scatter the two series versicolor and virginica
        ax.scatter(features_virginica[:, f0], features_virginica[:, f1], c='b', marker='o', s=40, label="virginica")
        ax.scatter(features_versicolor[:, f0], features_versicolor[:, f1], c='r', marker='x', s=40, label="versicolor")

        ax.set_ylim(y0, y1)
        ax.set_xlim(x0, x1)
        ax.set_xlabel(feature_names[f0])
        ax.set_ylabel(feature_names[f1])
        ax.legend(loc='upper left')

        fig.tight_layout()

        fig.savefig('figure2.png')


if __name__ == "__main__":
    IrisClassification().main()
