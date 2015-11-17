import numpy as np
from  utils import CHART_DIR
import os
from sklearn.neighbors import KNeighborsClassifier
import load
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',
]

data = load.load_dataset("seeds")
# np.set_printoptions(threshold=np.nan)

def drawFigure(features, labels, neighbors=1, parameters=[], figName="no_name"):
    names = sorted(set(labels))
    labels = np.array([names.index(ell) for ell in labels])

    idX, idY = parameters[0], parameters[1]

    print("Xaxis :{0} - Yaxis : {1}").format(idX, idY)

    # define lower and upper limit on both axis (x=area, y =compactness)
    x0, y0 = features[:, idX].min() * 0.9, features[:, idY].min() * 0.9
    x1, y1 = features[:, idX].max() * 1.1, features[:, idY].max() * 1.1

    # create a meshgrid resulting of 2 X/Y-Linespaces
    X = np.linspace(x0, x1, 50)
    Y = np.linspace(y0, y1, 50)

    featuresMesh = convertLinspaceToFeatures(X, Y)

    X, Y = np.meshgrid(X, Y)

    Xravel = X.ravel()
    Yravel = Y.ravel()

    vStacked = np.vstack([Xravel, Yravel])

    vStackedTransposed = vStacked.T

    # create a predicate resulting of a model
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(features[:, (idX, idY)], labels)

    predictionT = classifier.predict(vStackedTransposed)
    predictionManual = classifier.predict(featuresMesh)

    CT = predictionT.reshape(X.shape)
    CManual = predictionManual.reshape(X.shape)

    drawChart(CT, X, Y, x0, x1, y0, y1, figName + "_T")
    drawChart(CManual, X, Y, x0, x1, y0, y1, figName + "_Manual")


def drawChart(C, X, Y, x0, x1, y0, y1, figName):
    # create a ListedColormap
    cmap = ListedColormap(["r", "g", "b"])
    # plot the prediction area using pcolormesh
    fig, ax = plt.subplots()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[2])
    ax.pcolormesh(X, Y, C, cmap=cmap)
    # iter on the three type of seeds and scatter them.
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, figName + ".png"))


def convertLinspaceToFeatures(linspaceX, linespaceY):
    matrix = []
    for y in range(len(linespaceY)):
        for x in range(len(linspaceX)):
            matrix.append([linspaceX[x], linespaceY[y]])
    return np.asarray(matrix)
