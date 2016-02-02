import os

import scipy as sp
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

__author__ = 'ThinkPad'

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")


class Linear_regression(object):
    def main(self):
        x, y = self.load_single_data()
        lr = LinearRegression()

        lr.fit(x, y)
        y_predicted = lr.predict(x)

        fig, ax = plt.subplots()
        ax.set_xlabel("size of city")
        ax.set_ylabel("profits")
        ax.scatter(x, y)
        ax.scatter(x, y_predicted, color='r', marker='s')

        xmin = x.min()
        xmax = x.max()
        # ax.plot([xmin, xmax], [lr.predict(xmin), lr.predict(xmax)], "-", lw=4)

        fig.savefig('Figure2.png')

    def load_single_data(self):
        single_data = sp.genfromtxt(os.path.join(DATA_DIR, "ex1data1.txt"), delimiter=",")
        x = single_data[:, 0]
        y = single_data[:, 1]

        x = np.transpose(np.atleast_2d(x))

        return (x, y)


if __name__ == '__main__':
    Linear_regression().main()
