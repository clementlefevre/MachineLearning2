import calendar

from ..config import CHART_DIR
from ..model.site import SEASONS
import numpy as np
import os

__author__ = 'ThinkPad'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotFigure(x, y, season, day, siteName, parameter):
    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.scatter(x, y, s=4, color='r', marker='s')
    plt.title('{0} - {1} - {2}.png'.format(siteName, SEASONS.get(season), calendar.day_name[day - 1]))
    plt.xlabel(parameter)
    plt.ylabel("Optimized In")
    # pdb.set_trace()
    fname = '{0} - {1} - {2} - {3}.png'.format(siteName, SEASONS.get(season), calendar.day_name[day - 1], parameter)
    fig.savefig(os.path.join(CHART_DIR, fname))


def plot3DFigure(x, y, z, season, day, siteName):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('Temp Min C')
    ax.set_ylabel('Temp Max C')
    ax.set_zlabel('Optimized In')

    fig.savefig('{0} - {1} - {2}-3D.png'.format(siteName, SEASONS.get(season), calendar.day_name[day - 1]))
