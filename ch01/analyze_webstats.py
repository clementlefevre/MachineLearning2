import os
import scipy as sp
from utils import DATA_DIR, CHART_DIR
import matplotlib.pyplot as plt
import traceback
import logging

logging.basicConfig(format=' %(message)s', level=logging.INFO)

# all examples will have three classes in this file
colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-', '--', ':', '-']


class AnalyzeWebStats():
    global x
    global y
    global data
    global f

    def main(self):
        self.setdata()
        global x
        global y
        global f
        self.initData()
        # self.inflectData()
        self.trainingAndTesting()
        self.findMaxCapacity()

    def initData(self):
        global x, y
        self.plot_models(x, y, None, "1400_01_01.png")
        models = self.getModels(x, y)
        self.plot_models(x, y, models, "1400_01_02.png")

    def getModels(self, x, y, fName=None):
        modelList = []
        rangeDeg = [1, 10, 53]
        for i in rangeDeg:
            modelList.append(self.getPoly(x, y, i))
        return modelList

    def inflectData(self):
        global x, y

        modelsBefore = self.getModels(xa, ya)
        modelAfterInflection = self.getModels(xb, yb)
        models = modelsBefore + modelAfterInflection
        self.plot_models(x, y, models, "1400_01_03.png", mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
                         ymax=10000, xmin=0 * 7 * 24)
        self.plot_models(x, y, modelAfterInflection, "1400_01_04.png", mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
                         ymax=10000, xmin=0 * 7 * 24)

    def getPoly(self, x, y, deg):

        f = self.polyfy(x, y, deg)
        fp = sp.poly1d(f)
        error = self.error(fp, x, y)
        # logging.info(" error : %i --- function deg : %i -- function : %s ", error, deg, fp)
        return fp

    def setdata(self):
        print('prepare_data')
        global data
        data = sp.genfromtxt(os.path.join(DATA_DIR, "web_traffic.tsv"), delimiter="\t")
        self.setXY()

    def setXY(self):
        global x, y, xa, xb, ya, yb
        test = data
        x = data[:, 0]
        y = data[:, 1]

        x = x[~sp.isnan(y)]
        y = y[~sp.isnan(y)]

        inflection = 3.5 * 7 * 24
        xa = x[:inflection]
        ya = y[:inflection]
        xb = x[inflection:]
        yb = y[inflection:]

    def error(self, f, x, y):
        return sp.sum((f(x) - y) ** 2)

    def polyfy(self, x, y, deg):
        fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, deg, full=True)

        return fp1

    def trainingAndTesting(self):
        global train
        global x, y, xa, xb, ya, yb
        # separating training from testing data
        frac = 0.3
        split_idx = int(frac * len(xb))
        rangeX = range(len(xb))
        listX = list(rangeX)
        logging.info("delta : %i", len(set(rangeX).difference(listX)))

        shuffled = sp.random.permutation(list(range(len(xb))))
        test = sorted(shuffled[:split_idx])

        train = sorted(shuffled[split_idx:])
        fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train], 1))
        fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
        print("fbt2(x)= \n%s" % fbt2)
        print("fbt2(x)-100,000= \n%s" % (fbt2 - 100000))
        print("fbt2(x)= \n%s" % (fbt2))
        fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train], 3))
        fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train], 10))
        fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train], 100))

        print("Test errors for only the time after inflection point")
        for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
            print("Error d=%i: %f" % (f.order, self.error(f, xb[test], yb[test])))

    def findMaxCapacity(self):
        global train
        fbt2 = sp.poly1d(sp.polyfit(xb, yb, 2))
        from scipy.optimize import fsolve

        timeToMax = sp.optimize.fsolve(fbt2 - 100000, x0=743)
        logging.info("Time to 100.000 views with full sample = %f", timeToMax / 7 / 24)

        fbt2Train = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))

        fLine = sp.poly1d([0]) + 100000

        timeToMaxTrain = fsolve(fbt2Train - 100000, x0=743)
        logging.info("Time to 100.000 views with train sample only  = %f", timeToMaxTrain / 7 / 24)

        self.plot_models(xb, yb, [fbt2, fbt2Train, fLine], "1400_01_05.png",
                         mx=sp.linspace(0 * 7 * 24, timeToMaxTrain, 100),
                         ymax=120000, xmin=743)

    # plot input data
    def plot_models(self, x, y, models, fname, mx=None, ymax=None, xmin=None):
        plt.figure(num=None, figsize=(8, 6))
        plt.clf()
        plt.scatter(x, y, s=1, color='#DAA520')

        plt.title("Web traffic over the last month")
        plt.xlabel("Time")
        plt.ylabel("Hits/hour")
        plt.xticks(
            [w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])

        if models:
            if mx is None:
                mx = sp.linspace(0, x[-1], 1000)

            for model, style, color in zip(models, linestyles, colors):
                plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

            plt.legend(["d=%i" % m.order for m in models], loc="upper left")

        plt.autoscale(tight=True)
        plt.ylim(ymin=0)
        if ymax:
            plt.ylim(ymax=ymax)
        if xmin:
            plt.xlim(xmin=xmin)
        plt.grid(True, linestyle='-', color='0.75')
        plt.savefig(os.path.join(CHART_DIR, fname))


if __name__ == "__main__":
    AnalyzeWebStats().main()
