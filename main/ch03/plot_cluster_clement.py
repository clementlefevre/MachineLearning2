
import scipy.stats as stats
import matplotlib.pylab
import matplotlib.pyplot as plt
import os
import numpy as np




# [0] #means line 0 of your matrix
# [(0,0)] #means cell at 0,0 of your matrix
# [0:1] #means lines 0 to 1 excluded of your matrix
# [:1] #excluding the first value means all lines until line 1 excluded
# [1:] #excluding the last param mean all lines starting form line 1 included
# [:] #excluding both means all lines
# [::2] #the addition of a second ':' is the sampling. (1 item every 2)
# [::] #exluding it means a sampling of 1
# [:,:] #simply uses a tuple (a single , represents an empty tuple) instead of an index.


CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")
if not os.path.exists(CHART_DIR):
    os.mkdir(CHART_DIR)

print "helloStatic"

class Plotter():
    
    def main(self):
        print "hello"
        #define a set of random data
        dataX, dataY = self.createNormalizedDataSets()

        #plot the data
        self.plot_chart(dataX, dataY, "Vectors")

        #define a meshgrid from -1 to 1

    def createNormalizedDataSets(self):
        xw1 = stats.norm(loc=0.3,scale=.15).rvs(20)
        yw1 = stats.norm(loc=0.3,scale=.15).rvs(20)
        xw2 = stats.norm(loc=0.7,scale=.15).rvs(20)
        yw2 = stats.norm(loc=0.7,scale=.15).rvs(20)
        xw3 = stats.norm(loc=0.8,scale=.15).rvs(20)
        yw3 = stats.norm(loc=0.2,scale=.15).rvs(20)

        dataX = np.append(np.append(xw1,xw2),xw3)
        dataY = np.append(np.append(yw1,yw2),yw3)

        return dataX, dataY


    def plot_chart(self,dataX,dataY,title):
        print "plotting"
        plt.figure(num=None, figsize=(8, 6))
        plt.clf()
        plt.scatter(dataX, dataY, s=1, color='#DAA520')

        plt.title(title)
        plt.xlabel("word1")
        plt.ylabel("word2")
       

        plt.autoscale(tight=True)
       
        plt.grid(True, linestyle='-', color='0.75')
        plt.savefig(os.path.join(CHART_DIR, title))
        print "finished plotting"

       
        
if __name__ == "__main__":
    print "hallo"
    Plotter().main()


