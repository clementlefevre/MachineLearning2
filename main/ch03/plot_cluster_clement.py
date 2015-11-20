
from scipy.stats import norm
import matplotlib.pylab
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy as sp
import pdb
from sklearn.cluster import KMeans




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

num_clusters = 3
seed = 2


#define the scope
rangeScope = np.arange(0,1,0.001)

Xscope,Yscope = np.meshgrid(rangeScope)

#create random dataset of 2 vectors, one for each word
xDataset,yDataset = self.createNormalizedDataSets()


class Plotter():
    
    def main(self):

        #plot the vectors
        self.plot_chart("Vectors")
        print "vectors plotted"


        
        km = KMeans(init='random', n_clusters=num_clusters, verbose=1,
            n_init=1, max_iter=1,
            random_state=seed)
        training = np.array(list(zip(x, y)))
        km.fit(training)
        
        
        
        testing = self.convertMeshGrid(X,Y)

        #create a prediction model from dataset
        model = self.createPredictionModel(x,y , km,testing)



        self.plot_chart(x, y, "Prediction",model)
        print "With Prediction Mesh plotted"

 

    def createNormalizedDataSets(self):
        xw1 = norm(loc=0.3, scale=.15).rvs(20)
        yw1 = norm(loc=0.3, scale=.15).rvs(20)

        xw2 = norm(loc=0.7, scale=.15).rvs(20)
        yw2 = norm(loc=0.7, scale=.15).rvs(20)

        xw3 = norm(loc=0.2, scale=.15).rvs(20)
        yw3 = norm(loc=0.8, scale=.15).rvs(20)

        x = sp.append(sp.append(xw1, xw2), xw3)
        y = sp.append(sp.append(yw1, yw2), yw3)

        return x,y


    def convertMeshGrid(self, X,Y):
        vstack = np.vstack([X.ravel(),Y.ravel()])
        transposed =  vstack.T
        return transposed

    def createPredictionModel(self, x,y ,km,testing, X):
       Z = km.predict(testing)
       return  Z.reshape(X.shape)

    def plot_chart(self,title,C=None, X=None, Y=None):
        plt.figure(num=None, figsize=(8, 6))
        plt.clf()
        
        if C !=None:
            pdb.set_trace()
            plt.pcolormesh(X, Y, C, cmap=plt.cm.Blues)
        plt.scatter(xDataset, yDataset, s=3, color='r')
        plt.title(title)
        plt.xlabel("word1")
        plt.ylabel("word2")
        plt.autoscale(tight=True)
        plt.grid(True, linestyle='-', color='0.75')

        plt.savefig(os.path.join(CHART_DIR, title))


        
if __name__ == "__main__":
    Plotter().main()


