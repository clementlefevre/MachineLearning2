__author__ = 'JW'

import utils
import load
import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation


logging.basicConfig(format=' %(message)s', level=logging.INFO)
global features,labels 

class SeedsClassification():

    def __init__(self):
    	global features,labels 
        features, labels = load.load_dataset("seeds")
        load.test_load(features, labels)
        logging.info("data loaded successfully")

    def main(self):
    	global features,labels 
    	self.knn()
    

    #Create a model on knn with One neighbours and and calculate the accuracy on testing data with cross_validation on 5 folds
    def knn(self):
    	global features,labels 
    	classifier = KNeighborsClassifier(n_neighbors=1)
    	kFold = cross_validation.KFold(len(features), n_folds = 5, shuffle = True)
    	accuracy = 0.0
    	globalaccuracy=0.0

    	for training,testing in kFold:
    		model = classifier.fit(features[training], labels[training])
    		prediction = classifier.predict(features[testing])
    		accuracy = np.mean(prediction==labels[testing])
    		globalaccuracy+=accuracy
    		print("accuracy {0:.1%}").format(accuracy)
    	print("globalaccuracy {0:.1%}").format(globalaccuracy/5)
    	
if __name__ == '__main__':
	SeedsClassification().main()
