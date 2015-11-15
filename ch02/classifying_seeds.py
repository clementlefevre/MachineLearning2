__author__ = 'JW'

import utils
import load
import logging
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

    	for training,testing in kFold:
    		model = classifier.fit(features[training], labels[training])
    		prediction = classifier.predict(features[testing])
    		print ("testing : {0} \n Prediction : {1}".format(features[testing], prediction))
if __name__ == '__main__':
	SeedsClassification().main()
