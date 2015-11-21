import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb
import scipy as sp
import numpy as np
import main.ch03.clustering

groups = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'comp.windows.x', 'sci.space']

training_data = sklearn.datasets.fetch_20newsgroups(subset="train",categories=groups)

testing_data = sklearn.datasets.fetch_20newsgroups(subset="test", categories=groups) 

class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (main.ch03.clustering.english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=1,stop_words="english", decode_error='ignore')

class NewsGroups():
    def main(self):
        print (training_data.target_names)
        self.searchNearestPost(vectorizer, "StemmedTfidf", True)

    def searchNearestPost(self, vectorizer, methodName, normalized=True):
        pdb.set_trace()
        trainingVector = vectorizer.fit_transform(training_data)
        testingVector = vectorizer.transform(testing_data)

        #let is iterate on all training Vectors and compute the distance with one test post:
        bestDistance = 100
        bestPostIdx = -1

       
        for i in range(trainingVector.shape[0]):
            trainingV = trainingVector.getrow(i).toarray()
            testingV = testingVector.getrow(i).toarray
            d = self.euclidianDistance(trainingV, testingV)
            if d < bestDistance:
                bestDistance = d
                bestPostIdx = i
        print "best d: {0} : {1}".format(bestDistance,training_data.data[bestPostIdxt])



    def euclidianDistance(self, V1,V2, normalized=False):
        if normalized:
            V1 = V1/ sp.linalg.norm(V1)
            V2 = V2/ sp.linalg.norm(V2)
        pdb.set_trace()
        delta = V1-V2
        return sp.linalg.norm(delta)
        
if __name__ == "__main__":
    NewsGroups().main()


