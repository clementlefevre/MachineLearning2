import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pdb
import scipy as sp
import numpy as np
import main.ch03.clustering

groups = ['comp.graphics', 'comp.os.ms-windows.misc',
          'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
          'comp.windows.x', 'sci.space']

groups = ["comp.graphics"]

searchIdx = 0

training_data = sklearn.datasets.fetch_20newsgroups(subset="train", categories=groups)

testing_data = sklearn.datasets.fetch_20newsgroups(subset="test", categories=groups)

all_data = sklearn.datasets.fetch_20newsgroups(subset="all", categories=groups)


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (main.ch03.clustering.english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words="english", decode_error='ignore')
vectorizer = CountVectorizer(min_df=1)


class NewsGroups():
    def main(self):
        print (training_data.target_names)
        self.searchNearestPost(vectorizer, "StemmedTfidf", True)

    def searchNearestPost(self, vectorizer, methodName, normalized=True):
        trainingVector = vectorizer.fit_transform(training_data.data)
        testingVector = vectorizer.transform([testing_data.data[searchIdx]])

        print "Testing data : {0}".format(testing_data.data[searchIdx])
        voc = vectorizer.vocabulary_

        

        vocArrayWords = np.array(voc.keys(), dtype='U20')
        vocArrayIdx = np.array(voc.values())



        testVectorArray = testingVector.getrow(0).toarray().T[:, 0]

        nonZerostest = np.nonzero(testVectorArray)

        nonZeroList = nonZerostest[0].tolist()
        pdb.set_trace()


        print nonZeroList

        self.findInTrainingVocabulary(voc,nonZeroList )



        # let is iterate on all training Vectors and compute the distance with one test post:
        bestDistance = 100
        bestPostIdx = -1

        testingV = testingVector.getrow(searchIdx).toarray()

        result = {}
        for i in range(trainingVector.shape[0]):
            trainingV = trainingVector.getrow(i).toarray()

        d = self.euclidianDistance(trainingV, testingV)
        result[i] = d

        if d < bestDistance:
            bestDistance = d
        bestPostIdx = i

        print "for post : {0} : \n" \
              "--------------------------------------" \
              "best d: {1} \n" \
              "------------------------------------------: {2} - {3}".format(testing_data.data[140], bestDistance,
                                                                             bestPostIdx,
                                                                             training_data.data[bestPostIdx]
                                                                             )

    def euclidianDistance(self, V1, V2, normalized=False):
        if normalized:
            V1 = V1 / sp.linalg.norm(V1)
            V2 = V2 / sp.linalg.norm(V2)
        # pdb.set_trace()
        delta = V1 - V2
        return sp.linalg.norm(delta)

    def findInTrainingVocabulary(self,vocabulary, listWordsIdx):
        for k,v in vocabulary.iteritems():
            for foundWord in listWordsIdx:
                if foundWord == v:
                    print "key : {0} for idx {1}".format(k,v)

if __name__ == "__main__":
    NewsGroups().main()
