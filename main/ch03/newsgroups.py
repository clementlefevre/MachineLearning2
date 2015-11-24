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

#all_data = sklearn.datasets.fetch_20newsgroups(subset="all", categories=groups)


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (main.ch03.clustering.english_stemmer.stem(w) for w in analyzer(doc))


vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words="english", decode_error='ignore')
#vectorizer = CountVectorizer(min_df=1)


class NewsGroups():
    def main(self):
        print (training_data.target_names)
        self.searchNearestPost(vectorizer, "StemmedTfidf", True)

    def searchNearestPost(self, vectorizer, methodName, normalized=True):
        trainingVector = vectorizer.fit_transform(training_data.data)
        testingVector = vectorizer.transform(testing_data.data)

        voc = vectorizer.vocabulary_
        vocArrayWords = np.array(voc.keys(), dtype='U20')
        vocArrayIdx = np.array(voc.values())

      

        bestDistance = 100
        bestPostIdx = -1

        result = {}

        print "Testing Post : {0} - Training Post {1}".format(testingVector.shape[0],trainingVector.shape[0] )
        totalTestingPost = testingVector.shape[0]
       

        for i in range(0,30):
            print " still to go : {0} / {1}".format( i ,testingVector.shape[0] )
            for j in range(trainingVector.shape[0]):


                testingV = testingVector.getrow(i).toarray()
                trainingV = trainingVector.getrow(j).toarray()

                d = self.euclidianDistance(trainingV, testingV,True)
                result[i] = d

                if d < bestDistance:
                    bestDistance = d
                    bestTestingIdx, bestTrainingIdx = i,j
                    bestTestingVector,bestTrainingVector = testingV, trainingV
           
        testingKeyWords = self.findWordsOnVector(bestTestingVector, vocArrayWords,vocArrayIdx)
        trainingKeyWords = self.findWordsOnVector(bestTrainingVector, vocArrayWords,vocArrayIdx)

        print "Best distance : {0}".format(bestDistance)

        textToPrint =[testing_data.data[bestTestingIdx],training_data.data[bestTrainingIdx],testingKeyWords,trainingKeyWords]

     
        self.saveToFile(textToPrint)

    def findWordsOnVector(self,vector, vocArrayWords,vocArrayIdx):

        nonZeroIdx = np.nonzero(vector)[1]
        listWords=[]

        for idx in nonZeroIdx.tolist():
            idxVoc = np.where(vocArrayIdx==idx)[0]
            word = vocArrayWords[idxVoc][0]
            
            listWords.append(word)

        return listWords
                                                                           
    def euclidianDistance(self, V1, V2, normalized=False):
        if normalized:
            V1 = V1 / sp.linalg.norm(V1)
            V2 = V2 / sp.linalg.norm(V2)
        delta = V1 - V2
        return sp.linalg.norm(delta)

    def findInTrainingVocabulary(self, vocabulary, listWordsIdx):
        listWords=[]
        for k, v in vocabulary.iteritems():
            for foundWord in listWordsIdx:
                if foundWord == v:
                    print "key : {0} for idx {1}".format(k, v)
                    listWords.append(k)
        return listWords



    def saveToFile(self,textToPrint):
        text_file = open("Output.txt", "w")
        text_file.write("Testing Post: %s" % textToPrint[0])
        text_file.write("\n---------------------------------------------------------------\n")
        text_file.write("Testing Post KeyWords: %s" % textToPrint[2])
        text_file.write("\n---------------------------------------------------------------\n")
        text_file.write("Training Post: %s" % textToPrint[1])
        text_file.write("Training Post KeyWords: %s" % textToPrint[3])
        text_file.close() 

if __name__ == "__main__":
    NewsGroups().main()
