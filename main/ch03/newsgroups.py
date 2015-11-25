import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

import pdb
import scipy as sp
import numpy as np
import main.ch03.clustering

groups = ['comp.graphics', 'comp.os.ms-windows.misc',
          'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
          'comp.windows.x', 'sci.space']

#groups = ["comp.graphics"]


num_clusters = 50

training_data = sklearn.datasets.fetch_20newsgroups(subset="train", categories=groups)
testing_data = sklearn.datasets.fetch_20newsgroups(subset="test", categories=groups)

#all_data = sklearn.datasets.fetch_20newsgroups(subset="all", categories=groups)


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (main.ch03.clustering.english_stemmer.stem(w) for w in analyzer(doc))


vectorizer = StemmedTfidfVectorizer(min_df=10,max_df=0.5, stop_words="english", decode_error='ignore')
#vectorizer = CountVectorizer(min_df=1)

trainingVector = vectorizer.fit_transform(training_data.data)

new_post ="Disk drive problems. Hi, I have a problem with my hard disk.After 1 year it is working only sporadically now.I tried to format it, but now it doesn't boot any more.Any ideas? Thanks"

def findByText(text):
    result =[]
    for i,post in enumerate(training_data.data):
        if text in post:
            result.append([i,post])
    return result

class NewsGroups():
    def main(self):
        print (training_data.target_names)
        samples, features = trainingVector.shape
        print ("Number of Samples : {0} - Number of Features :{1}".format(samples, features))
        #self.searchNearestPost(vectorizer, "StemmedTfidf", True)
        self.clusterKmeans(vectorizer,"StemmedTfidf", True)

    def clusterKmeans(self, vectorized, method, normalized=True):
        km = KMeans(n_clusters=num_clusters, init='random', n_init=1,
verbose=1, random_state=3)
        km.fit(trainingVector)
        print km.labels_
        new_post_vectorized = vectorizer.transform([new_post])
        prediction = km.predict(new_post_vectorized)[0]
        predictionIdx = list(np.where(km.labels_==prediction)[0])
        result = self.sortOnDistance(predictionIdx, new_post_vectorized)

        
        self.saveClusterPostsToFile(predictionIdx)




    def searchNearestPost(self, vectorizer, methodName, normalized=True):
        num_samples, num_features = trainingVector.shape
        pdb.set_trace()
        testingVector = vectorizer.transform(testing_data.data)
        voc = vectorizer.vocabulary_
        vocArrayWords = np.array(voc.keys(), dtype='U20')
        vocArrayIdx = np.array(voc.values())
        bestDistance = 100
        bestPostIdx = -1
        resultIdx = []
        print "Testing Post : {0} - Training Post {1}".format(testingVector.shape[0],trainingVector.shape[0] )
        totalTestingPost = testingVector.shape[0]
        for i in range(0,10):
        #for i in range(testingVector.shape[0]):
            print " still to go : {0:.0f}% ".format( float(i)/totalTestingPost*100)
            for j in range(trainingVector.shape[0]):
                testingV = testingVector.getrow(i).toarray()
                trainingV = trainingVector.getrow(j).toarray()
                d = self.euclidianDistance(trainingV, testingV,True)
                resultIdx.append([i,j,d])
                if d < bestDistance:
                    bestDistance = d
                    bestTestingIdx, bestTrainingIdx = i,j
                    bestTestingVector,bestTrainingVector = testingV, trainingV
           
        testingKeyWords = self.findWordsOnVector(bestTestingVector, vocArrayWords,vocArrayIdx)
        trainingKeyWords = self.findWordsOnVector(bestTrainingVector, vocArrayWords,vocArrayIdx)
        sortedResult = resultIdx.sort(key=lambda x: x[2])
        textToPrint =[testing_data.data[bestTestingIdx],training_data.data[bestTrainingIdx],testingKeyWords,trainingKeyWords,resultIdx]
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

    def sortOnDistance(self, predictionIdx, new_post_vectorized):
        ranking=[]

        for idx in predictionIdx:
            d = sp.linalg.norm((trainingVector.getrow(idx)-new_post_vectorized).toarray())
            ranking.append([idx,d])
        ranking.sort(key=lambda x: x[1])
        pdb.set_trace()
        return ranking



    def saveClusterPostsToFile(self, listIdx):
        text_file = open("OutputPostsCluster.txt", "w")

        for idx in listIdx:

            text_file.write("----------------------------------\n Training Post Idx : {0} \n".format(idx))
            text_file.write("Testing Post: %s \n" % training_data.data[idx].encode("utf-8"))
        

        text_file.close()



    def saveToFile(self,textToPrint):
        text_file = open("Output.txt", "w")
        text_file.write("Testing Post: %s \n" % textToPrint[0])
        text_file.write("\n---------------------------------------------------------------\n")
        text_file.write("Testing Post KeyWords: %s" % textToPrint[2])
        text_file.write("\n---------------------------------------------------------------\n")
        text_file.write("Training Post: %s \n" % textToPrint[1])
        text_file.write("Training Post KeyWords: %s \n" % textToPrint[3])
        results = textToPrint[4][:10]

        for i,j,k in results:
            text_file.write("testingIdx : {0} - trainingIdx : {1} - distance {2} \n".format(i,j,k))

        text_file.close() 

if __name__ == "__main__":
    NewsGroups().main()
