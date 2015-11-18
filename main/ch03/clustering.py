from main import tools
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
import scipy as sp
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/toy")

posts = [open(os.path.join(DATA_DIR, f)).read() for f in
         os.listdir(DATA_DIR)]
new_post = "imaging databases"


class Clustering():
    def main(self):
        # self.test()
        self.analyze_posts()

    def test(self):
        vectorizer = CountVectorizer(min_df=1)
        content = ["How to format my hard disk", " Hard disk format problems"]
        X = vectorizer.fit_transform(content)
        features = vectorizer.get_feature_names()
        print features
        print (X.toarray().transpose())

    def analyze_posts(self):

        vectorizerCount = CountVectorizer(min_df=1)
        self.searchNearestPost(vectorizerCount, "CountVectorizer", False)
        self.searchNearestPost(vectorizerCount, "CountVectorizer", True)

    def getEuclidianDistances(self, training, testing, normalized=False):
        if normalized:
            training = training.toarray()/sp.linalg.norm(training.toarray())
            testing  = testing.toarray()/sp.linalg.norm(testing.toarray())
            delta = training - testing
        else:
             delta = (training - testing).toarray()
        distance = sp.linalg.norm(delta)
        return distance

    def searchNearestPost(self, vectorizer, vectorizedMethod, normalized=False):
        X_train = vectorizer.fit_transform(posts)
        X_testing = vectorizer.transform([new_post])
        best_distance = 100
        bestPostIdx = 0

        for i in range(X_train.shape[0]):
            distance = self.getEuclidianDistances(X_train.getrow(i), X_testing, normalized)
            if distance<best_distance:
                best_distance = distance
                bestPostIdx= i
            print "Distance : {1} : {0} - ".format(posts[i],distance)
        print "With method {1} {2} -> Closest Post : {0}".format(posts[bestPostIdx],vectorizedMethod, normalized)


if __name__ == "__main__":
    Clustering().main()
