__author__ = 'JW'

import nltk.stem
import scipy as sp
import os
from main import tools
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/toy")


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
        posts = [open(os.path.join(DATA_DIR, f)).read() for f in
                 os.listdir(DATA_DIR)]
        vectorizer = CountVectorizer(min_df=1)
        X_train = vectorizer.fit_transform(posts)

        new_post = "imaging databases"
        new_postVector = vectorizer.transform([new_post])

        self.getEuclidianDistances(X_train, new_postVector)

    def getEuclidianDistances(self, training, testing):
        for i in range(0, 5):
            trainingVector = training.getrow(i)
            testingVector = testing
            delta = trainingVector - testingVector

            distance = sp.linalg.norm(delta.toarray())
            print distance


if __name__ == "__main__":
    Clustering().main()
