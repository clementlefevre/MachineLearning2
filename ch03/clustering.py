__author__ = 'JW'

from sklearn.feature_extraction.text import CountVectorizer


class Clustering():
    def main(self):
        print "hello"
        print __package__
        self.test()

    def test(self):
        vectorizer = CountVectorizer(min_df=1)
        content = ["How to format my hard disk", " Hard disk format problems"]
        X = vectorizer.fit_transform(content)
        features = vectorizer.get_feature_names()
        print features
        print (X.toarray().transpose())


if __name__ == "__main__":
    Clustering().main()
