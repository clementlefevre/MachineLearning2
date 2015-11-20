import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from main.ch03.clustering import StemmedCountVectorizer
import pdb

groups = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'comp.windows.x', 'sci.space']

training_data = sklearn.datasets.fetch_20newsgroups(subset="train",categories=groups)

testing__data = sklearn.datasets.fetch_20newsgroups(subset="test", categories=groups) 

vectorizer = StemmedCountVectorizer()

