import os
import logging

import scipy as sp

from main.ch02 import utils

logging.basicConfig(format=' %(message)s', level=logging.INFO)

def load_dataset(fileName):
	features = sp.genfromtxt(os.path.join(utils.DATA_DIR,fileName+".tsv"),delimiter="\t")[:,:7]
	labels = sp.genfromtxt(os.path.join(utils.DATA_DIR,fileName+".tsv"),delimiter="\t",dtype=str)[:,7]
	return features, labels


def test_load(features,labels):
	
	assert features[-1,-1] == 5.063
	assert labels[0]=="Kama"
	assert labels[-1]=="Canadian"
	print("labels.shape : {0} , features.shape : {1}".format(labels.shape, features.shape))
	assert labels.shape[0]==210
	assert features.shape[1]== 7
	assert len(labels) == len(features)
	logging.info("load test ok")
	
