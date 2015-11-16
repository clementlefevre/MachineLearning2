import numpy as np
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import load
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',
]


data = load.load_dataset("seeds")
# np.set_printoptions(threshold=np.nan)


def drawFigure_no_normalization(features,labels,neighbors=1,parameters =[]):
	print("Labels before ramaiougana : {0} :").format(labels)
	names = sorted(set(labels))
	labels = np.array([names.index(ell) for ell in labels])

	
	print("Labels after ramaiougana : {0} :").format(labels)

	idX,idY=parameters[0],parameters[1]
	
	print("Xaxis :{0} - Yaxis : {1}").format(idX,idY)

	#define lower and upper limit on both axis (x=area, y =compactness)
	x0,y0 = features[:,idX].min()*0.9, features[:,idY].min()*0.9
	x1,y1 = features[:,idX].max()*1.1, features[:,idY].max()*1.1

	#create a meshgrid resulting of 2 X/Y-Linespaces
	X = np.linspace(x0,x1,1000)
	Y = np.linspace(y0,y1,1000)
	X,Y = np.meshgrid(X,Y)

	#create a predicate resulting of a model 
	classifier = KNeighborsClassifier(n_neighbors= neighbors)
	model = classifier.fit(features[:,(idX,idY)], labels)


	stack =np.vstack([X.ravel(), Y.ravel()]).T
	# print "stack : \n "
	# print stack

	prediction = classifier.predict(stack)

	print "prediction before reshape :\n"
	print prediction

	C = prediction.reshape(X.shape)
	print"predict reshaped :\n \n"
	print C

	
	#create a ListedColormap
	cmap = ListedColormap([(1., 1., 1.), (.2, .2, .2), (.6, .6, .6)])

	#plot the prediction area using pcolormesh
	fig,ax = plt.subplots()
	ax.set_xlim(x0, x1)
	ax.set_ylim(y0, y1)
	ax.set_xlabel(feature_names[0])
	ax.set_ylabel(feature_names[2])
	ax.pcolormesh(X, Y, C, cmap=cmap)


	#iter on the three type of seeds and scatter them.


	fig.tight_layout()
	fig.savefig('figure4sklearn.png')






